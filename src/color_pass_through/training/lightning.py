from __future__ import annotations

from typing import Any

import lightning.pytorch as L
import torch
import torch.nn.functional as F

from color_pass_through.models.camera_null import CameraNullModel
from color_pass_through.models.projector import CameraDisplayProjector
from color_pass_through.training.visualization import (
    CameraNullVisualizer,
    ProjectorLUTExporter,
    ProjectorVisualizer,
)


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
    return (-10.0 * torch.log10(mse.clamp_min(1e-12))).mean()


class ProjectorTask(L.LightningModule):
    def __init__(
        self,
        model: CameraDisplayProjector,
        learning_rate: float = 1e-3,
        scheduler_step: int = 10,
        scheduler_gamma: float = 0.5,
        visualizer: ProjectorVisualizer | None = None,
        visualize_samples: int = 0,
        visualize_every_n_epochs: int = 5,
        save_test_predictions: bool = False,
        lut_exporter: ProjectorLUTExporter | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.visualizer = visualizer
        self.visualize_samples = visualize_samples
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.save_test_predictions = save_test_predictions
        self.lut_exporter = lut_exporter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, batch: dict[str, Any], stage: str, batch_idx: int
    ) -> torch.Tensor:
        prediction = self(batch["Input"])
        loss = F.l1_loss(prediction, batch["Target"])
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(
            f"{stage}/psnr",
            _psnr(prediction, batch["Target"]),
            prog_bar=True,
            on_epoch=True,
        )
        if (
            self.visualizer is not None
            and batch_idx < self.visualize_samples
            and self.trainer.is_global_zero
        ):
            if (
                stage == "val"
                and not self.trainer.sanity_checking
                and self.current_epoch % self.visualize_every_n_epochs == 0
            ):
                self.visualizer.save(
                    f"epoch_{self.current_epoch:03d}", batch, prediction, batch_idx
                )
            elif stage == "test" and self.save_test_predictions:
                self.visualizer.save("test_best", batch, prediction, batch_idx)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "val", batch_idx)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "test", batch_idx)

    def on_validation_epoch_end(self) -> None:
        if (
            self.lut_exporter is not None
            and self.trainer.is_global_zero
            and not self.trainer.sanity_checking
        ):
            self.lut_exporter.maybe_export(self.model, int(self.current_epoch))

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CameraNullTask(L.LightningModule):
    def __init__(
        self,
        model: CameraNullModel,
        learning_rate: float = 4e-4,
        coefficient_weight: float = 1.0,
        tv_weight: float = 0.01,
        spectrum_weight: float = 1.0,
        visualizer: CameraNullVisualizer | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.coefficient_weight = coefficient_weight
        self.tv_weight = tv_weight
        self.spectrum_weight = spectrum_weight
        self.visualizer = visualizer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, batch: dict[str, Any], stage: str, batch_idx: int
    ) -> torch.Tensor:
        prediction = self(batch["Input"])
        losses = self.model.losses(
            prediction,
            batch["Target"],
            self.coefficient_weight,
            self.tv_weight,
            self.spectrum_weight,
        )
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, prog_bar=name == "loss", on_epoch=True)
        if stage != "train":
            target = batch["Target"]
            predicted_rgb = self.model.spectrum_to_camera_rgb(prediction)
            target_rgb = self.model.spectrum_to_camera_rgb(target)
            predicted_coefficient = self.model.decompose(prediction)[2]
            target_coefficient = self.model.decompose(target)[2]
            self.log(f"{stage}/psnr_spec", _psnr(prediction, target), on_epoch=True)
            self.log(f"{stage}/psnr_rgb", _psnr(predicted_rgb, target_rgb), on_epoch=True)
            coefficient_mse = (
                predicted_coefficient - target_coefficient
            ).square().flatten(1).mean(1)
            coefficient_peak = (
                target_coefficient.abs().flatten(1).amax(1).clamp_min(1e-6)
            )
            coefficient_psnr = (
                20.0 * torch.log10(coefficient_peak)
                - 10.0 * torch.log10(coefficient_mse.clamp_min(1e-12))
            ).mean()
            self.log(f"{stage}/psnr_coef", coefficient_psnr, on_epoch=True)
            if (
                stage == "val"
                and self.visualizer is not None
                and self.trainer.is_global_zero
            ):
                self.visualizer.save_batch(
                    batch,
                    prediction,
                    target,
                    predicted_rgb,
                    target_rgb,
                    predicted_coefficient,
                    target_coefficient,
                    batch_idx,
                )
        return losses["loss"]

    def on_validation_epoch_start(self) -> None:
        if self.visualizer is not None:
            self.visualizer.start_epoch(
                int(self.current_epoch), bool(self.trainer.sanity_checking)
            )

    def on_validation_epoch_end(self) -> None:
        if self.visualizer is not None and self.trainer.is_global_zero:
            self.visualizer.finish_epoch()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "val", batch_idx)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "test", batch_idx)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
