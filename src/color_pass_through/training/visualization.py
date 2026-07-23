from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


def _rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def _sample_id(meta: Any, fallback: int) -> str:
    if isinstance(meta, dict):
        value = meta.get("id", fallback)
        if isinstance(value, (list, tuple)):
            value = value[0]
        return str(value)
    return str(fallback)


def projector_error_heatmap(
    prediction: torch.Tensor,
    target: torch.Tensor,
    vmax_abs: float = 0.08,
) -> Image.Image:
    """Reproduce the legacy fixed-scale per-pixel RGB RMSE heatmap."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"Expected matching CHW tensors, got {prediction.shape} and {target.shape}"
        )
    if vmax_abs <= 0:
        raise ValueError(f"vmax_abs must be positive, got {vmax_abs}")
    rmse = torch.sqrt((prediction.float() - target.float()).square().mean(dim=0))
    normalized = (rmse / vmax_abs).clamp(0.0, 1.0)
    positions = torch.tensor(
        [0.00, 0.25, 0.50, 0.75, 1.00],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    colors = torch.tensor(
        [
            [0.00, 0.00, 0.35],
            [0.00, 0.50, 1.00],
            [0.50, 1.00, 0.50],
            [1.00, 1.00, 0.00],
            [1.00, 0.00, 0.00],
        ],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    samples = torch.linspace(
        0.0, 1.0, 256, dtype=normalized.dtype, device=normalized.device
    )
    indices = torch.bucketize(samples, positions) - 1
    indices = indices.clamp(0, len(positions) - 2)
    weights = (
        (samples - positions[indices])
        / (positions[indices + 1] - positions[indices] + 1e-12)
    ).unsqueeze(-1)
    lookup = colors[indices] * (1.0 - weights) + colors[indices + 1] * weights
    lookup_indices = (normalized * 255.0 + 0.5).long().clamp(0, 255)
    heatmap = lookup[lookup_indices]
    return _rgb_image(heatmap.permute(2, 0, 1))


class ProjectorVisualizer:
    """Save projector input/prediction/target/error image groups."""

    def __init__(self, output: str | Path) -> None:
        self.output = Path(output)

    def save(
        self,
        phase: str,
        batch: dict[str, Any],
        prediction: torch.Tensor,
        batch_index: int,
    ) -> None:
        sample_id = _sample_id(batch.get("Meta"), batch_index)
        destination = self.output / phase
        destination.mkdir(parents=True, exist_ok=True)

        source_tensor = batch["Input"][0]
        prediction_tensor = prediction[0]
        target_tensor = batch["Target"][0]
        error_tensor = (prediction_tensor - target_tensor).abs()
        mae = float(error_tensor.detach().float().mean().cpu())
        rmse_mean = float(
            torch.sqrt(
                (prediction_tensor.float() - target_tensor.float()).square().mean(dim=0)
            )
            .mean()
            .cpu()
        )

        images = {
            "input": _rgb_image(source_tensor),
            "prediction": _rgb_image(prediction_tensor),
            "target": _rgb_image(target_tensor),
            "diff": projector_error_heatmap(prediction_tensor, target_tensor),
        }
        for name, image in images.items():
            image.save(destination / f"{sample_id}_{name}.png")

        labels = (
            "Input",
            "Prediction",
            "Ground Truth",
            f"RGB RMSE Heatmap  vmax=0.08  mean={rmse_mean:.6f}",
        )
        width, height = images["input"].size
        label_height = 24
        comparison = Image.new("RGB", (4 * width, height + label_height), "white")
        draw = ImageDraw.Draw(comparison)
        for index, (name, label) in enumerate(zip(images, labels)):
            left = index * width
            comparison.paste(images[name], (left, label_height))
            draw.text((left + 6, 5), label, fill="black")
        comparison.save(destination / f"{sample_id}_comparison.png")

        manifest_path = destination / "manifest.json"
        records = []
        if manifest_path.is_file():
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in records if record["id"] != sample_id]
        records.append(
            {
                "id": sample_id,
                "mae": mae,
                "pixel_rgb_rmse_mean": rmse_mean,
                "heatmap_vmax": 0.08,
            }
        )
        records.sort(key=lambda record: record["id"])
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


class ProjectorLUTExporter:
    """Export the legacy projector 3D CUBE and RGB-vector quiver plot."""

    def __init__(
        self,
        output: str | Path,
        size: int = 33,
        every_n_epochs: int = 10,
        chunk_pixels: int = 262_144,
        vis_subsample: int = 3,
        vis_vec_scale: float = 1.0,
        vis_arrow_length: float = 0.08,
    ) -> None:
        self.output = Path(output)
        self.size = size
        self.every_n_epochs = every_n_epochs
        self.chunk_pixels = chunk_pixels
        self.vis_subsample = vis_subsample
        self.vis_vec_scale = vis_vec_scale
        self.vis_arrow_length = vis_arrow_length

    @torch.inference_mode()
    def maybe_export(self, model: torch.nn.Module, epoch: int) -> bool:
        if (epoch + 1) % self.every_n_epochs != 0:
            return False
        parameter = next(model.parameters())
        device = parameter.device
        line = torch.linspace(0.0, 1.0, self.size, device=device)
        red, green, blue = torch.meshgrid(line, line, line, indexing="ij")
        grid = torch.stack([red, green, blue], dim=-1).reshape(-1, 3)
        outputs = []
        chunk = min(self.chunk_pixels, len(grid))
        for start in range(0, len(grid), chunk):
            outputs.append(model.map_rgb_samples(grid[start : start + chunk]))
        lut = (
            torch.cat(outputs, dim=0)
            .clamp(0.0, 1.0)
            .reshape(self.size, self.size, self.size, 3)
            .cpu()
            .numpy()
        )

        self.output.mkdir(parents=True, exist_ok=True)
        stem = f"epoch{epoch:03d}_size{self.size}"
        cube_path = self.output / f"{stem}.cube"
        with cube_path.open("w", encoding="utf-8") as handle:
            handle.write("# Generated by NGPColorMap (periodic export)\n")
            handle.write(f"LUT_3D_SIZE {self.size}\n")
            handle.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            handle.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            for red_value, green_value, blue_value in lut.reshape(-1, 3):
                handle.write(
                    f"{red_value:.6f} {green_value:.6f} {blue_value:.6f}\n"
                )
        self._save_quiver(lut, self.output / f"{stem}_vis.png")
        return True

    def _save_quiver(self, lut: Any, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        line = np.linspace(0.0, 1.0, self.size, dtype=np.float32)
        red, green, blue = np.meshgrid(line, line, line, indexing="ij")
        source = np.stack([red, green, blue], axis=-1)
        step = max(1, int(self.vis_subsample))
        selection = slice(0, self.size, step)
        source = source[selection, selection, selection].reshape(-1, 3)
        destination = lut[selection, selection, selection].reshape(-1, 3)
        vectors = (destination - source) * self.vis_vec_scale
        colors = np.clip(source, 0.0, 1.0)

        figure = plt.figure(figsize=(7, 5))
        axes = figure.add_subplot(111, projection="3d")
        corners = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for first, second in edges:
            x_values, y_values, z_values = corners[[first, second]].T
            axes.plot(x_values, y_values, z_values, color=(0, 0, 0, 0.4), lw=1.0)
        axes.quiver(
            source[:, 0],
            source[:, 1],
            source[:, 2],
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            length=self.vis_arrow_length,
            normalize=False,
            colors=colors,
            linewidths=0.7,
        )
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_zlim(0, 1)
        axes.set_xlabel("R")
        axes.set_ylabel("G")
        axes.set_zlabel("B")
        axes.set_box_aspect((1, 1, 1))
        figure.savefig(path, dpi=180)
        plt.close(figure)


def _gray_abs_image(tensor: torch.Tensor) -> Image.Image:
    gray = tensor.detach().float().abs().clamp(0.0, 1.0)
    if gray.ndim == 3:
        gray = gray[0]
    return _rgb_image(gray.unsqueeze(0).repeat(3, 1, 1))


def _psnr(mse: torch.Tensor, peak: torch.Tensor | float = 1.0) -> float:
    mse = mse.detach().float().clamp_min(1e-12)
    peak_tensor = torch.as_tensor(peak, dtype=mse.dtype, device=mse.device)
    value = 20.0 * torch.log10(peak_tensor.clamp_min(1e-12)) - 10.0 * torch.log10(mse)
    return float(value.cpu())


class CameraNullVisualizer:
    """Export the migrated camera-null validation RGB/PC diagnostics."""

    def __init__(
        self,
        output: str | Path,
        max_samples: int = 182,
        every_n_epochs: int = 10,
        save_csv: bool = True,
    ) -> None:
        self.output = Path(output)
        self.max_samples = max_samples
        self.every_n_epochs = every_n_epochs
        self.save_csv = save_csv
        self.active = False
        self.epoch = 0
        self.rows: list[dict[str, Any]] = []

    def start_epoch(self, epoch: int, sanity_checking: bool) -> None:
        self.epoch = epoch
        self.rows = []
        self.active = (
            not sanity_checking and (epoch + 1) % self.every_n_epochs == 0
        )

    def save_batch(
        self,
        batch: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        predicted_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        predicted_coefficient: torch.Tensor,
        target_coefficient: torch.Tensor,
        batch_index: int,
    ) -> None:
        if not self.active:
            return
        sample_id = _sample_id(batch.get("Meta"), batch_index)
        spectrum_mse = (prediction[0].float() - target[0].float()).square().mean()
        rgb_mse = (predicted_rgb[0].float() - target_rgb[0].float()).square().mean()
        coefficient_difference = (
            predicted_coefficient[0].float() - target_coefficient[0].float()
        )
        coefficient_mse = coefficient_difference.square().mean()
        coefficient_peak = target_coefficient[0].float().abs().amax().clamp_min(1e-6)
        self.rows.append(
            {
                "idx": len(self.rows),
                "base_name": sample_id,
                "H": int(prediction.shape[-2]),
                "W": int(prediction.shape[-1]),
                "psnr_spec": round(_psnr(spectrum_mse), 4),
                "psnr_coef": round(_psnr(coefficient_mse, coefficient_peak), 4),
                "psnr_rgb": round(_psnr(rgb_mse), 4),
            }
        )
        if batch_index >= self.max_samples:
            return

        destination = self.output / f"epoch_{self.epoch:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        gt_rgb = _rgb_image(target_rgb[0])
        pred_rgb = _rgb_image(predicted_rgb[0])
        diff_rgb = _rgb_image((predicted_rgb[0] - target_rgb[0]).abs())
        rgb_images = {
            "rangeRGB_GT": gt_rgb,
            "rangeRGB_PRED": pred_rgb,
            "rangeRGB_DIFF": diff_rgb,
        }
        for name, image in rgb_images.items():
            image.save(destination / f"{sample_id}_{name}.png", compress_level=3)
        rgb_strip = Image.new("RGB", (3 * gt_rgb.width, gt_rgb.height))
        for index, image in enumerate(rgb_images.values()):
            rgb_strip.paste(image, (index * gt_rgb.width, 0))
        rgb_strip.save(
            destination / f"{sample_id}_rangeRGB_GT_PRED_DIFF.png",
            compress_level=3,
        )

        coefficient_rows = []
        for component in range(predicted_coefficient.shape[1]):
            gt = _gray_abs_image(target_coefficient[0, component])
            pred = _gray_abs_image(predicted_coefficient[0, component])
            diff = _gray_abs_image(coefficient_difference[component])
            images = {"GT_abs": gt, "PRED_abs": pred, "DIFF_abs": diff}
            prefix = f"{sample_id}_coef{component + 1}"
            for name, image in images.items():
                image.save(destination / f"{prefix}_{name}.png", compress_level=3)
            strip = Image.new("RGB", (3 * gt.width, gt.height))
            for index, image in enumerate(images.values()):
                strip.paste(image, (index * gt.width, 0))
            strip.save(
                destination / f"{prefix}_GT_PRED_DIFF_abs.png",
                compress_level=3,
            )
            coefficient_rows.append(strip)
        if len(coefficient_rows) > 1:
            stacked = Image.new(
                "RGB",
                (coefficient_rows[0].width, len(coefficient_rows) * coefficient_rows[0].height),
            )
            for index, image in enumerate(coefficient_rows):
                stacked.paste(image, (0, index * image.height))
            stacked.save(
                destination / f"{sample_id}_coefALL_GT_PRED_DIFF_abs.png",
                compress_level=3,
            )

    def finish_epoch(self) -> None:
        if not self.active or not self.save_csv:
            self.rows = []
            return
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"val_results_epoch{self.epoch:03d}.csv"
        fields = [
            "idx",
            "base_name",
            "H",
            "W",
            "psnr_spec",
            "psnr_coef",
            "psnr_rgb",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        self.rows = []
