from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .mst_plus_plus import MST_Plus_Plus


def load_camera_sensitivity(
    path: str | Path, wavelengths: np.ndarray
) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False)
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(f"Expected CSS shape (N,4+), got {array.shape}")
    available = array[:, 0].astype(np.int32)
    rows: list[int] = []
    for wavelength in wavelengths.astype(np.int32):
        hits = np.flatnonzero(available == wavelength)
        if len(hits) != 1:
            raise ValueError(f"CSS does not uniquely contain {wavelength} nm")
        rows.append(int(hits[0]))
    return array[rows, 1:4].astype(np.float32)


class CameraNullModel(nn.Module):
    """Predict a 31-band spectrum and project its camera-null component."""

    def __init__(
        self,
        camera_sensitivity: str | Path,
        pca_components: str | Path,
        num_components: int = 1,
        backbone: nn.Module | None = None,
        wavelength_start: int = 400,
        wavelength_step: int = 10,
        bands: int = 31,
    ) -> None:
        super().__init__()
        if not 1 <= num_components <= 3:
            raise ValueError("num_components must be between 1 and 3")
        wavelengths = np.arange(
            wavelength_start, wavelength_start + wavelength_step * bands, wavelength_step
        )
        sensitivity = load_camera_sensitivity(camera_sensitivity, wavelengths)
        components = np.load(Path(pca_components), allow_pickle=False)
        if components.ndim != 2 or components.shape[1] != bands:
            raise ValueError(
                f"Expected PCA components shape (K,{bands}), got {components.shape}"
            )
        matrix = torch.from_numpy(sensitivity)
        projection = matrix @ torch.linalg.pinv(matrix)
        self.register_buffer("camera_sensitivity", matrix)
        self.register_buffer("range_projection", projection)
        self.register_buffer(
            "null_components",
            torch.from_numpy(components[:num_components].astype(np.float32)),
        )
        self.backbone = backbone or MST_Plus_Plus(
            in_channels=3, out_channels=bands, n_feat=bands, stage=3
        )

    def forward(self, camera_rgb: torch.Tensor) -> torch.Tensor:
        return self.backbone(camera_rgb)

    def decompose(
        self, spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        range_spectrum = torch.einsum(
            "ij,bjhw->bihw", self.range_projection, spectrum
        )
        null_spectrum = spectrum - range_spectrum
        coefficient = torch.einsum(
            "ki,bihw->bkhw", self.null_components, null_spectrum
        )
        return range_spectrum, null_spectrum, coefficient

    def predict_coefficient(self, camera_rgb: torch.Tensor) -> torch.Tensor:
        return self.decompose(self(camera_rgb))[2]

    def spectrum_to_camera_rgb(self, spectrum: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "ic,bihw->bchw", self.camera_sensitivity, spectrum
        ).clamp(0.0, 1.0)

    def losses(
        self,
        predicted_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
        coefficient_weight: float = 1.0,
        tv_weight: float = 0.01,
        spectrum_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        predicted_coefficient = self.decompose(predicted_spectrum)[2]
        target_coefficient = self.decompose(target_spectrum)[2]
        coefficient = F.l1_loss(predicted_coefficient, target_coefficient)
        dh = predicted_coefficient[:, :, 1:] - predicted_coefficient[:, :, :-1]
        dw = predicted_coefficient[:, :, :, 1:] - predicted_coefficient[:, :, :, :-1]
        tv = dh.abs().mean() + dw.abs().mean()
        spectrum = F.l1_loss(predicted_spectrum, target_spectrum)
        total = (
            coefficient_weight * coefficient
            + tv_weight * tv
            + spectrum_weight * spectrum
        )
        return {
            "loss": total,
            "coefficient_l1": coefficient,
            "coefficient_tv": tv,
            "spectrum_l1": spectrum,
        }
