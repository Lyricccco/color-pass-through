from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.camera_null import CameraNullModel
from .models.projector import CameraDisplayProjector


def apply_observer_correction(
    camera_rgb: torch.Tensor,
    null_coefficient: torch.Tensor,
    phi: torch.Tensor | tuple[float, float, float] | list[float],
    correction_sign: int = -1,
) -> torch.Tensor:
    """Apply the observer correction in camera RGB and clamp to display range."""
    if correction_sign not in (-1, 1):
        raise ValueError("correction_sign must be -1 or +1")
    if null_coefficient.ndim != 4 or null_coefficient.shape[1] != 1:
        raise ValueError(
            "The main paper correction requires one coefficient with shape (N,1,H,W)"
        )
    phi_tensor = torch.as_tensor(
        phi, dtype=camera_rgb.dtype, device=camera_rgb.device
    ).view(1, 3, 1, 1)
    return (camera_rgb + correction_sign * phi_tensor * null_coefficient).clamp(
        0.0, 1.0
    )


@dataclass
class PipelineResult:
    """Intermediate and final tensors returned by one inference pass."""
    camera_rgb: torch.Tensor
    null_coefficient: torch.Tensor
    corrected_camera_rgb: torch.Tensor
    display_rgb: torch.Tensor


class ColorPassThroughPipeline:
    """Compose camera-null prediction, observer correction, and projection."""
    def __init__(
        self,
        camera_null: CameraNullModel,
        projector: CameraDisplayProjector,
        correction_sign: int = -1,
    ) -> None:
        self.camera_null = camera_null
        self.projector = projector
        self.correction_sign = correction_sign

    @torch.inference_mode()
    def predict(
        self,
        camera_rgb: torch.Tensor,
        phi: tuple[float, float, float] | list[float] | torch.Tensor,
    ) -> PipelineResult:
        """Run the full color pass-through pipeline for a batch of camera RGB."""
        coefficient = self.camera_null.predict_coefficient(camera_rgb)
        corrected = apply_observer_correction(
            camera_rgb, coefficient, phi, self.correction_sign
        )
        display = self.projector(corrected).clamp(0.0, 1.0)
        return PipelineResult(camera_rgb, coefficient, corrected, display)
