from __future__ import annotations

from collections.abc import Sequence

import torch

from color_pass_through.models.projector import CameraDisplayProjector


@torch.inference_mode()
def render_phi_candidates(
    camera_rgb: torch.Tensor,
    null_coefficient: torch.Tensor,
    projector: CameraDisplayProjector,
    phis: Sequence[tuple[float, float, float]],
    correction_sign: int = -1,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Render a candidate stack with shape ``(T,3,H,W)`` for one scene."""
    if camera_rgb.shape[0] != 1 or null_coefficient.shape[0] != 1:
        raise ValueError("Observer calibration currently renders one scene at a time")
    outputs: list[torch.Tensor] = []
    for start in range(0, len(phis), chunk_size):
        values = phis[start : start + chunk_size]
        count = len(values)
        camera = camera_rgb.expand(count, -1, -1, -1)
        coefficient = null_coefficient.expand(count, -1, -1, -1)
        phi = torch.as_tensor(
            values, dtype=camera.dtype, device=camera.device
        ).view(count, 3, 1, 1)
        corrected = (
            camera + correction_sign * phi * coefficient
        ).clamp(0.0, 1.0)
        outputs.append(projector(corrected).clamp(0.0, 1.0).cpu())
    return torch.cat(outputs, dim=0)
