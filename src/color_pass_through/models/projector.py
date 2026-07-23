from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .encodings import FourierEncoding, IdentityEncoding


@dataclass(frozen=True)
class ProjectorSettings:
    input_mode: Literal["raw_green_context", "rgb", "hdr_assisted"] = (
        "raw_green_context"
    )
    encoding: Literal["fourier", "identity"] = "fourier"
    num_frequencies: int = 4
    hidden_dim: int = 128
    hidden_layers: int = 2
    output_activation: Literal["none", "relu", "sigmoid"] = "none"


def _input_dimensions(mode: str) -> int:
    # Main paper: RGB + blur(G). Supplement: RAW, blur(RAW), camera ISP.
    dimensions = {"raw_green_context": 4, "rgb": 3, "hdr_assisted": 9}
    if mode not in dimensions:
        raise ValueError(f"Unknown projector input mode: {mode}")
    return dimensions[mode]


class CameraDisplayProjector(nn.Module):
    """Per-pixel camera-to-display mapping used in Color Pass-Through."""

    def __init__(self, settings: ProjectorSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or ProjectorSettings()
        input_dim = _input_dimensions(self.settings.input_mode)
        if self.settings.encoding == "fourier":
            self.encoder = FourierEncoding(
                input_dim=input_dim,
                num_frequencies=self.settings.num_frequencies,
            )
        elif self.settings.encoding == "identity":
            self.encoder = IdentityEncoding(input_dim)
        else:
            raise ValueError(f"Unknown encoding: {self.settings.encoding}")

        layers: list[nn.Module] = []
        previous = self.encoder.output_dim
        for _ in range(self.settings.hidden_layers):
            layers.extend(
                [nn.Linear(previous, self.settings.hidden_dim), nn.ReLU(inplace=True)]
            )
            previous = self.settings.hidden_dim
        layers.append(nn.Linear(previous, 3))
        if self.settings.output_activation == "relu":
            layers.append(nn.ReLU())
        elif self.settings.output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.settings.output_activation != "none":
            raise ValueError(
                f"Unknown output activation: {self.settings.output_activation}"
            )
        self.mlp = nn.Sequential(*layers)

    def _features(
        self, camera_rgb: torch.Tensor, camera_isp: torch.Tensor | None
    ) -> torch.Tensor:
        if camera_rgb.ndim != 4 or camera_rgb.shape[1] != 3:
            raise ValueError(
                f"camera_rgb must have shape (N,3,H,W), got {tuple(camera_rgb.shape)}"
            )
        if self.settings.input_mode == "rgb":
            return camera_rgb
        if self.settings.input_mode == "raw_green_context":
            green = camera_rgb[:, 1:2]
            green_blur = F.avg_pool2d(green, kernel_size=3, stride=1, padding=1)
            return torch.cat([camera_rgb, green_blur], dim=1)
        if camera_isp is None:
            raise ValueError("hdr_assisted mode requires camera_isp")
        if camera_isp.shape != camera_rgb.shape:
            raise ValueError(
                f"camera_isp shape {tuple(camera_isp.shape)} != "
                f"camera_rgb shape {tuple(camera_rgb.shape)}"
            )
        raw_blur = F.avg_pool2d(camera_rgb, kernel_size=3, stride=1, padding=1)
        return torch.cat([camera_rgb, raw_blur, camera_isp], dim=1)

    def forward(
        self, camera_rgb: torch.Tensor, camera_isp: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self._features(camera_rgb, camera_isp)
        batch, _, height, width = features.shape
        pixels = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        output = self.mlp(self.encoder(pixels))
        return output.view(batch, height, width, 3).permute(0, 3, 1, 2).contiguous()

    def map_rgb_samples(self, rgb: torch.Tensor) -> torch.Tensor:
        """Map an ``(N,3)`` RGB grid for legacy-compatible 3D LUT export."""
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape (N,3), got {tuple(rgb.shape)}")
        if self.settings.input_mode == "rgb":
            features = rgb
        elif self.settings.input_mode == "raw_green_context":
            features = rgb[:, [0, 1, 2, 1]]
        else:
            raise ValueError(
                "A 3D RGB LUT is undefined for hdr_assisted projector input"
            )
        return self.mlp(self.encoder(features))
