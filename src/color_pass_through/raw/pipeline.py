from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .isp_pipeline import run_pipeline
from .metadata import rawpy_metadata, read_exif, relative_exposure


def _orient_chw(image: torch.Tensor, orientation: int) -> torch.Tensor:
    if orientation == 1:
        return image
    if orientation == 2:
        return image.flip(-2)
    if orientation == 3:
        return torch.rot90(image, 2, (-2, -1))
    if orientation == 4:
        return image.flip(-1)
    if orientation == 5:
        return torch.rot90(image.flip(-2), 1, (-2, -1))
    if orientation == 6:
        return torch.rot90(image, 3, (-2, -1))
    if orientation == 7:
        return torch.rot90(image.flip(-2), 3, (-2, -1))
    if orientation == 8:
        return torch.rot90(image, 1, (-2, -1))
    raise ValueError(f"Unknown RAW orientation: {orientation}")


def _center_crop_multiple(image: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    target_h, target_w = (height // multiple) * multiple, (width // multiple) * multiple
    if target_h == 0 or target_w == 0:
        raise ValueError(f"Image {height}x{width} is smaller than {multiple}")
    top, left = (height - target_h) // 2, (width - target_w) // 2
    return image[..., top : top + target_h, left : left + target_w]


class RawProcessor:
    """DNG to linear camera RGB, matching the archived inference stages."""

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def process(self, path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            import rawpy
        except ImportError as error:
            raise RuntimeError("RAW input requires the 'rawpy' package") from error

        source = Path(path)
        exif = read_exif(source, self.config.get("exiftool", "exiftool"))
        with rawpy.imread(str(source)) as raw:
            orientation = int(raw.sizes.flip)
            metadata = rawpy_metadata(raw, self.device)
            metadata["demosaic_type"] = self.config.get("demosaic", "AHD")
            white_balance_mode = self.config.get("white_balance_mode", "camera")
            if white_balance_mode == "unity":
                metadata["wb_matrix"] = torch.ones(
                    (1, 1, 1, 4), dtype=torch.float32, device=self.device
                )
            elif white_balance_mode == "analog_balance_to_unity":
                value = exif.get("AnalogBalance")
                if not value:
                    raise KeyError("AnalogBalance is required by white_balance_mode")
                if isinstance(value, str):
                    values = [
                        float(item)
                        for item in value.strip().replace(",", " ").split()
                    ]
                else:
                    values = [float(item) for item in value]
                if len(values) < 3:
                    raise ValueError(f"Invalid AnalogBalance: {value}")
                analog = torch.tensor(
                    [values[0], values[1], values[2], values[1]],
                    dtype=torch.float32,
                    device=self.device,
                )
                metadata["wb_matrix"] = (
                    1.0 / analog.clamp_min(1e-6)
                ).view(1, 1, 1, 4)
            elif white_balance_mode != "camera":
                raise ValueError(f"Unknown white_balance_mode: {white_balance_mode}")
            mosaic = torch.from_numpy(raw.raw_image_visible.astype(np.float32))
            mosaic = mosaic.view(1, 1, *mosaic.shape).to(self.device)
            linear = run_pipeline(mosaic, metadata, "raw", "demosaic").squeeze(0)

        linear = _orient_chw(linear, orientation)
        factor = relative_exposure(exif, float(self.config["base_exposure"]))
        linear = linear * factor

        gain_path = self.config.get("gain_map")
        if gain_path and self.config.get("apply_gain_map", False):
            key = self.config.get("gain_key", "gain_center1")
            gain_array = np.load(gain_path, allow_pickle=False)[key]
            gain = torch.from_numpy(gain_array.astype(np.float32)).permute(2, 0, 1)
            gain = _orient_chw(gain, orientation).to(self.device)
            if gain.shape[-2:] != linear.shape[-2:]:
                gain = F.interpolate(
                    gain.unsqueeze(0),
                    size=linear.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            linear = linear * gain

        crop = int(self.config.get("crop_top_bottom", 0))
        if crop:
            if 2 * crop >= linear.shape[-2]:
                raise ValueError("crop_top_bottom removes the entire image")
            linear = linear[:, crop:-crop]
        scale = float(self.config.get("resize_scale", 1.0))
        if scale != 1.0:
            linear = F.interpolate(
                linear.unsqueeze(0),
                scale_factor=scale,
                mode="area",
            ).squeeze(0)
        linear = _center_crop_multiple(
            linear, int(self.config.get("crop_multiple", 8))
        )
        return linear.clamp(0.0, 1.0), {
            "source": str(source),
            "orientation": orientation,
            "relative_exposure": factor,
            "white_balance_mode": self.config.get("white_balance_mode", "camera"),
            "apply_gain_map": bool(self.config.get("apply_gain_map", False)),
            "shape": list(linear.shape),
        }
