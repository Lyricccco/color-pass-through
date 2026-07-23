from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rawpy_metadata(raw: Any, device: torch.device) -> dict[str, Any]:
    rgb_xyz = np.asarray(raw.rgb_xyz_matrix[:3, :3], dtype=np.float32)
    if np.all(rgb_xyz):
        rgb_xyz = rgb_xyz / np.sum(rgb_xyz, axis=1, keepdims=True)
        rgb_xyz = np.linalg.inv(rgb_xyz)
    else:
        color = np.asarray(raw.color_matrix[:3, :3], dtype=np.float32)
        xyz_to_srgb = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=np.float32,
        )
        xyz_to_srgb /= xyz_to_srgb.sum(axis=-1, keepdims=True)
        rgb_xyz = np.linalg.inv(xyz_to_srgb) @ color
        rgb_xyz /= rgb_xyz.sum(axis=1, keepdims=True)
    white_balance = np.asarray(raw.camera_whitebalance, dtype=np.float32)
    if white_balance[0] == 0:
        white_balance = np.asarray(raw.daylight_whitebalance, dtype=np.float32)
    if white_balance[3] == 0:
        white_balance[3] = white_balance[1]
    return {
        "black_level": list(raw.black_level_per_channel),
        "white_level": float(raw.white_level),
        "wb_matrix": torch.from_numpy(white_balance)
        .view(1, 1, 1, 4)
        .to(device),
        "color_desc": "RGBG",
        "color_mask": torch.from_numpy(raw.raw_colors_visible.astype(np.int64))
        .view(1, 1, *raw.raw_colors_visible.shape)
        .to(device),
        "rgb_xyz_matrix": torch.from_numpy(rgb_xyz).unsqueeze(0).to(device),
        "demosaic_type": "AHD",
        "ref": "D65",
        "gamma_type": "Rec709",
        "alpha": 0,
    }


def read_exif(path: str | Path, executable: str = "exiftool") -> dict[str, Any]:
    process = subprocess.run(
        [executable, "-j", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"exiftool failed for {path}: {process.stderr.strip()}")
    try:
        rows = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid exiftool JSON for {path}") from error
    if len(rows) != 1:
        raise RuntimeError(f"Expected one exiftool record for {path}")
    return rows[0]


def _number(value: Any, name: str) -> float:
    if value is None:
        raise KeyError(f"Missing EXIF field: {name}")
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid EXIF {name}: {value}") from error


def relative_exposure(exif: dict[str, Any], base_exposure: float) -> float:
    exposure_time = _number(exif.get("ExposureTime"), "ExposureTime")
    f_number = _number(exif.get("FNumber"), "FNumber")
    iso = _number(exif.get("ISO"), "ISO")
    baseline = _number(exif.get("BaselineExposure", 0), "BaselineExposure")
    current = exposure_time * iso * (2.0**baseline) / (f_number**2)
    if current <= 0:
        raise ValueError(f"Non-positive exposure value: {current}")
    return float(base_exposure) / current
