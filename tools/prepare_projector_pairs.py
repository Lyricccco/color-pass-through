#!/usr/bin/env python3
"""Launch the paper's device-specific camera-display pair construction.

The maintained entrypoints live under ``preprocessing/projector_pairs``. This
wrapper resolves portable dataset paths, forwards device settings, validates
inputs, and records the exact command used for a derived dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare aligned projector pairs")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-name",
        default="rebuilt-v1",
        help="Directory name created below preparation.pair_output_root",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from color_pass_through.config import REPOSITORY_ROOT, load_yaml

    config = load_yaml(args.config)
    device = config["device_id"]
    target = Path(config["preparation"]["pair_output_root"]) / args.output_name
    canonical = Path(config["data"]["projector"]["pairs"]).parent
    if target.resolve() == canonical.resolve():
        raise ValueError(
            f"Refusing to overwrite canonical packaged pairs at {canonical}; "
            "choose an output name other than 'v1'."
        )
    if device.startswith("huawei"):
        script = "build_huawei.py"
        gain_args = [
            "--gainmap",
            config["raw"]["gain_map"],
            "--gain_type",
            config["raw"].get("gain_key", "gain_center1"),
        ]
    elif device.startswith("xiaomi"):
        script = "build_xiaomi.py"
        gain_args = []
    else:
        raise ValueError(f"No archived pair-preparation script for {device}")
    preprocessing_root = REPOSITORY_ROOT / "preprocessing" / "projector_pairs"
    raw_roi = config["raw"]["pair_roi"]["raw"]
    isp_roi = config["raw"]["pair_roi"]["isp"]
    preparation = config["preparation"]
    command = [
        sys.executable,
        str(preprocessing_root / script),
        "--dataset",
        config["data"]["capture_root"],
        "--output-root",
        str(Path(config["preparation"]["pair_output_root"]) / args.output_name),
        "--base_exposure",
        str(config["raw"]["base_exposure"]),
        "--flow-checkpoint",
        config["preparation"]["flow_checkpoint"],
        "--denoise-checkpoint",
        config["preparation"]["denoise_checkpoint"],
        "--flat-raw",
        config["raw"]["dark_frame_raw"],
        "--flat-rgb",
        config["raw"]["dark_frame_rgb"],
        "--raw-roi",
        str(raw_roi["center_x"]),
        str(raw_roi["center_y"]),
        str(raw_roi["height"]),
        "--isp-roi",
        str(isp_roi["center_x"]),
        str(isp_roi["center_y"]),
        str(isp_roi["height"]),
        "--resize-scale",
        str(preparation["resize_scale"]),
        "--border-crop",
        str(preparation["border_crop"]),
        "--flow-iterations",
        str(preparation["flow_iterations"]),
        "--blur-display-kernel",
        str(preparation["blur_display_kernel"]),
        *gain_args,
    ]
    print(" ".join(command))
    manifest = {
        "device_id": device,
        "command": command,
        "flow_checkpoint": config["preparation"]["flow_checkpoint"],
        "denoise_checkpoint": config["preparation"]["denoise_checkpoint"],
        "output_root": str(target),
        "note": "Camera-display pair construction with centralized device settings.",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    required = [
        Path(config["data"]["capture_root"]) / "Camera_Raw",
        Path(config["data"]["capture_root"]) / "Camera_ISP",
        Path(config["data"]["capture_root"]) / "Display_sRGB",
        Path(config["preparation"]["flow_checkpoint"]),
        Path(config["preparation"]["denoise_checkpoint"]),
        Path(config["raw"]["dark_frame_raw"]),
        Path(config["raw"]["dark_frame_rgb"]),
    ]
    if device.startswith("huawei"):
        required.append(Path(config["raw"]["gain_map"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pair-preparation inputs:\n" + "\n".join(missing))
    if shutil.which("exiftool") is None:
        raise RuntimeError(
            "ExifTool is required for RAW exposure and NoiseProfile metadata. "
            "Install it and confirm `exiftool -ver` before rebuilding pairs."
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{preprocessing_root}:{environment.get('PYTHONPATH', '')}".rstrip(":")
    )
    subprocess.run(command, check=True, cwd=preprocessing_root, env=environment)
    target.mkdir(parents=True, exist_ok=True)
    (target / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
