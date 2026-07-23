#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate all inputs for one device")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Optional JSON report")
    parser.add_argument("--deep", action="store_true", help="Load array headers")
    return parser


def _ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np

    from color_pass_through.config import load_yaml

    config = load_yaml(args.config)
    report: dict[str, object] = {
        "device_id": config["device_id"],
        "checks": {},
        "errors": [],
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    def require_file(path: Path) -> bool:
        if not path.is_file():
            errors.append(f"missing: {path}")
            return False
        return True

    def require_dir(path: Path) -> bool:
        if not path.is_dir():
            errors.append(f"missing directory: {path}")
            return False
        return True

    capture_root = Path(config["data"]["capture_root"])
    checks["capture_root"] = str(capture_root)
    capture_counts: dict[str, int] = {}
    for name, pattern in (
        ("Camera_Raw", "*.dng"),
        ("Camera_ISP", "*.jpg"),
        ("Display_sRGB", "*.png"),
    ):
        directory = capture_root / name
        capture_counts[name] = len(list(directory.glob(pattern))) if require_dir(directory) else 0
    checks["capture_counts"] = capture_counts
    if any(value != 901 for value in capture_counts.values()):
        errors.append(f"expected 901 raw capture triplets, got {capture_counts}")

    raw_config = config["raw"]
    for key in ("gain_map", "dark_frame_raw", "dark_frame_rgb"):
        require_file(Path(raw_config[key]))
    preparation = config["preparation"]
    for key in ("flow_checkpoint", "denoise_checkpoint"):
        require_file(Path(preparation[key]))

    colorchecker_root = Path(config["data"]["colorchecker"])
    colorchecker_files = [
        path
        for path in colorchecker_root.glob("Colorchecker*")
        if path.is_file()
    ] if require_dir(colorchecker_root) else []
    checks["colorchecker_files"] = len(colorchecker_files)
    if not colorchecker_files:
        errors.append(f"no Colorchecker captures: {colorchecker_root}")

    projector = config["data"]["projector"]
    pair_root = Path(projector["pairs"])
    train_split = Path(projector["train_split"])
    test_split = Path(projector["test_split"])
    hald_split = Path(projector["hald_split"])
    train_ids = _ids(train_split) if require_file(train_split) else []
    test_ids = _ids(test_split) if require_file(test_split) else []
    hald_ids = _ids(hald_split) if require_file(hald_split) else []
    checks["projector_train_ids"] = len(train_ids)
    checks["projector_test_ids"] = len(test_ids)
    checks["projector_hald_ids"] = len(hald_ids)
    if (len(train_ids), len(test_ids), len(hald_ids)) != (800, 100, 1):
        errors.append(
            "expected projector splits 800/100/1, got "
            f"{len(train_ids)}/{len(test_ids)}/{len(hald_ids)}"
        )
    checks["projector_pair_root"] = str(pair_root)
    for sample_id in train_ids + test_ids + hald_ids:
        stem = f"{int(sample_id):04d}" if sample_id.isdigit() else sample_id
        for suffix in ("CameraRaw.npy", "DisplaysRGB.png"):
            path = pair_root / f"{stem}_{suffix}"
            if not path.is_file():
                errors.append(f"missing: {path}")

    null_data = config["data"]["camera_null"]
    spectrum_root = Path(null_data["spectrum"])
    camera_root = Path(null_data["camera_rgb"])
    validation_split = Path(null_data["validation_split"])
    validation_ids = set(_ids(validation_split)) if require_file(validation_split) else set()
    require_dir(spectrum_root)
    require_dir(camera_root)
    spectrum_files = sorted(spectrum_root.glob("*.npy"))
    checks["hsi_total"] = len(spectrum_files)
    checks["hsi_validation_ids"] = len(validation_ids)
    if len(spectrum_files) != 1182:
        errors.append(f"expected 1182 HSI cubes, got {len(spectrum_files)}")
    if len(validation_ids) != 182:
        errors.append(f"expected 182 HSI validation IDs, got {len(validation_ids)}")
    for spectrum in spectrum_files:
        camera = camera_root / f"{spectrum.stem}_1.npy"
        if not camera.is_file():
            errors.append(f"missing: {camera}")

    model = config["models"]["camera_null"]
    css_path = Path(model["camera_sensitivity"])
    pca_path = Path(model["pca_components"])
    for path in (css_path, pca_path):
        require_file(path)
    if args.deep and css_path.is_file() and pca_path.is_file():
        checks["css_shape"] = list(np.load(css_path, mmap_mode="r").shape)
        checks["pca_shape"] = list(np.load(pca_path, mmap_mode="r").shape)
        if checks["css_shape"] != [33, 4]:
            errors.append(f"expected CSS shape [33, 4], got {checks['css_shape']}")
        if checks["pca_shape"] != [3, 31]:
            errors.append(f"expected PCA shape [3, 31], got {checks['pca_shape']}")
        if spectrum_files:
            checks["hsi_example_shape"] = list(
                np.load(spectrum_files[0], mmap_mode="r").shape
            )

    overlap = (set(train_ids) & set(test_ids)) | (set(train_ids) & set(hald_ids)) | (
        set(test_ids) & set(hald_ids)
    )
    checks["overlapping_projector_ids"] = sorted(overlap)
    if overlap:
        errors.append(f"overlapping projector IDs: {sorted(overlap)}")
    checks["status"] = "passed" if not errors else "failed"
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
