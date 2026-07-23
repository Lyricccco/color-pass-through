#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate camera RGB from 31-band HSI cubes"
    )
    parser.add_argument("--hsi-dir", required=True)
    parser.add_argument("--camera-sensitivity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--normalization",
        choices=["legacy_minmax", "none"],
        default="legacy_minmax",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np

    from color_pass_through.models.camera_null import load_camera_sensitivity

    hsi_dir = Path(args.hsi_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    wavelengths = np.arange(400, 710, 10)
    sensitivity = load_camera_sensitivity(args.camera_sensitivity, wavelengths)
    records: list[dict[str, object]] = []
    for source in sorted(hsi_dir.glob("*.npy")):
        target = output / f"{source.stem}_1.npy"
        if target.exists() and not args.overwrite:
            records.append({"source": str(source), "status": "exists"})
            continue
        spectrum = np.load(source, mmap_mode="r")
        if spectrum.ndim != 3 or spectrum.shape[-1] != 31:
            raise ValueError(f"Expected H×W×31 in {source}, got {spectrum.shape}")
        camera = np.tensordot(spectrum, sensitivity, axes=([-1], [0])).astype(
            np.float32
        )
        if args.normalization == "legacy_minmax":
            low, high = float(camera.min()), float(camera.max())
            camera = (camera - low) / max(high - low, 1e-12)
        np.save(target, camera)
        records.append(
            {"source": str(source), "target": str(target), "status": "written"}
        )
    manifest = {
        "normalization": args.normalization,
        "camera_sensitivity": str(Path(args.camera_sensitivity).resolve()),
        "count": len(records),
        "records": records,
    }
    (output / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"processed {len(records)} cubes -> {output}")


if __name__ == "__main__":
    main()
