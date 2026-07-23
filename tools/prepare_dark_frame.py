#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average dark/flat-frame captures and write a manifest"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*.dng")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np
    import rawpy
    from PIL import Image

    files = sorted(Path(args.input).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.input}")
    total = None
    shape = None
    for path in files:
        with rawpy.imread(str(path)) as raw:
            image = raw.raw_image_visible.astype(np.float64)
        if shape is None:
            shape = image.shape
            total = np.zeros(shape, dtype=np.float64)
        if image.shape != shape:
            raise ValueError(f"Shape mismatch: {path} is {image.shape}, expected {shape}")
        total += image
    assert total is not None
    average = (total / len(files)).astype(np.float32)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "Fit_flat-frame.npy", average)
    preview = (255 * (average - average.min()) / max(np.ptp(average), 1e-12)).astype(
        np.uint8
    )
    Image.fromarray(preview).save(output / "Fit_flat-frame.png")
    manifest = {
        "semantic_note": (
            "Archived code subtracts this average. Confirm whether captures are "
            "dark frames before public release."
        ),
        "pattern": args.pattern,
        "count": len(files),
        "shape": list(average.shape),
        "files": [str(path.resolve()) for path in files],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
