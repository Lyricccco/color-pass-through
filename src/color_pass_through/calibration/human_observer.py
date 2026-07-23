from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_contact_sheet(
    images: Sequence[np.ndarray],
    phis: Sequence[tuple[float, float, float]],
    output: str | Path,
    columns: int = 5,
    thumbnail_width: int = 320,
) -> Path:
    """Save numbered candidates without embedding observer identity."""
    if len(images) != len(phis) or not images:
        raise ValueError("images and phis must have the same non-zero length")
    first = Image.fromarray(images[0].astype(np.uint8))
    ratio = thumbnail_width / first.width
    thumbnail_height = max(1, round(first.height * ratio))
    label_height = 28
    rows = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (array, phi) in enumerate(zip(images, phis, strict=True)):
        row, column = divmod(index, columns)
        image = Image.fromarray(array.astype(np.uint8)).resize(
            (thumbnail_width, thumbnail_height), Image.Resampling.BILINEAR
        )
        left, top = column * thumbnail_width, row * (thumbnail_height + label_height)
        sheet.paste(image, (left, top))
        draw.text(
            (left + 5, top + thumbnail_height + 5),
            f"{index + 1:03d}  phi=({phi[0]:.4f},{phi[1]:.4f},{phi[2]:.4f})",
            fill="black",
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target
