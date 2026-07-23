from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def patch_mse(
    candidate: np.ndarray,
    reference: np.ndarray,
    rois: Sequence[tuple[int, int, int, int]],
) -> float:
    errors: list[float] = []
    for left, top, right, bottom in rois:
        a = candidate[top:bottom, left:right].astype(np.float64)
        b = reference[top:bottom, left:right].astype(np.float64)
        if a.shape != b.shape or a.size == 0:
            raise ValueError(f"Invalid ROI {(left, top, right, bottom)}")
        errors.append(float(np.mean((a - b) ** 2)))
    return float(np.mean(errors))


def select_digital_observer(
    candidates: Sequence[np.ndarray],
    phis: Sequence[tuple[float, float, float]],
    reference: np.ndarray,
    rois: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[float, float, float], list[float]]:
    if len(candidates) != len(phis) or not candidates:
        raise ValueError("candidates and phis must have the same non-zero length")
    scores = [patch_mse(image, reference, rois) for image in candidates]
    best = int(np.argmin(scores))
    return phis[best], scores
