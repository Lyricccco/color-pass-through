from __future__ import annotations

from decimal import Decimal
from itertools import product


def _axis(low: float, high: float, step: float) -> list[float]:
    lo, hi, delta = Decimal(str(low)), Decimal(str(high)), Decimal(str(step))
    if delta <= 0 or hi < lo:
        raise ValueError("Require step > 0 and high >= low")
    count = int((hi - lo) / delta)
    values = [float(lo + index * delta) for index in range(count + 1)]
    if Decimal(str(values[-1])) != hi:
        raise ValueError("(high - low) must be divisible by step")
    return values


def generate_phi_grid(
    low: float = 0.025, high: float = 0.075, step: float = 0.0125
) -> list[tuple[float, float, float]]:
    """Return the paper's 5×5×5 observer calibration grid."""
    values = _axis(low, high, step)
    return list(product(values, repeat=3))
