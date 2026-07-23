from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_calibration(
    output: str | Path,
    observer_id: str,
    device_id: str,
    phi: tuple[float, float, float] | list[float],
    observer_type: str,
    **details: Any,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "observer_id": observer_id,
        "observer_type": observer_type,
        "device_id": device_id,
        "phi": [float(value) for value in phi],
        "details": details,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
