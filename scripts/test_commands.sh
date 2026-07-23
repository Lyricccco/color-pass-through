#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-static}"
DEVICE="${2:-xiaomi}"

if [[ "$MODE" == "data" || "$MODE" == "train-smoke" ]]; then
  if [[ -z "${CPT_DATA_ROOT:-}" ]]; then
    echo "CPT_DATA_ROOT must point to the downloaded color-pass-through dataset" >&2
    exit 2
  fi
fi

if [[ "$DEVICE" == "xiaomi" ]]; then
  CONFIG="configs/devices/xiaomi_17promax.yaml"
elif [[ "$DEVICE" == "huawei" ]]; then
  CONFIG="configs/devices/huawei_pura70pro.yaml"
else
  echo "device must be xiaomi or huawei" >&2
  exit 2
fi

case "$MODE" in
  static)
    python -m compileall -q src tools tests preprocessing/projector_pairs
    python -m color_pass_through.cli.train_projector --help >/dev/null
    python -m color_pass_through.cli.train_camera_null --help >/dev/null
    python -m color_pass_through.cli.predict --help >/dev/null
    python -m color_pass_through.cli.calibrate_observer --help >/dev/null
    python tools/estimate_camera_sensitivity.py --help >/dev/null
    python tools/fit_null_pca.py --help >/dev/null
    python tools/fit_vignetting_gainmap.py --help >/dev/null
    python tools/simulate_camera_rgb.py --help >/dev/null
    python tools/prepare_projector_pairs.py --help >/dev/null
    python tools/export_trained_checkpoint.py --help >/dev/null
    python preprocessing/projector_pairs/build_xiaomi.py --help >/dev/null
    python preprocessing/projector_pairs/build_huawei.py --help >/dev/null
    python tools/convert_tcnn_projector.py --help >/dev/null
    if python -c 'import pytest' 2>/dev/null; then
      python -m pytest -q
    else
      python - <<'PY'
import runpy
from pathlib import Path
for file in sorted(Path("tests").glob("test_*.py")):
    scope = runpy.run_path(str(file))
    for name, value in sorted(scope.items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASS {file}::{name}")
PY
    fi
    ;;
  data)
    python tools/validate_dataset.py \
      --config configs/devices/xiaomi_17promax.yaml --deep
    python tools/validate_dataset.py \
      --config configs/devices/huawei_pura70pro.yaml --deep
    ;;
  model)
    python -m pytest -q tests/test_models.py tests/test_pipeline.py
    ;;
  train-smoke)
    python -m color_pass_through.cli.train_projector \
      --config "$CONFIG" \
      --output "runs/smoke/${DEVICE}_projector" \
      --epochs 1 --batch-size 1 --workers 0 --limit-train-batches 1
    python -m color_pass_through.cli.train_camera_null \
      --config "$CONFIG" \
      --output "runs/smoke/${DEVICE}_camera_null" \
      --epochs 1 --batch-size 1 --workers 0 --crop-size 64 \
      --limit-train-batches 1
    ;;
  *)
    echo "usage: $0 {static|data|model|train-smoke} [xiaomi|huawei]" >&2
    exit 2
    ;;
esac
