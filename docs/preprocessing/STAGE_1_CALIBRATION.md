# Stage 1: calibration assets

This stage prepares the device-specific corrections required before projector
pair construction. Paths below are examples relative to `$CPT_DATA_ROOT`.

## 1. Dark-frame correction

Collect multiple dark DNG frames with the lens fully covered and exposure
settings matching the capture protocol. Fit the mean dark frame:

```bash
python tools/prepare_dark_frame.py \
  --input "$CPT_DATA_ROOT/projector/xiaomi_17promax/calibration/dark_frame_inputs" \
  --output "$CPT_DATA_ROOT/projector/xiaomi_17promax/calibration/dark_frame_fitted" \
  --pattern '*.dng'
```

The fitted output paths must match `raw.dark_frame_raw` and
`raw.dark_frame_rgb` in the selected device configuration.

## 2. Vignetting gain map

For a device whose configuration enables `raw.apply_gain_map`, estimate the
gain map from the calibration captures:

```bash
python tools/fit_vignetting_gainmap.py --help
```

This tool exposes the capture paths, ROI geometry, fit settings, and output
location as command-line arguments. Inspect its help output and use the ROI
parameters recorded in the relevant device YAML. The resulting `.npz` file
must contain the key named by `raw.gain_key`.

The Xiaomi configuration keeps gain-map application disabled. The Huawei
configuration enables it and points to its fitted map.

## 3. Verify configuration resolution

```bash
python - <<'PY'
from color_pass_through.config import load_yaml

cfg = load_yaml("configs/devices/xiaomi_17promax.yaml")
for key in ("dark_frame_raw", "dark_frame_rgb", "gain_map"):
    print(key, cfg["raw"][key])
PY
```

Do not edit the public YAML to insert an absolute machine path. Keep the
`dataset://` URI and set `CPT_DATA_ROOT` instead.

## 4. Camera spectral sensitivity

Camera-null training requires a fitted camera spectral sensitivity array. Use
the device ColorChecker package and the estimator:

```bash
python tools/estimate_camera_sensitivity.py --help
```

Write the fitted result to
`camera_null/fitted/<device>/estimated_css.npy`. The Core Training Data package
already includes this result; rerun the estimator only when reproducing the
calibration or adding a device.

