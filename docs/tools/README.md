# Command-line tools

Run every command from the repository root. Use `--help` for its complete and
current argument list.

## Dataset and preprocessing

| Command | Purpose | Typical package |
|---|---|---|
| `tools/validate_dataset.py` | Validate the complete data bundle, splits, counts, and array headers | Full bundle |
| `tools/prepare_dark_frame.py` | Fit dark-frame correction assets from DNG captures | Preprocessing Sources |
| `tools/fit_vignetting_gainmap.py` | Fit a spatial gain map from calibration captures | Preprocessing Sources |
| `tools/prepare_projector_pairs.py` | Launch device-specific denoising, RAW processing, and RAFT alignment | Preprocessing Sources |
| `tools/estimate_camera_sensitivity.py` | Estimate camera spectral sensitivity from ColorChecker captures | Preprocessing Sources |
| `tools/export_trained_checkpoint.py` | Strip Lightning trainer state from a retrained release model | Best training checkpoint |
| `tools/fit_null_pca.py` | Fit null-space PCA components from hyperspectral data | Core or source HSI |
| `tools/simulate_camera_rgb.py` | Convert HSI cubes to device-simulated camera RGB | Core or source HSI |

The full validator intentionally expects optional raw and calibration data. A
missing optional package is not a core-training failure.

The pair launcher calls `preprocessing/projector_pairs/build_xiaomi.py` or
`build_huawei.py` and writes process images and alignment metrics alongside the
canonical pairs.

## RAW inspection helpers

- `tools/isp_pipeline.py` contains reusable RAW-to-RGB processing helpers.
- `tools/metadata.py` reads capture metadata used by preprocessing.

These modules are primarily imported by preparation scripts rather than used
as end-user CLIs.

## Archived projector conversion

`tools/convert_tcnn_projector.py` converts a historical tiny-cuda-nn projector
checkpoint into the pure-PyTorch projector format used by this repository:

```bash
python tools/convert_tcnn_projector.py \
  --input archived.ckpt \
  --output projector-pytorch.ckpt
```

tiny-cuda-nn is needed only for this optional conversion and numerical
cross-check. It is not imported by normal training or inference.

## Main package CLIs

```text
python -m color_pass_through.cli.train_projector
python -m color_pass_through.cli.train_camera_null
python -m color_pass_through.cli.predict
python -m color_pass_through.cli.calibrate_observer
```

Installed console-script aliases are `cpt-train-projector`,
`cpt-train-camera-null`, `cpt-predict`, and `cpt-calibrate-observer`.
