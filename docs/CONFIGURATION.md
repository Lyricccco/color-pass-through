# Configuration reference

Device configurations live in `configs/devices/`. They are ordinary YAML files
resolved by `color_pass_through.config.load_yaml`.

## Portable paths

- `dataset://path` resolves below `$CPT_DATA_ROOT`.
- `repo://path` resolves below the repository root.
- relative paths resolve below the repository root.

Keep public configuration portable: never commit machine-specific absolute
paths.

The supplied Xiaomi and Huawei configurations already reference the matching
projector and camera-null inference weights, plus the shared RAFT and DualDn
preprocessing weights, in `checkpoints/`. Prediction and pair-preparation CLIs
use these values automatically; explicit checkpoint arguments are only needed
to evaluate retrained or alternative models. See
[`checkpoints/README.md`](../checkpoints/README.md) for provenance and checksums.

## Sections

| Section | Contents |
|---|---|
| `data` | capture roots, prepared pairs, HSI data, simulated RGB, and split files |
| `raw` | demosaicing, white balance, crop, exposure, dark-frame, gain-map, and ROI settings |
| `models.projector` | input mode, Fourier encoding, MLP dimensions, and checkpoint |
| `models.camera_null` | sensitivity/PCA paths, component count, and loss weights |
| `observer` | coefficient search interval and correction sign |
| `preparation` | derived output root, RAFT/DualDn checkpoints, alignment, crop, and blur settings |

## Experiment summaries

`configs/experiments/` records paper-level optimizer and schedule defaults. The
training CLIs currently expose run length, batch size, workers, device,
precision, visualization, and smoke-test controls directly. Model and loss
defaults are read from the device YAML or implemented by the corresponding
training task.

## Adding a device

Copy the closest existing device YAML, give it a new `device_id`, and provide
device-matched prepared pairs, camera sensitivity, PCA components, calibration
assets, frame shape, and RAW geometry. Do not reuse fitted camera assets across
devices. Add tests for any new path or model behavior.
