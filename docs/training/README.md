# Training

Training uses canonical prepared data selected through a device YAML. No raw
capture preprocessing runs inside either trainer.

## Common setup

```bash
export CPT_DATA_ROOT=/path/to/color-pass-through
export CPT_CONFIG=configs/devices/xiaomi_17promax.yaml
mkdir -p runs
```

Use `configs/devices/huawei_pura70pro.yaml` for Huawei data.

## Retrain all four release models

The following commands train directly from the canonical prepared projector
pairs and camera-null arrays selected by the two device configurations. They do
not rebuild pairs or read original RAW captures.

```bash
export RUN_ROOT="${RUN_ROOT:-runs/retrain-release}"
mkdir -p "$RUN_ROOT/08_training"

python -m color_pass_through.cli.train_projector \
  --config configs/devices/xiaomi_17promax.yaml \
  --output "$RUN_ROOT/08_training/xiaomi-projector-canonical" \
  --epochs 50 --batch-size 8 --workers 8 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python -m color_pass_through.cli.train_projector \
  --config configs/devices/huawei_pura70pro.yaml \
  --output "$RUN_ROOT/08_training/huawei-projector-canonical" \
  --epochs 50 --batch-size 8 --workers 8 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python -m color_pass_through.cli.train_camera_null \
  --config configs/devices/xiaomi_17promax.yaml \
  --output "$RUN_ROOT/08_training/xiaomi-camera-null" \
  --epochs 50 --batch-size 4 --workers 4 --crop-size 256 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42

python -m color_pass_through.cli.train_camera_null \
  --config configs/devices/huawei_pura70pro.yaml \
  --output "$RUN_ROOT/08_training/huawei-camera-null" \
  --epochs 50 --batch-size 4 --workers 4 --crop-size 256 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42
```

Run the commands sequentially on one GPU, or assign separate GPUs externally
when launching parallel jobs. Projector visualizations, test predictions, and
LUT exports are enabled by default. Camera-null validation visualization and
CSV export are also enabled by default.

### Export the new release checkpoints

Each fresh output contains `last.ckpt` plus one best-scoring epoch checkpoint.
Select the epoch checkpoint (not `last.ckpt`) and strip Lightning trainer and
optimizer state before publishing it:

```bash
XIAOMI_PROJECTOR_BEST=("$RUN_ROOT"/08_training/xiaomi-projector-canonical/checkpoints/projector-*.ckpt)
HUAWEI_PROJECTOR_BEST=("$RUN_ROOT"/08_training/huawei-projector-canonical/checkpoints/projector-*.ckpt)
XIAOMI_NULL_BEST=("$RUN_ROOT"/08_training/xiaomi-camera-null/checkpoints/camera-null-*.ckpt)
HUAWEI_NULL_BEST=("$RUN_ROOT"/08_training/huawei-camera-null/checkpoints/camera-null-*.ckpt)

python tools/export_trained_checkpoint.py \
  --config configs/devices/xiaomi_17promax.yaml --model projector \
  --input "${XIAOMI_PROJECTOR_BEST[0]}" \
  --output checkpoints/xiaomi_projector.ckpt --force

python tools/export_trained_checkpoint.py \
  --config configs/devices/huawei_pura70pro.yaml --model projector \
  --input "${HUAWEI_PROJECTOR_BEST[0]}" \
  --output checkpoints/huawei_projector.ckpt --force

python tools/export_trained_checkpoint.py \
  --config configs/devices/xiaomi_17promax.yaml --model camera-null \
  --input "${XIAOMI_NULL_BEST[0]}" \
  --output checkpoints/xiaomi_camera_null.ckpt --force

python tools/export_trained_checkpoint.py \
  --config configs/devices/huawei_pura70pro.yaml --model camera-null \
  --input "${HUAWEI_NULL_BEST[0]}" \
  --output checkpoints/huawei_camera_null.ckpt --force
```

Use fresh output directories so each glob resolves to exactly one best epoch
checkpoint. After exporting, run `sha256sum checkpoints/*.ckpt` and update the
epochs, validation metrics, and hashes in `checkpoints/MANIFEST.json` before
publishing.

## Camera-display projector

```bash
python -m color_pass_through.cli.train_projector \
  --config "$CPT_CONFIG" \
  --output runs/projector \
  --epochs 50 --batch-size 8 --workers 8 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42
```

The trainer reads `data.projector.pairs`, `train_split`, and `test_split` from
the resolved device configuration. It writes checkpoints under
`runs/projector/checkpoints/` and reloads the checkpoint with the best
validation PSNR for its final test pass.

The following diagnostics are enabled by default:

- first 10 test-split samples visualized every 5 epochs;
- test predictions saved after loading the best checkpoint;
- a size-33 `.cube` LUT and RGB quiver plot exported every 10 epochs.

Useful controls:

```text
--visualize-samples N
--visualize-every-n-epochs N
--[no-]save-test-predictions
--[no-]lut-visualization
--lut-size N
--lut-every-n-epochs N
```

Use `--visualize-samples 0 --no-save-test-predictions
--no-lut-visualization` for a run without visual exports.

## Camera-null model

```bash
python -m color_pass_through.cli.train_camera_null \
  --config "$CPT_CONFIG" \
  --output runs/camera-null \
  --epochs 50 --batch-size 4 --workers 4 --crop-size 256 \
  --accelerator gpu --devices 1 --precision 16-mixed --seed 42
```

The training set excludes identifiers in `hsi_val_182.txt`; validation uses
exactly those identifiers. The default loss is

```text
coefficient_weight = 1.0
tv_weight          = 0.01
spectrum_weight    = 1.0
```

The values are explicit in each device configuration. Validation visualization
is enabled by default for up to 182 samples every 10 epochs, including a
per-sample CSV. Controls are:

```text
--[no-]validation-visualization
--visualize-samples N
--visualize-every-n-epochs N
--[no-]save-validation-csv
```

## Small smoke runs

Before a full run, verify data loading and one optimizer step:

```bash
python -m color_pass_through.cli.train_projector \
  --config "$CPT_CONFIG" --output runs/smoke/projector \
  --epochs 1 --batch-size 1 --workers 0 --limit-train-batches 1 \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --visualize-samples 0 --no-save-test-predictions --no-lut-visualization

python -m color_pass_through.cli.train_camera_null \
  --config "$CPT_CONFIG" --output runs/smoke/camera-null \
  --epochs 1 --batch-size 1 --workers 0 --crop-size 64 \
  --limit-train-batches 1 --accelerator gpu --devices 1 \
  --precision 16-mixed --no-validation-visualization
```

`--workers 0` is useful when diagnosing data-loader errors because exceptions
are then reported in the main process.

## Output ownership

Training outputs are local experiment artifacts and are ignored by Git. Keep
the device configuration, random seed, exact checkpoint, and inference `phi`
values with any reported result.
