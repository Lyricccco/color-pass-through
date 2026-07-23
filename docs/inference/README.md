# Inference

Inference combines a camera-null model and camera-display projector for the
same device. Portable pretrained weights for both supported devices are
included under `checkpoints/`.

## Command

```bash
python -m color_pass_through.cli.predict \
  --config configs/devices/xiaomi_17promax.yaml \
  --input "$CPT_DATA_ROOT/projector/xiaomi_17promax/derived/v1/Image_Pairs_Flow/0900_CameraRaw.npy" \
  --phi 0.075 0.025 0.025 \
  --output runs/inference/output.png \
  --device cuda
```

Use checkpoints trained with the same device YAML. The command writes the RGB
PNG plus `output.png.json`, which records the configuration, inputs,
checkpoints, observer coefficients, and correction sign.

The CLI reads bundled checkpoint paths from the device YAML. Use
`--projector-checkpoint` and `--camera-null-checkpoint` to override both paths
when evaluating newly trained models.

## Bundled models

| Device config | Projector | Camera-null |
|---|---|---|
| `xiaomi_17promax.yaml` | `xiaomi_projector.ckpt` | `xiaomi_camera_null.ckpt` |
| `huawei_pura70pro.yaml` | `huawei_projector.ckpt` | `huawei_camera_null.ckpt` |

The checkpoints contain tensors only and do not embed source-machine paths or
training-framework objects. Their checksums and validation metrics are in
[`checkpoints/MANIFEST.json`](../../checkpoints/MANIFEST.json).

## Inputs

### Processed RGB `.npy`

The array must be three-dimensional float RGB in either `H x W x 3` or
`3 x H x W` order. Values are expected in the same linear camera-RGB domain as
the projector training input. This is the simplest input for checking the
trained pipeline.

### DNG

For non-`.npy` input, the configured RAW processor performs device-specific
demosaicing, dark-frame correction, optional gain-map correction, cropping,
and resizing. DNG inference therefore requires the optional RAW dependencies
and the calibration files from the preprocessing package.

## Observer coefficient `phi`

`--phi R G B` supplies the three observer correction coefficients. The example
is a valid invocation, not a universal calibration. Determine coefficients for
your observer and experimental setup with the calibration CLI:

```bash
python -m color_pass_through.cli.calibrate_observer --help
```

The permitted search interval and correction sign are defined under
`observer` in the device YAML.

## CPU inference

Use `--device cpu` for CPU execution. This is slower but useful for portability
checks. Checkpoints are loaded into ordinary PyTorch modules; tiny-cuda-nn is
not required for normal inference.
