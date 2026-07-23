# Stage 2: aligned projector pairs

This stage converts synchronized capture triplets into the camera/display
pairs consumed by projector training.

## Inputs

For one device, the configured capture root must contain matching identifiers:

```text
raw/
├── Camera_Raw/*.dng
├── Camera_ISP/*.jpg
└── Display_sRGB/*.png
```

The selected YAML must also resolve:

- fitted dark-frame RAW and RGB files;
- a fitted gain map when enabled;
- the bundled `checkpoints/raft_projector_alignment.pth`;
- the bundled `checkpoints/dualdn_raw_denoising.pth`;
- device-specific exposure, crop, and ROI settings.

## Inspect before running

First print the exact generated command without writing data:

```bash
python tools/prepare_projector_pairs.py \
  --config configs/devices/xiaomi_17promax.yaml \
  --output-name rebuilt-v1 \
  --dry-run
```

## Build the pairs

```bash
python tools/prepare_projector_pairs.py \
  --config configs/devices/xiaomi_17promax.yaml \
  --output-name rebuilt-v1
```

For Huawei, substitute `configs/devices/huawei_pura70pro.yaml`. Output is
created at:

```text
$CPT_DATA_ROOT/projector/<device>/derived/rebuilt-v1/
```

The launcher intentionally refuses `--output-name v1`, which protects the
packaged canonical pairs.

## Processing sequence

The behavior-preserving preparation script performs the paper preprocessing
sequence:

1. read capture and exposure metadata;
2. denoise the RAW frame with DualDn;
3. apply device RAW processing and calibration corrections;
4. crop the device-specific regions of interest;
5. align camera and display content with RAFT optical flow;
6. apply the configured border crop and display blur;
7. save `NNNN_CameraRaw.npy` and `NNNN_DisplaysRGB.png` pairs;
8. save ROI, pre-alignment, and aligned process images by default;
9. record RAW/display and ISP/display PSNR and SSIM;
10. save a launcher manifest describing the inputs and generated command.

The implementation is organized under
`preprocessing/projector_pairs/`. Its inactive BGU experimental branch and
unused alternative resize/denoising code were removed; these paths did not
produce the canonical paper pairs.

## Promote a rebuilt dataset

Do not overwrite `derived/v1` during an experiment. Compare rebuilt counts,
shapes, representative alignments, and numerical statistics first. If you
publish a new canonical revision, give it a new version directory and update
the device YAML in the same release.
