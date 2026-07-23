# Camera-display pair construction

This directory contains the optional preprocessing implementation that creates
canonical camera-display projector pairs from synchronized capture triplets.
It is separate from `src/color_pass_through/` because training and inference
consume prepared pairs and do not import RAW denoising or optical-flow code.

## Entry points

- `build_xiaomi.py`: Xiaomi RAW processing and pair alignment.
- `build_huawei.py`: Huawei processing with the fitted vignetting gain map.
- `utils/`: RAW ISP, DualDn, image I/O, warping, and metric helpers.
- `raft/`: the RAFT implementation used for pair alignment.

Use the repository-level launcher instead of invoking these scripts directly:

```bash
python tools/prepare_projector_pairs.py \
  --config configs/devices/xiaomi_17promax.yaml \
  --output-name rebuilt-v1
```

The launcher forwards the device exposure, RAW and ISP ROIs, resize scale,
border crop, flow iterations, display blur, calibration assets, and pretrained
RAFT/DualDn paths from YAML.

## Outputs

```text
derived/<output-name>/
├── Image_Pairs_Flow/
│   ├── NNNN_CameraRaw.npy
│   ├── NNNN_DisplaysRGB.png
│   └── optional alignment diagnostics
├── Process/                         # ROI and pre-alignment diagnostics
├── Exposure.txt
├── Flow_validation.txt
└── launcher_manifest.json
```

Process and alignment diagnostic images are enabled by default. Pass
`--no-process-images` only when invoking a device script directly; canonical
pair arrays and display targets are always written.

The removed BGU fitting/slicing branch was inactive in the paper pair path and
produced no canonical training files. The public scripts retain only the
executed DualDn, ISP, ROI, RAFT, metric, diagnostic, and pair-export sequence.
