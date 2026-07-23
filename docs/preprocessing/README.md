# Optional data preparation

Most users should train from the canonical prepared pairs. This section is for
researchers who want to reproduce how raw capture data becomes training data.

The projector preparation pipeline has two stages:

1. [Build calibration assets](STAGE_1_CALIBRATION.md): estimate dark-frame
   correction and, where configured, a vignetting gain map.
2. [Build aligned projector pairs](STAGE_2_PROJECTOR_PAIRS.md): denoise and
   process RAW captures, align them to display targets with optical flow, and
   export canonical pair files.

Camera-null preparation is independent:

```text
ColorChecker captures ──> fitted camera spectral sensitivity
hyperspectral cubes ─────> null-space PCA + simulated device RGB
```

The individual commands are indexed in [Command-line tools](../tools/README.md).

## Requirements

- the optional **Preprocessing Sources** dataset package;
- the bundled RAFT and DualDn weights under `checkpoints/`;
- an NVIDIA GPU for the archived denoising and optical-flow pipeline;
- ExifTool for DNG metadata;
- dependencies from `requirements.txt`.

All outputs should be written below a new derived directory. The pair launcher
refuses to overwrite the packaged canonical `derived/v1` directory.

## Implementation layout

The pair-generation implementation lives under
`preprocessing/projector_pairs/`, with explicit Xiaomi and Huawei entrypoints,
shared RAW/denoising helpers, and the RAFT alignment implementation. See its
[directory README](../../preprocessing/projector_pairs/README.md). Training and
inference use the maintained package under `src/color_pass_through/` and do not
import preprocessing code.
