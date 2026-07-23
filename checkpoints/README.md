# Pretrained checkpoints

This directory contains every pretrained weight used by the released workflow:
four portable inference models and two shared pair-preprocessing models.

## Inference models

| Device | Camera-display projector | Camera-null model |
|---|---|---|
| Xiaomi 17 Pro Max | `xiaomi_projector.ckpt` | `xiaomi_camera_null.ckpt` |
| Huawei Pura 70 Pro | `huawei_projector.ckpt` | `huawei_camera_null.ckpt` |

The files contain model tensors only. Lightning callbacks, optimizer state,
Hydra objects, experiment directories, and source-machine paths were removed
during export. See [MANIFEST.json](MANIFEST.json) for architecture, validation,
source-training, checksum, and fitted-camera-asset metadata.

Projector and camera-null checkpoints are device-specific and must not be
mixed. Newly trained checkpoints are written below the selected training
`--output` directory. Use `tools/export_trained_checkpoint.py` to remove
Lightning optimizer and trainer state before replacing a release checkpoint.

## Pair-preprocessing models

| File | Purpose | License |
|---|---|---|
| `raft_projector_alignment.pth` | RAFT/FLOW alignment of camera and display captures | BSD 3-Clause |
| `dualdn_raw_denoising.pth` | DualDn RAW denoising before pair alignment | CC BY-NC 4.0 |

Both device configurations reference these readable filenames directly. They
are needed only when rebuilding pairs from original captures, not for training
from canonical pairs or for inference. The DualDn code and weight are
noncommercial assets; redistributors and users must comply with CC BY-NC 4.0.
The original upstream filenames and checksums are recorded in
[MANIFEST.json](MANIFEST.json).
