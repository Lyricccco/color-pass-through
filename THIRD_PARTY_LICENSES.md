# Third-party components and data

This document is an attribution and release checklist; it does not relicense
third-party software, weights, or datasets.

| Component | Upstream project | Use in this repository |
|---|---|---|
| DualDn | OpenImagingLab/DualDn | Optional RAW denoising during pair preparation; [CC BY-NC 4.0](third_party/licenses/DualDn.txt) |
| MST++ | caiyuanhao1998/MST-plus-plus | Camera-null spectrum-prediction backbone; [MIT](third_party/licenses/MST-plus-plus.txt) |
| RAFT | Princeton Vision & Learning Lab RAFT | Optional optical-flow alignment during pair preparation; [BSD 3-Clause](third_party/licenses/RAFT.txt) |
| tiny-cuda-nn | NVlabs/tiny-cuda-nn | Optional historical-checkpoint conversion only |
| ARAD-1K, CAVE, ICVL | Respective dataset owners | Hyperspectral training sources |
| DIV2K | Respective dataset owners | Display stimuli |

Normal projector training, camera-null training, and inference do not require
tiny-cuda-nn. The bundled `dualdn_raw_denoising.pth` is licensed for
noncommercial use, so distributing this repository as a whole includes a
CC BY-NC 4.0 asset even though the original project code is MIT-licensed. The
bundled RAFT and DualDn checkpoints remain governed by their upstream licenses
and terms.

Before distributing a release artifact, the publisher must include every
required upstream notice and verify redistribution permission separately for:

- DualDn- and RAFT-derived code retained under `preprocessing/projector_pairs/`;
- pretrained RAFT and DualDn weights;
- camera captures and display stimuli;
- hyperspectral and ColorChecker-derived data.

Users are responsible for complying with the licenses and terms attached to
the dataset package they download.
