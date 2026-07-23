# Testing

These checks are intended for contributors and release packaging. They do not
contain machine-specific experiment paths or reproduce an internal validation
session.

## Static and unit tests

```bash
bash scripts/test_commands.sh static
```

This compiles Python sources, checks CLI imports, and runs the unit-test suite.
The unit tests use synthetic inputs and do not require the downloaded dataset.

Run pytest directly when iterating on one area:

```bash
python -m pytest -q
python -m pytest -q tests/test_projector_visualization.py
ruff check src tests tools \
  preprocessing/projector_pairs/build_xiaomi.py \
  preprocessing/projector_pairs/build_huawei.py
```

## Full dataset validation

After downloading both Core Training Data and Preprocessing Sources:

```bash
export CPT_DATA_ROOT=/path/to/color-pass-through
bash scripts/test_commands.sh data
```

This mode checks both supported devices and therefore requires the complete
bundle, including raw captures and preprocessing weights.

## One-batch training smoke test

With the Core Training Data package available:

```bash
bash scripts/test_commands.sh train-smoke xiaomi
bash scripts/test_commands.sh train-smoke huawei
```

For easier exception traces, the scripted smoke run uses one worker and one
training batch. Successful smoke training is not evidence of paper-level model
quality; full training and held-out evaluation are still required.

## Public-release hygiene

Before publishing, verify that the staged repository contains no experiment
runs, model weights without redistribution permission, Python caches, local
absolute paths, credentials, internal hostnames, or private data-copy scripts.
The four inference checkpoints and two preprocessing checkpoints in
`checkpoints/` are intentional release assets and are covered by
`checkpoints/MANIFEST.json` and `THIRD_PARTY_LICENSES.md`.
