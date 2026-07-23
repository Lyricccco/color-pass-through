# Contributing

Keep changes aligned with the paper pipeline and cover numerical behavior with
tests.  Do not commit captures, checkpoints, user-study records, secrets or
author-machine paths.  New device support belongs in a new device YAML plus a
dataset validation fixture; it must not be implemented by editing Huawei or
Xiaomi branches in source.

Before submitting a change:

```bash
bash scripts/test_commands.sh static
```

When the complete dataset bundle is available, also run
`bash scripts/test_commands.sh data`. Dataset-free contributors are not
expected to download raw captures for ordinary code or documentation changes.

Any change to gain-map application, HSI normalization, observer correction
sign, wavelength selection or loss definitions must include before/after
metrics.
