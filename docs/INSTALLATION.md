# Environment setup

## Recommended environment

- Linux
- Python 3.10
- NVIDIA GPU for training and preprocessing
- A CUDA-compatible PyTorch installation

Create an isolated environment and install the complete dependency set:

```bash
conda create -n color-pass-through python=3.10 pip -y
conda activate color-pass-through
pip install -r requirements.txt
```

The requirements file installs training, inference, preprocessing, and test
dependencies. The package can also be installed in editable form:

```bash
pip install -e '.[raw,train,prepare,dev]'
```

## Verify PyTorch and CUDA

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

If importing PyTorch reports an unresolved symbol in a CUDA library, the
environment contains incompatible CUDA runtime packages. Reinstall PyTorch and
its NVIDIA dependencies from one consistent channel or wheel index; do not mix
CUDA package generations in the same environment.

## Optional system dependency

Rebuilding data from DNG files requires ExifTool:

```bash
exiftool -ver
```

Prepared-pair training and `.npy` inference do not require ExifTool.

## Import check

From the repository root:

```bash
python -c "import color_pass_through; print('import ok')"
python -m color_pass_through.cli.train_projector --help
python -m color_pass_through.cli.train_camera_null --help
python -m color_pass_through.cli.predict --help
```

If the package is not installed, use `pip install -e .` rather than adding a
machine-specific source path to `PYTHONPATH`.

