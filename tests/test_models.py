from pathlib import Path

import torch
from torch import nn

from color_pass_through.models.camera_null import CameraNullModel
from color_pass_through.models.encodings import FourierEncoding
from color_pass_through.models.projector import CameraDisplayProjector


ROOT = Path(__file__).resolve().parents[1]


def test_fourier_dimensions() -> None:
    encoding = FourierEncoding(4, 4)
    output = encoding(torch.zeros(2, 4))
    assert output.shape == (2, 36)


def test_projector_shape_and_gradient() -> None:
    model = CameraDisplayProjector()
    inputs = torch.rand(2, 3, 8, 10, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    output.mean().backward()
    assert inputs.grad is not None


class TinySpectrum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 31, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


def test_camera_null_shapes() -> None:
    model = CameraNullModel(
        ROOT / "artifacts/xiaomi/estimated_css.npy",
        ROOT / "artifacts/xiaomi/pca_components_n3.npy",
        backbone=TinySpectrum(),
    )
    camera = torch.rand(1, 3, 4, 5)
    spectrum = model(camera)
    _, null, coefficient = model.decompose(spectrum)
    assert spectrum.shape == (1, 31, 4, 5)
    assert null.shape == spectrum.shape
    assert coefficient.shape == (1, 1, 4, 5)
