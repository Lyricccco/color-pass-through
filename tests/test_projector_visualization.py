from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import numpy as np
from PIL import Image

from color_pass_through.cli.train_projector import build_parser
from color_pass_through.training.visualization import (
    ProjectorVisualizer,
    projector_error_heatmap,
)


def test_projector_visualizer_writes_complete_image_group() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory)
        batch = {
            "Input": torch.full((1, 3, 8, 12), 0.25),
            "Target": torch.full((1, 3, 8, 12), 0.75),
            "Meta": {"id": ["0006"]},
        }
        prediction = torch.full((1, 3, 8, 12), 0.5)
        ProjectorVisualizer(output).save("test_best", batch, prediction, 0)

        destination = output / "test_best"
        for name in ("input", "prediction", "target", "diff", "comparison"):
            assert (destination / f"0006_{name}.png").is_file()
        assert Image.open(destination / "0006_input.png").size == (12, 8)
        assert Image.open(destination / "0006_comparison.png").size == (48, 32)
        assert '"mae": 0.25' in (destination / "manifest.json").read_text()


def test_projector_error_heatmap_matches_legacy_absolute_lut() -> None:
    target = torch.zeros(3, 1, 3)
    prediction = torch.tensor([0.0, 0.04, 0.08]).view(1, 1, 3).repeat(3, 1, 1)
    pixels = np.asarray(projector_error_heatmap(prediction, target))
    assert pixels.tolist() == [
        [[0, 0, 89], [128, 255, 127], [255, 0, 0]],
    ]


def test_projector_visualization_cli_options() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "device.yaml",
            "--output",
            "run",
            "--visualize-samples",
            "10",
            "--visualize-every-n-epochs",
            "5",
            "--save-test-predictions",
        ]
    )
    assert args.visualize_samples == 10
    assert args.visualize_every_n_epochs == 5
    assert args.save_test_predictions


def test_projector_visualization_is_enabled_by_default_and_can_be_disabled() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["--config", "device.yaml", "--output", "run"])
    assert defaults.visualize_samples == 10
    assert defaults.visualize_every_n_epochs == 5
    assert defaults.save_test_predictions

    disabled = parser.parse_args(
        [
            "--config",
            "device.yaml",
            "--output",
            "run",
            "--visualize-samples",
            "0",
            "--no-save-test-predictions",
        ]
    )
    assert disabled.visualize_samples == 0
    assert not disabled.save_test_predictions
