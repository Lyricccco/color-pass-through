import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from color_pass_through.cli.train_camera_null import build_parser
from color_pass_through.training.visualization import CameraNullVisualizer


def test_camera_null_visualization_defaults_match_migrated_configuration() -> None:
    args = build_parser().parse_args(["--config", "device.yaml", "--output", "run"])
    assert args.validation_visualization
    assert args.visualize_samples == 182
    assert args.visualize_every_n_epochs == 10
    assert args.save_validation_csv


def test_camera_null_visualizer_exports_reference_diagnostics() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory)
        visualizer = CameraNullVisualizer(output, max_samples=1, every_n_epochs=10)
        visualizer.start_epoch(9, sanity_checking=False)
        target = torch.full((1, 31, 8, 12), 0.4)
        prediction = torch.full((1, 31, 8, 12), 0.3)
        target_rgb = torch.full((1, 3, 8, 12), 0.6)
        prediction_rgb = torch.full((1, 3, 8, 12), 0.5)
        target_coefficient = torch.full((1, 1, 8, 12), 0.2)
        predicted_coefficient = torch.full((1, 1, 8, 12), 0.1)
        visualizer.save_batch(
            {"Meta": {"id": ["sample"]}},
            prediction,
            target,
            prediction_rgb,
            target_rgb,
            predicted_coefficient,
            target_coefficient,
            0,
        )
        visualizer.finish_epoch()

        epoch = output / "epoch_009"
        required = (
            "sample_rangeRGB_GT.png",
            "sample_rangeRGB_PRED.png",
            "sample_rangeRGB_DIFF.png",
            "sample_rangeRGB_GT_PRED_DIFF.png",
            "sample_coef1_GT_abs.png",
            "sample_coef1_PRED_abs.png",
            "sample_coef1_DIFF_abs.png",
            "sample_coef1_GT_PRED_DIFF_abs.png",
        )
        assert all((epoch / name).is_file() for name in required)
        assert Image.open(epoch / required[3]).size == (36, 8)
        with (output / "val_results_epoch009.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["base_name"] == "sample"
        assert set(rows[0]) == {
            "idx",
            "base_name",
            "H",
            "W",
            "psnr_spec",
            "psnr_coef",
            "psnr_rgb",
        }
