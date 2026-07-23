from pathlib import Path
from tempfile import TemporaryDirectory
from inspect import signature

import yaml

from color_pass_through.config import resolve_paths
from color_pass_through.models.camera_null import CameraNullModel
from color_pass_through.training.lightning import CameraNullTask


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_repo_and_dataset_paths() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        repository = root / "repository"
        dataset = root / "dataset"
        resolved = resolve_paths(
            {
                "source": "repo://configs/example.yaml",
                "pairs": "dataset://projector/device/derived/v1",
            },
            repository_root=repository,
            dataset_root=dataset,
        )
        assert resolved == {
            "source": str((repository / "configs/example.yaml").resolve()),
            "pairs": str((dataset / "projector/device/derived/v1").resolve()),
        }


def test_dataset_path_requires_root() -> None:
    with TemporaryDirectory() as directory:
        try:
            resolve_paths(
                "dataset://camera_null/hyperspectral_gt/v1",
                repository_root=Path(directory),
            )
        except ValueError as error:
            assert "CPT_DATA_ROOT" in str(error)
        else:
            raise AssertionError("dataset:// path unexpectedly resolved without a root")


def test_camera_null_defaults_match_paper_loss_weights() -> None:
    expected = {
        "coefficient_weight": 1.0,
        "tv_weight": 0.01,
        "spectrum_weight": 1.0,
    }
    for device in ("xiaomi_17promax", "huawei_pura70pro"):
        config = yaml.safe_load(
            (ROOT / f"configs/devices/{device}.yaml").read_text(encoding="utf-8")
        )
        null = config["models"]["camera_null"]
        assert {name: float(null[name]) for name in expected} == expected

    experiment = yaml.safe_load(
        (ROOT / "configs/experiments/camera_null.yaml").read_text(encoding="utf-8")
    )
    assert experiment["loss"] == {
        "coefficient_l1": 1.0,
        "coefficient_tv": 0.01,
        "spectrum_l1": 1.0,
    }
    assert signature(CameraNullModel.losses).parameters["tv_weight"].default == 0.01
    assert signature(CameraNullTask).parameters["tv_weight"].default == 0.01
