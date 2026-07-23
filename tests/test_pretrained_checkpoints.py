from pathlib import Path

from color_pass_through.checkpoints import load_model_weights
from color_pass_through.models.camera_null import CameraNullModel
from color_pass_through.models.projector import (
    CameraDisplayProjector,
    ProjectorSettings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_preprocessing_checkpoints_are_bundled_and_configured() -> None:
    expected = {
        "flow_checkpoint: repo://checkpoints/raft_projector_alignment.pth",
        "denoise_checkpoint: repo://checkpoints/dualdn_raw_denoising.pth",
    }
    for filename in ("xiaomi_17promax.yaml", "huawei_pura70pro.yaml"):
        text = (ROOT / "configs/devices" / filename).read_text(encoding="utf-8")
        assert all(line in text for line in expected)
    assert (ROOT / "checkpoints/raft_projector_alignment.pth").is_file()
    assert (ROOT / "checkpoints/dualdn_raw_denoising.pth").is_file()


def test_bundled_projector_checkpoints_load_strictly() -> None:
    for device in ("xiaomi", "huawei"):
        model = CameraDisplayProjector(ProjectorSettings())
        load_model_weights(model, ROOT / f"checkpoints/{device}_projector.ckpt")


def test_bundled_camera_null_checkpoints_load_strictly() -> None:
    for device in ("xiaomi", "huawei"):
        model = CameraNullModel(
            ROOT / f"artifacts/{device}/estimated_css.npy",
            ROOT / f"artifacts/{device}/pca_components_n3.npy",
        )
        load_model_weights(model, ROOT / f"checkpoints/{device}_camera_null.ckpt")
