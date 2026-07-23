from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from color_pass_through.data import ProjectorPairDataset


def _fixture(root: Path) -> Path:
    split = root / "split.txt"
    split.write_text("0000\n0001\n0002\n0003\n", encoding="utf-8")
    for index in range(4):
        camera = np.full((8, 12, 3), index / 10, dtype=np.float32)
        display = np.full((8, 12, 3), index * 10, dtype=np.uint8)
        np.save(root / f"{index:04d}_CameraRaw.npy", camera)
        Image.fromarray(display).save(root / f"{index:04d}_DisplaysRGB.png")
    return split


def test_full_frame_augmentation_preserves_batch_shape() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = ProjectorPairDataset(root, _fixture(root), augment=True)
        for _ in range(8):
            batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
            assert tuple(batch["Input"].shape) == (4, 3, 8, 12)
            assert tuple(batch["Target"].shape) == (4, 3, 8, 12)


def test_square_crop_allows_shape_safe_quarter_turns() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = ProjectorPairDataset(
            root, _fixture(root), random_crop=6, augment=True
        )
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        assert tuple(batch["Input"].shape) == (4, 3, 6, 6)
        assert tuple(batch["Target"].shape) == (4, 3, 6, 6)


def test_mixed_frame_widths_are_center_cropped_before_collation() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        split = root / "split.txt"
        split.write_text("0000\n0001\n0002\n0003\n", encoding="utf-8")
        for index, width in enumerate((12, 10, 12, 10)):
            camera = np.full((8, width, 3), index / 10, dtype=np.float32)
            display = np.full((8, width, 3), index * 10, dtype=np.uint8)
            np.save(root / f"{index:04d}_CameraRaw.npy", camera)
            Image.fromarray(display).save(root / f"{index:04d}_DisplaysRGB.png")

        dataset = ProjectorPairDataset(root, split, augment=True)
        batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=2)))
        assert dataset.frame_shape == (8, 10)
        assert tuple(batch["Input"].shape) == (4, 3, 8, 10)
        assert tuple(batch["Target"].shape) == (4, 3, 8, 10)
