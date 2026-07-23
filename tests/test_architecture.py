from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "src/color_pass_through"


def _python_texts(root: Path) -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_runtime_has_no_hydra_family_imports() -> None:
    forbidden = ("import hydra", "from hydra", "omegaconf", "rootutils")
    hits = [
        str(path.relative_to(ROOT))
        for path, text in _python_texts(ACTIVE)
        if any(token in text.lower() for token in forbidden)
    ]
    assert hits == []


def test_runtime_has_no_tinycudann_import() -> None:
    hits = [
        str(path.relative_to(ROOT))
        for path, text in _python_texts(ACTIVE)
        if "import tinycudann" in text
    ]
    assert hits == []


def test_lightning_is_isolated_to_training_and_train_cli() -> None:
    allowed = {
        Path("src/color_pass_through/training/lightning.py"),
        Path("src/color_pass_through/cli/train_projector.py"),
        Path("src/color_pass_through/cli/train_camera_null.py"),
    }
    hits = {
        path.relative_to(ROOT)
        for path, text in _python_texts(ACTIVE)
        if "import lightning" in text or "from lightning" in text
    }
    assert hits <= allowed


def test_pair_entrypoints_do_not_override_cuda_device() -> None:
    entrypoints = (
        ROOT / "preprocessing/projector_pairs/build_xiaomi.py",
        ROOT / "preprocessing/projector_pairs/build_huawei.py",
    )
    forbidden = ("CUDA_VISIBLE_DEVICES", "device_count.cache_clear")
    hits = {
        str(path.relative_to(ROOT)): token
        for path in entrypoints
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert hits == {}


def test_pair_outputs_are_rooted_at_explicit_output() -> None:
    entrypoints = (
        ROOT / "preprocessing/projector_pairs/build_xiaomi.py",
        ROOT / "preprocessing/projector_pairs/build_huawei.py",
    )
    required = (
        'osp.join(OutputFolder, "Process")',
        'osp.join(OutputFolder, "Image_Pairs_Flow")',
    )
    forbidden = (
        'osp.join(args.dataset, args.output, "Process")',
        'osp.join(args.dataset, args.output, "Image_Pairs_Flow")',
    )
    for path in entrypoints:
        text = path.read_text(encoding="utf-8")
        assert all(token in text for token in required)
        assert all(token not in text for token in forbidden)
