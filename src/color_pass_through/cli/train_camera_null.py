from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the camera-null predictor")
    parser.add_argument("--config", required=True, help="Device YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export camera-null validation diagnostics",
    )
    parser.add_argument("--visualize-samples", type=int, default=182)
    parser.add_argument("--visualize-every-n-epochs", type=int, default=10)
    parser.add_argument(
        "--save-validation-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.visualize_samples < 0:
        raise ValueError("--visualize-samples must be non-negative")
    if args.visualize_every_n_epochs <= 0:
        raise ValueError("--visualize-every-n-epochs must be positive")
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from color_pass_through.config import load_yaml
    from color_pass_through.data import HyperspectralPairDataset
    from color_pass_through.models.camera_null import CameraNullModel
    from color_pass_through.training.lightning import CameraNullTask
    from color_pass_through.training.visualization import CameraNullVisualizer

    L.seed_everything(args.seed, workers=True)
    config = load_yaml(args.config)
    data = config["data"]["camera_null"]
    null = config["models"]["camera_null"]
    train_set = HyperspectralPairDataset(
        data["camera_rgb"],
        data["spectrum"],
        data["validation_split"],
        include_split=False,
        crop_size=args.crop_size,
        augment=True,
    )
    validation_set = HyperspectralPairDataset(
        data["camera_rgb"],
        data["spectrum"],
        data["validation_split"],
        include_split=True,
    )
    output = Path(args.output)
    visualizer = (
        CameraNullVisualizer(
            output / "validation_visualizations",
            max_samples=args.visualize_samples,
            every_n_epochs=args.visualize_every_n_epochs,
            save_csv=args.save_validation_csv,
        )
        if args.validation_visualization
        else None
    )
    task = CameraNullTask(
        CameraNullModel(
            null["camera_sensitivity"],
            null["pca_components"],
            num_components=int(null.get("num_components", 1)),
        ),
        coefficient_weight=float(null.get("coefficient_weight", 1.0)),
        tv_weight=float(null.get("tv_weight", 0.01)),
        spectrum_weight=float(null.get("spectrum_weight", 1.0)),
        visualizer=visualizer,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 4),
    )
    callback = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="camera-null-{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    trainer_options = {}
    if args.limit_train_batches is not None:
        trainer_options["limit_train_batches"] = args.limit_train_batches
    trainer = L.Trainer(
        default_root_dir=output,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[callback],
        **trainer_options,
    )
    trainer.fit(task, train_loader, validation_loader)
    trainer.test(task, validation_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
