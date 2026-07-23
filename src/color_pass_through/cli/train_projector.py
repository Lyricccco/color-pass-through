from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the camera-display projector")
    parser.add_argument("--config", required=True, help="Device YAML")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=10,
        help="Save visualizations for the first N test-split samples (default: 10; 0 disables)",
    )
    parser.add_argument(
        "--visualize-every-n-epochs",
        type=int,
        default=5,
        help="Validation visualization interval; epoch 0 is always included",
    )
    parser.add_argument(
        "--save-test-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save visualizations after loading the best checkpoint for test",
    )
    parser.add_argument(
        "--lut-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the projector as a CUBE LUT and 3D quiver visualization",
    )
    parser.add_argument("--lut-size", type=int, default=33)
    parser.add_argument("--lut-every-n-epochs", type=int, default=10)
    parser.add_argument("--lut-chunk-pixels", type=int, default=262_144)
    parser.add_argument("--lut-vis-subsample", type=int, default=3)
    parser.add_argument("--lut-vis-vec-scale", type=float, default=1.0)
    parser.add_argument("--lut-vis-arrow-length", type=float, default=0.08)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.visualize_samples < 0:
        raise ValueError("--visualize-samples must be non-negative")
    if args.visualize_every_n_epochs <= 0:
        raise ValueError("--visualize-every-n-epochs must be positive")
    for name in ("lut_size", "lut_every_n_epochs", "lut_chunk_pixels", "lut_vis_subsample"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from color_pass_through.config import load_yaml
    from color_pass_through.data import ProjectorPairDataset
    from color_pass_through.models.projector import (
        CameraDisplayProjector,
        ProjectorSettings,
    )
    from color_pass_through.training.lightning import ProjectorTask
    from color_pass_through.training.visualization import (
        ProjectorLUTExporter,
        ProjectorVisualizer,
    )

    L.seed_everything(args.seed, workers=True)
    config = load_yaml(args.config)
    data = config["data"]["projector"]
    model_config = config["models"]["projector"]
    if args.lut_visualization and model_config.get("input_mode") == "hdr_assisted":
        raise ValueError(
            "Projector LUT export is unavailable for hdr_assisted input mode"
        )
    train_set = ProjectorPairDataset(
        data["pairs"],
        data["train_split"],
        random_crop=args.crop_size,
        augment=True,
        frame_shape=data.get("frame_shape"),
    )
    validation_set = ProjectorPairDataset(
        data["pairs"],
        data["test_split"],
        frame_shape=data.get("frame_shape"),
    )
    settings = ProjectorSettings(
        input_mode=model_config.get("input_mode", "raw_green_context"),
        encoding=model_config.get("encoding", "fourier"),
        num_frequencies=int(model_config.get("num_frequencies", 4)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        hidden_layers=int(model_config.get("hidden_layers", 2)),
        output_activation=model_config.get("output_activation", "none"),
    )
    output = Path(args.output)
    visualizer = (
        ProjectorVisualizer(output / "visualizations")
        if args.visualize_samples > 0
        else None
    )
    lut_exporter = (
        ProjectorLUTExporter(
            output / "visualizations/luts",
            size=args.lut_size,
            every_n_epochs=args.lut_every_n_epochs,
            chunk_pixels=args.lut_chunk_pixels,
            vis_subsample=args.lut_vis_subsample,
            vis_vec_scale=args.lut_vis_vec_scale,
            vis_arrow_length=args.lut_vis_arrow_length,
        )
        if args.lut_visualization
        else None
    )
    task = ProjectorTask(
        CameraDisplayProjector(settings),
        visualizer=visualizer,
        visualize_samples=args.visualize_samples,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
        save_test_predictions=args.save_test_predictions,
        lut_exporter=lut_exporter,
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
        filename="projector-{epoch:03d}",
        monitor="val/psnr",
        mode="max",
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
