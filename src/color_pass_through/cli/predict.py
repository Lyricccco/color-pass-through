from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Color Pass-Through inference")
    parser.add_argument("--config", required=True, help="Device YAML")
    parser.add_argument("--input", required=True, help="DNG or processed RGB NPY")
    parser.add_argument(
        "--projector-checkpoint",
        default=None,
        help="Override models.projector.checkpoint from the device config",
    )
    parser.add_argument(
        "--camera-null-checkpoint",
        default=None,
        help="Override models.camera_null.checkpoint from the device config",
    )
    parser.add_argument("--phi", type=float, nargs=3, required=True, metavar=("R", "G", "B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np
    import torch
    from PIL import Image

    from color_pass_through.checkpoints import load_model_weights
    from color_pass_through.config import load_yaml
    from color_pass_through.models.camera_null import CameraNullModel
    from color_pass_through.models.projector import (
        CameraDisplayProjector,
        ProjectorSettings,
    )
    from color_pass_through.pipeline import ColorPassThroughPipeline
    from color_pass_through.raw import RawProcessor

    device = torch.device(args.device)
    config = load_yaml(args.config)
    null_config = config["models"]["camera_null"]
    projector_config = config["models"]["projector"]
    projector_checkpoint = args.projector_checkpoint or projector_config.get(
        "checkpoint"
    )
    camera_null_checkpoint = args.camera_null_checkpoint or null_config.get(
        "checkpoint"
    )
    if not projector_checkpoint or not camera_null_checkpoint:
        raise ValueError(
            "Provide both checkpoints through the device config or CLI overrides"
        )
    null_model = CameraNullModel(
        null_config["camera_sensitivity"],
        null_config["pca_components"],
        num_components=int(null_config.get("num_components", 1)),
    )
    projector = CameraDisplayProjector(
        ProjectorSettings(
            input_mode=projector_config.get("input_mode", "raw_green_context"),
            encoding=projector_config.get("encoding", "fourier"),
            num_frequencies=int(projector_config.get("num_frequencies", 4)),
            hidden_dim=int(projector_config.get("hidden_dim", 128)),
            hidden_layers=int(projector_config.get("hidden_layers", 2)),
            output_activation=projector_config.get("output_activation", "none"),
        )
    )
    load_model_weights(null_model, camera_null_checkpoint)
    load_model_weights(projector, projector_checkpoint)
    null_model.to(device).eval()
    projector.to(device).eval()

    source = Path(args.input)
    if source.suffix.lower() == ".npy":
        array = np.load(source, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC/CHW processed RGB, got {array.shape}")
        if array.shape[-1] == 3:
            array = np.moveaxis(array, -1, 0)
        camera_rgb = torch.from_numpy(array.astype(np.float32)).to(device)
        raw_manifest = {"source": str(source), "processed_input": True}
    else:
        camera_rgb, raw_manifest = RawProcessor(config["raw"], device).process(source)
    pipeline = ColorPassThroughPipeline(
        null_model,
        projector,
        correction_sign=int(config["observer"].get("correction_sign", -1)),
    )
    result = pipeline.predict(camera_rgb.unsqueeze(0), args.phi)
    image = (
        result.display_rgb[0]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    manifest = {
        "config": str(Path(args.config).resolve()),
        "input": raw_manifest,
        "phi": args.phi,
        "correction_sign": int(config["observer"].get("correction_sign", -1)),
        "projector_checkpoint": str(Path(projector_checkpoint).resolve()),
        "camera_null_checkpoint": str(Path(camera_null_checkpoint).resolve()),
        "output": str(output.resolve()),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
