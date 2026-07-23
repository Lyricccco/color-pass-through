#!/usr/bin/env python3
"""Export a training checkpoint as a portable model-only state dictionary."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strip a projector or camera-null training checkpoint for release"
    )
    parser.add_argument("--config", required=True, help="Matching device YAML")
    parser.add_argument(
        "--model",
        required=True,
        choices=("projector", "camera-null"),
        help="Model architecture to instantiate",
    )
    parser.add_argument("--input", required=True, help="Lightning or plain checkpoint")
    parser.add_argument("--output", required=True, help="Portable checkpoint to write")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of an existing output file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}; pass --force to replace it")

    import torch

    from color_pass_through.checkpoints import load_model_weights
    from color_pass_through.config import load_yaml

    config = load_yaml(args.config)
    if args.model == "projector":
        from color_pass_through.models.projector import (
            CameraDisplayProjector,
            ProjectorSettings,
        )

        model_config = config["models"]["projector"]
        model = CameraDisplayProjector(
            ProjectorSettings(
                input_mode=model_config.get("input_mode", "raw_green_context"),
                encoding=model_config.get("encoding", "fourier"),
                num_frequencies=int(model_config.get("num_frequencies", 4)),
                hidden_dim=int(model_config.get("hidden_dim", 128)),
                hidden_layers=int(model_config.get("hidden_layers", 2)),
                output_activation=model_config.get("output_activation", "none"),
            )
        )
    else:
        from color_pass_through.models.camera_null import CameraNullModel

        model_config = config["models"]["camera_null"]
        model = CameraNullModel(
            model_config["camera_sensitivity"],
            model_config["pca_components"],
            num_components=int(model_config.get("num_components", 1)),
        )

    load_model_weights(model, args.input)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)

    # Reload strictly before reporting success so an incomplete export cannot pass silently.
    load_model_weights(model, output)
    print(f"Exported {len(state)} tensors to {output}")


if __name__ == "__main__":
    main()
