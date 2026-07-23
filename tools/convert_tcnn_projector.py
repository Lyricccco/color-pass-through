#!/usr/bin/env python3
"""Convert the archived Fourier + FullyFusedMLP projector to pure PyTorch.

The known paper run has 36 encoded inputs, width 128, two hidden layers and
three outputs.  tiny-cuda-nn pads the input to 48 and output to 16, producing
24,576 packed parameters.  This converter validates the unpacking numerically
against a live tiny-cuda-nn network before writing a portable checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a tcnn projector checkpoint")
    parser.add_argument("--input", required=True, help="Archived Lightning checkpoint")
    parser.add_argument("--output", required=True, help="Portable PyTorch checkpoint")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    try:
        import tinycudann as tcnn
    except ImportError as error:
        raise RuntimeError(
            "This one-time converter requires tiny-cuda-nn; normal training and "
            "inference do not."
        ) from error

    from color_pass_through.models.projector import (
        CameraDisplayProjector,
        ProjectorSettings,
    )

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if "mlp.params" not in state:
        raise KeyError("Checkpoint does not contain the archived 'mlp.params' tensor")
    packed = state["mlp.params"].detach().float().flatten()
    if packed.numel() != 24576:
        raise ValueError(
            f"Expected 24,576 packed parameters for the paper projector, "
            f"got {packed.numel()}"
        )

    width, encoded, padded_input, padded_output = 128, 36, 48, 16
    first_size = width * padded_input
    hidden_size = width * width
    first = packed[:first_size].reshape(width, padded_input)
    hidden = packed[first_size : first_size + hidden_size].reshape(width, width)
    output = packed[first_size + hidden_size :].reshape(padded_output, width)

    settings = ProjectorSettings(
        input_mode="raw_green_context",
        encoding="fourier",
        num_frequencies=4,
        hidden_dim=width,
        hidden_layers=2,
        output_activation="none",
    )
    model = CameraDisplayProjector(settings)
    linears = [module for module in model.mlp if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        linears[0].weight.copy_(first[:, :encoded])
        linears[0].bias.zero_()
        linears[1].weight.copy_(hidden)
        linears[1].bias.zero_()
        linears[2].weight.copy_(output[:3])
        linears[2].bias.zero_()

    if not torch.cuda.is_available():
        raise RuntimeError("Numerical conversion validation requires CUDA")
    legacy = tcnn.Network(
        n_input_dims=encoded,
        n_output_dims=3,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": width,
            "n_hidden_layers": 2,
        },
    ).cuda()
    legacy.load_state_dict({"params": packed.cuda()})
    model.cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(7)
    features = torch.rand(args.samples, 4, generator=generator, device="cuda")
    with torch.inference_mode():
        encoded_features = model.encoder(features)
        reference = legacy(encoded_features).float()
        converted = model.mlp(encoded_features).float()
    maximum_error = float((reference - converted).abs().max().cpu())
    mean_error = float((reference - converted).abs().mean().cpu())
    if maximum_error > args.tolerance and not args.allow_mismatch:
        raise RuntimeError(
            f"Conversion validation failed: max error {maximum_error:.6g} > "
            f"{args.tolerance}. No checkpoint was written."
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "color-pass-through-pytorch-projector-v1",
            "state_dict": model.cpu().state_dict(),
            "settings": asdict(settings),
            "conversion": {
                "source": str(Path(args.input).resolve()),
                "max_abs_error": maximum_error,
                "mean_abs_error": mean_error,
                "samples": args.samples,
            },
        },
        target,
    )
    print(
        f"saved {target} (max_abs_error={maximum_error:.6g}, "
        f"mean_abs_error={mean_error:.6g})"
    )


if __name__ == "__main__":
    main()
