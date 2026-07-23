from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_model_weights(
    model: nn.Module,
    checkpoint: str | Path,
    prefixes: tuple[str, ...] = ("model.",),
    strict: bool = True,
) -> None:
    """Load a plain or Lightning-style checkpoint into a PyTorch module."""
    payload: Any = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")
    model_keys = set(model.state_dict())
    if set(state).isdisjoint(model_keys):
        for prefix in prefixes:
            stripped = {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if stripped and not set(stripped).isdisjoint(model_keys):
                state = stripped
                break
    model.load_state_dict(state, strict=strict)
