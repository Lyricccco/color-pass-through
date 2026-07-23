from __future__ import annotations

import math

import torch
from torch import nn


class IdentityEncoding(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = int(input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class FourierEncoding(nn.Module):
    """The Fourier feature mapping used by the paper projector."""

    def __init__(
        self,
        input_dim: int = 4,
        num_frequencies: int = 4,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.output_dim = self.input_dim * (
            2 * self.num_frequencies + int(self.include_input)
        )
        frequencies = (2.0 ** torch.arange(self.num_frequencies)) * math.pi
        self.register_buffer("frequencies", frequencies.float())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phases = inputs.unsqueeze(-2) * self.frequencies.view(-1, 1)
        features = [torch.sin(phases).flatten(-2), torch.cos(phases).flatten(-2)]
        if self.include_input:
            features.insert(0, inputs)
        return torch.cat(features, dim=-1)
