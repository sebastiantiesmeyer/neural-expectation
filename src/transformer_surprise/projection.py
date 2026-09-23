"""Fixed, deterministic activation projections."""

from typing import Optional

import torch
from torch import Tensor, nn


class FixedRandomProjection(nn.Module):
    """Project hidden states with a fixed row-normalized Gaussian matrix."""

    def __init__(self, input_dim: int, output_dim: int = 32, seed: int = 42) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(output_dim, input_dim, generator=generator)
        matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
        self.register_buffer("matrix", matrix)

    @property
    def output_dim(self) -> int:
        return self.matrix.shape[0]

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return the last dimension projected into ``output_dim``."""
        return hidden_states.to(self.matrix.dtype) @ self.matrix.T

    def project_cpu(self, hidden_states: Tensor) -> Tensor:
        """Project and detach activations for CPU-side sidecar processing."""
        return self(hidden_states).detach().to("cpu")
