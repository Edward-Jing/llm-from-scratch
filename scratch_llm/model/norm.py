"""Normalization layers."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        dim: Hidden size of the last tensor dimension.
        eps: Numerical stability epsilon.

    Expected forward input:
        x: Tensor shaped (..., dim).

    Expected forward output:
        Tensor with the same shape and dtype as x.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x across the last dimension."""
        normalized_x = x.float()/ torch.sqrt(x.float().pow(2).mean(dim = -1, keepdim=True) + self.eps)
        return (normalized_x * self.weight).type_as(x)