"""Feed-forward layer contracts."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

def derive_swiglu_hidden_dim(
    dim: int,
    hidden_dim: int | None = None,
    multiple_of: int = 64,
) -> int:
    """Derive the hidden size used by LLaMA-style SwiGLU MLPs.

    Args:
        dim: Residual stream hidden size.
        hidden_dim: Explicit hidden size. If provided, return it unchanged.
        multiple_of: Round derived size up to this multiple.

    Returns:
        Final hidden size.
    """
    if hidden_dim is not None:
            return hidden_dim

    hidden = int(2 * (4 * dim) / 3)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)

    return hidden

class SwiGLU(nn.Module):
    """LLaMA-style gated MLP.

    Args:
        dim: Residual stream hidden size.
        hidden_dim: Explicit hidden size or None to derive one.
        multiple_of: Round derived hidden size to this multiple.
        dropout: Dropout probability applied after the down projection.

    Expected forward input:
        x: Tensor shaped (batch, seq_len, dim).

    Expected forward output:
        Tensor shaped (batch, seq_len, dim).
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int | None,
            multiple_of: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = derive_swiglu_hidden_dim(
            dim=dim,
            hidden_dim=hidden_dim,
            multiple_of=multiple_of,
        )
        self.multiple_of = multiple_of
        self.dropout_p = dropout

        self.w1 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation."""

        gate = F.silu(self.w1(x))
        up = self.w3(x)
        out = self.w2(gate * up)
        out = self.dropout(out)

        return out