"""Rotary position embedding contracts."""

from __future__ import annotations

from typing import Optional

import torch


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device | str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine RoPE tables.
    w_i = \frac{1}{theta^{2i/d}}
    d = head_dim
    i = 0, 1,..., head_dim/2 -1
    angle(p,i) = p * wi
    -> cos(angle(p,i)) , sin(angle(p,i))
    Args:
        head_dim: Per-head hidden dimension. Must be even.
                  Why even? Because Rope use paired sin-cos to represent the position
                  [x0, x1, x2, x3, x4, x5] -> (x0, x1), (x2, x3), (x4, x5)
        max_seq_len: Number of positions to precompute.
        theta: RoPE base frequency.
        device: Optional torch device for created tensors.

    Returns:
        Tuple (freqs_cos, freqs_sin), each shaped (max_seq_len, head_dim / 2).
    """

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even but got {head_dim} as odds")

    # The following dim_indices will return 0, 2, 4, ..., head_dim - 2.
    dim_indices = torch.arange(0, head_dim, 2, device=device)

    inv_freq = 1.0 / (theta ** (dim_indices.float() / head_dim))
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions.float(), inv_freq)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a RoPE table so it can broadcast against q/k halves.

    Args:
        freqs: Tensor shaped (seq_len, head_dim / 2).
        x: Query or key half tensor shaped (batch, seq_len, heads, head_dim / 2).

    Returns:
        View of freqs shaped (1, seq_len, 1, head_dim / 2).
    """

    if freqs.shape != (x.shape[1], x.shape[-1]):
        raise ValueError(
            f"freqs must have shape {(x.shape[1], x.shape[-1])}, "
            f"but got {freqs.shape}"
        )

    return freqs.view(1, x.shape[1], 1, x.shape[-1])


def apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors.

    Args:
        q: Query tensor shaped (batch, seq_len, n_heads, head_dim).
        k: Key tensor shaped (batch, seq_len, n_kv_heads, head_dim).
        freqs_cos: Cosine table for current sequence length.
        freqs_sin: Sine table for current sequence length.

    Returns:
        Tuple (rotated_q, rotated_k) with original shapes and dtypes.
    """

    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same batch size")

    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k must have the same sequence length")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head_dim")

    head_dim = q.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, but got {head_dim}")

    seq_len = q.shape[1]

    # Use only the frequency rows needed for the current sequence length.
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]

    # Split the last dimension into even and odd coordinates.
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    # Reshape cos/sin tables so they can broadcast over batch and heads.
    q_cos = reshape_for_broadcast(freqs_cos, q_even)
    q_sin = reshape_for_broadcast(freqs_sin, q_even)
    k_cos = reshape_for_broadcast(freqs_cos, k_even)
    k_sin = reshape_for_broadcast(freqs_sin, k_even)

    # Apply 2D rotation to query and key pairs.
    """
    [ cos  -sin ] [x]
    [ sin   cos ] [y]
    """
    q_rotated_even = q_even * q_cos - q_odd * q_sin
    q_rotated_odd = q_even * q_sin + q_odd * q_cos
    k_rotated_even = k_even * k_cos - k_odd * k_sin
    k_rotated_odd = k_even * k_sin + k_odd * k_cos

    # Interleave even and odd coordinates back into the original head_dim.
    rotated_q = torch.stack((q_rotated_even, q_rotated_odd), dim=-1).flatten(-2)
    rotated_k = torch.stack((k_rotated_even, k_rotated_odd), dim=-1).flatten(-2)

    return rotated_q.type_as(q), rotated_k.type_as(k)