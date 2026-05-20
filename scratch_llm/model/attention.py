"""Attention layer contracts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import ModelConfig

import math

from scratch_llm.model.rope import apply_rotary_embedding

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped-query attention.

    Args:
        x: Tensor shaped (batch, seq_len, n_kv_heads, head_dim).
        n_rep: Number of repeats per key/value head.

    Returns:
        Tensor shaped (batch, seq_len, n_kv_heads * n_rep, head_dim).
    """
    if n_rep == 1:
        return x

    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    # Insert a repeat dimension after n_kv_heads -> (batch, seq_len, n_kv_heads, 1, head_dim)
    x = x[:,:,:,None,:]
    # Expand each kv head n_rep times -> (batch, seq_len, n_kv_heads, n_rep, head_dim)
    x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)

    return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

class CausalSelfAttention(nn.Module):
    """Masked self-attention with optional grouped-query attention.

    Args:
        config: ModelConfig with dim, n_heads, n_kv_heads, dropout, and
            max_seq_len.

    Expected forward input:
        x: Tensor shaped (batch, seq_len, dim).
        freqs_cos: RoPE cosine table shaped (seq_len, head_dim / 2).
        freqs_sin: RoPE sine table shaped (seq_len, head_dim / 2).
        attention_mask: Optional bool/int tensor shaped (batch, seq_len), where
            truthy values are valid tokens.

    Expected forward output:
        Tensor shaped (batch, seq_len, dim).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.effective_n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, seq_len, dim = x.shape
        # dimension check
        if dim != self.dim:
            raise ValueError(f"Expected x.shape[-1] == {self.dim}, got {dim}")

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )


        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_embedding(q, k, freqs_cos, freqs_sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        # get a minimal value to hold over exp: exp(-inf) -> 0 after softmax
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~causal_mask, mask_value)

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"attention_mask must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

            key_mask = attention_mask.to(
                device=scores.device,
                dtype=torch.bool,
            ).view(batch_size, 1, 1, seq_len)

            scores = scores.masked_fill(~key_mask, mask_value)

        attn_probs = torch.softmax(scores.float(), dim=-1).type_as(scores)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        out = self.wo(out)
        out = self.resid_dropout(out)

        return out