"""Top-level decoder-only language model contract."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from scratch_llm.config import ModelConfig
from scratch_llm.model.blocks import DecoderBlock
from scratch_llm.model.norm import RMSNorm
from scratch_llm.model.rope import precompute_rope_frequencies


class ScratchLLM(nn.Module):
    """A small LLaMA-style decoder-only language model.

    Args:
        config: ModelConfig for embedding, blocks, norm, and LM head.

    Expected forward input:
        input_ids: Token IDs shaped (batch, seq_len).
        labels: Optional target IDs shaped (batch, seq_len). Use -100 or
            pad_token_id for ignored positions.
        attention_mask: Optional bool/int tensor shaped (batch, seq_len).

    Expected forward output:
        A dictionary with:
        logits: Tensor shaped (batch, seq_len, vocab_size) during training, or
            (batch, 1, vocab_size) if you optimize inference later.
        loss: Optional scalar or per-token loss, depending on your loop design.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(layer_id=layer_id, config=config) for layer_id in range(config.n_layers)]
        )
        self.norm = RMSNorm(dim=config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_rope_frequencies(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self.init_weights)

        if config.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

    def init_weights(self, module: nn.Module) -> None:
        """Initialize Linear and Embedding weights.

        Args:
            module: Submodule passed by nn.Module.apply.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if 0 <= self.config.pad_token_id < module.num_embeddings:
                with torch.no_grad():
                    module.weight[self.config.pad_token_id].zero_()

    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Normalize attention masks to bool shape (batch, seq_len).

        Args:
            attention_mask: None, 2D, 3D, or 4D mask from callers/tokenizers.
            input_ids: Token IDs used to validate final mask shape.

        Returns:
            None or a bool tensor shaped exactly like input_ids.
        """

        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            return input_ids.ne(self.config.pad_token_id)

        mask = attention_mask.to(device=input_ids.device)

        if mask.dim() == 2:
            normalized = mask
        elif mask.dim() == 3:
            if mask.shape == (batch_size, 1, seq_len):
                normalized = mask[:, 0, :]
            elif mask.shape == (batch_size, seq_len, seq_len):
                normalized = mask.any(dim=1)
            else:
                raise ValueError(
                    "3D attention_mask must have shape "
                    f"{(batch_size, 1, seq_len)} or {(batch_size, seq_len, seq_len)}, "
                    f"got {tuple(mask.shape)}"
                )
        elif mask.dim() == 4:
            if mask.shape == (batch_size, 1, 1, seq_len):
                normalized = mask[:, 0, 0, :]
            elif mask.shape[0] == batch_size and mask.shape[2:] == (seq_len, seq_len):
                normalized = mask.any(dim=1).any(dim=1)
            else:
                raise ValueError(
                    "4D attention_mask must have shape "
                    f"{(batch_size, 1, 1, seq_len)} or "
                    f"(batch, heads, {seq_len}, {seq_len}), got {tuple(mask.shape)}"
                )
        else:
            raise ValueError(
                "attention_mask must be None, 2D, 3D, or 4D, "
                f"got {mask.dim()}D"
            )

        if normalized.shape != input_ids.shape:
            raise ValueError(
                f"normalized attention_mask must have shape {tuple(input_ids.shape)}, "
                f"got {tuple(normalized.shape)}"
            )

        return normalized.to(dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Run the Transformer and optionally compute language-model loss."""

        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len), got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(
                f"labels must have shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}"
            )

        normalized_attention_mask = self.prepare_attention_mask(attention_mask, input_ids)

        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.layers:
            h = layer(
                h,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                attention_mask=normalized_attention_mask,
            )

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if labels is not None:
            loss_labels = labels.to(device=logits.device).clone()
            loss_labels = loss_labels.masked_fill(loss_labels == self.config.pad_token_id, -100)

            flat_logits = logits.view(batch_size * seq_len, self.config.vocab_size)
            flat_labels = loss_labels.reshape(batch_size * seq_len)
            valid_labels = flat_labels.ne(-100)

            if valid_labels.any():
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            else:
                loss = logits.new_tensor(0.0)

        return {"logits": logits, "loss": loss}
