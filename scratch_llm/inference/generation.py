"""Autoregressive generation contracts."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from scratch_llm.config import GenerationConfig


def top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    """Mask logits outside the top-k choices.

    Args:
        logits: Tensor shaped (batch, vocab_size).
        top_k: Number of highest logits to keep. None keeps all.

    Returns:
        Filtered logits with non-top-k entries set to -inf.
    """

    if logits.dim() != 2:
        raise ValueError(f"logits must have shape (batch, vocab_size), got {tuple(logits.shape)}")

    if top_k is None:
        return logits

    if top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    vocab_size = logits.shape[-1]
    if top_k >= vocab_size:
        return logits

    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Choose the next token from logits.

    Args:
        logits: Tensor shaped (batch, vocab_size).
        temperature: 0 for greedy decoding, otherwise softmax temperature.
        top_k: Optional top-k filter before sampling.

    Returns:
        Token IDs shaped (batch, 1).
    """

    if logits.dim() != 2:
        raise ValueError(f"logits must have shape (batch, vocab_size), got {tuple(logits.shape)}")

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered_logits = top_k_filter(logits, top_k)
    scaled_logits = filtered_logits / temperature
    probs = torch.softmax(scaled_logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate continuation tokens.

    Args:
        model: ScratchLLM or compatible causal language model.
        input_ids: Prompt token IDs shaped (batch, prompt_len).
        config: GenerationConfig.
        attention_mask: Optional mask shaped (batch, prompt_len).

    Returns:
        New token IDs shaped (batch, generated_len), excluding the prompt.
    """

    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must have shape (batch, prompt_len), got {tuple(input_ids.shape)}"
        )

    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask must have shape {tuple(input_ids.shape)}, "
            f"got {tuple(attention_mask.shape)}"
        )

    if config.max_new_tokens == 0:
        return input_ids.new_empty((input_ids.shape[0], 0))

    was_training = model.training
    model.eval()

    current_ids = input_ids
    current_mask = attention_mask
    if current_mask is None and config.pad_token_id is not None:
        current_mask = current_ids.ne(config.pad_token_id)

    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    generated: list[torch.Tensor] = []

    for _ in range(config.max_new_tokens):
        output = model(input_ids=current_ids, attention_mask=current_mask)
        logits = output["logits"][:, -1, :]
        next_token = sample_next_token(
            logits,
            temperature=config.temperature,
            top_k=config.top_k,
        )

        if config.pad_token_id is not None:
            pad_tokens = torch.full_like(next_token, config.pad_token_id)
            next_token = torch.where(finished.unsqueeze(-1), pad_tokens, next_token)

        generated.append(next_token)

        if config.eos_token_id is not None:
            just_finished = next_token.squeeze(-1).eq(config.eos_token_id)
            finished = finished | just_finished

        current_ids = torch.cat([current_ids, next_token], dim=1)

        if current_mask is not None:
            next_mask = ~finished
            current_mask = torch.cat([current_mask.bool(), next_mask.unsqueeze(-1)], dim=1)

        if config.eos_token_id is not None and finished.all():
            break

    if was_training:
        model.train()

    return torch.cat(generated, dim=1)
