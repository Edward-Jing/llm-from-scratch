"""Learning-rate schedules."""

from __future__ import annotations

import math


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute warmup plus cosine-decay learning rate.

    Args:
        step: Current optimizer step, starting at 0.
        total_steps: Total planned optimizer steps.
        base_lr: Peak learning rate after warmup.
        warmup_steps: Number of linear warmup steps.
        min_lr_ratio: Final LR as base_lr * min_lr_ratio.

    Returns:
        Learning rate for this step.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if base_lr < 0:
        raise ValueError("base_lr must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    step = max(step, 0)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    min_lr = base_lr * min_lr_ratio

    if total_steps <= warmup_steps:
        return min_lr

    decay_step = min(step - warmup_steps, total_steps - warmup_steps)
    decay_steps = total_steps - warmup_steps
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine
