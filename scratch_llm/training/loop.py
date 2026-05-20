"""Training and evaluation loop contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
import math
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer

from scratch_llm.config import TrainConfig
from scratch_llm.training.lr import cosine_lr


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Logger = Callable[[dict[str, Any]], None]


def move_batch_to_device(batch: Batch, device: str | torch.device) -> Batch:
    """Move x, y, and loss_mask tensors to a device.

    Args:
        batch: Tuple (x, y, loss_mask).
        device: Target torch device.

    Returns:
        Tuple with all tensors moved to device.
    """

    x, y, loss_mask = batch
    return x.to(device), y.to(device), loss_mask.to(device)


def compute_masked_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-token loss with a binary mask.

    Args:
        per_token_loss: Loss tensor shaped (batch * seq_len,) or (batch, seq_len).
        loss_mask: 1/0 mask with a shape broadcastable to per_token_loss.

    Returns:
        Scalar average loss over positions where loss_mask is 1.
    """

    mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    if mask.shape != per_token_loss.shape and mask.numel() == per_token_loss.numel():
        mask = mask.reshape_as(per_token_loss)

    per_token_loss = per_token_loss * mask
    denominator = mask.sum()

    if denominator.item() == 0:
        return per_token_loss.sum() * 0.0

    return per_token_loss.sum() / denominator


def _autocast_context(config: TrainConfig) -> Any:
    """Return an autocast context for the configured device and dtype."""

    device = torch.device(config.device)

    if config.dtype == "float32":
        return nullcontext()
    if config.dtype == "float16":
        dtype = torch.float16
    elif config.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {config.dtype}")

    if device.type not in {"cuda", "cpu"}:
        return nullcontext()

    return torch.autocast(device_type=device.type, dtype=dtype)


def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    """Set the learning rate on every optimizer parameter group."""

    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    optimizer: Optimizer,
    config: TrainConfig,
    epoch: int,
    total_steps: int,
    start_step: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    logger: Optional[Logger] = None,
) -> int:
    """Train for one epoch.

    Args:
        model: ScratchLLM or compatible language model.
        dataloader: Iterable yielding (x, y, loss_mask).
        optimizer: Optimizer, usually AdamW.
        config: TrainConfig.
        epoch: Current epoch number.
        total_steps: Total planned optimizer steps for LR schedule.
        start_step: Global optimizer step before this epoch.
        scaler: Optional mixed-precision GradScaler.
        logger: Optional callback receiving metrics dictionaries.

    Returns:
        Updated global optimizer step.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    device = torch.device(config.device)
    accumulation_steps = max(config.accumulation_steps, 1)
    global_step = start_step
    pending_micro_steps = 0
    running_loss = 0.0
    running_tokens = 0.0

    for micro_step, batch in enumerate(dataloader, start=1):
        x, y, loss_mask = move_batch_to_device(batch, device)

        with _autocast_context(config):
            output = model(
                input_ids=x,
                attention_mask=x.ne(model.config.pad_token_id) if hasattr(model, "config") else None,
            )
            logits = output["logits"]
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(y)
            loss = compute_masked_loss(per_token_loss, loss_mask)

        scaled_loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        pending_micro_steps += 1
        running_loss += loss.detach().float().item() * loss_mask.sum().detach().float().item()
        running_tokens += loss_mask.sum().detach().float().item()

        is_update_step = pending_micro_steps == accumulation_steps

        if is_update_step:
            lr = cosine_lr(
                step=global_step,
                total_steps=total_steps,
                base_lr=config.learning_rate,
                warmup_steps=config.warmup_steps,
            )
            _set_optimizer_lr(optimizer, lr)

            if scaler is not None:
                scaler.unscale_(optimizer)

            if config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pending_micro_steps = 0

            if logger is not None and (
                global_step == 1 or global_step % max(config.log_interval, 1) == 0
            ):
                mean_loss = running_loss / max(running_tokens, 1.0)
                logger(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "micro_step": micro_step,
                        "loss": mean_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                )
                running_loss = 0.0
                running_tokens = 0.0

    if pending_micro_steps > 0:
        lr = cosine_lr(
            step=global_step,
            total_steps=total_steps,
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
        )
        _set_optimizer_lr(optimizer, lr)

        if scaler is not None:
            scaler.unscale_(optimizer)

        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=device)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if logger is not None:
            mean_loss = running_loss / max(running_tokens, 1.0)
            logger(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "micro_step": micro_step if "micro_step" in locals() else 0,
                    "loss": mean_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )

    return global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[Batch],
    config: TrainConfig,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate masked language-model loss.

    Args:
        model: ScratchLLM or compatible language model.
        dataloader: Iterable yielding (x, y, loss_mask).
        config: TrainConfig.
        max_batches: Optional cap for quick validation.

    Returns:
        Metrics dictionary, for example {"loss": 1.23, "ppl": 3.42}.
    """

    model.eval()

    device = torch.device(config.device)
    total_loss = 0.0
    total_tokens = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y, loss_mask = move_batch_to_device(batch, device)

        with _autocast_context(config):
            output = model(
                input_ids=x,
                attention_mask=x.ne(model.config.pad_token_id) if hasattr(model, "config") else None,
            )
            logits = output["logits"]
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(y)

        mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
        total_loss += (per_token_loss * mask).sum().detach().float().item()
        total_tokens += mask.sum().detach().float().item()

    if total_tokens == 0:
        loss = 0.0
    else:
        loss = total_loss / total_tokens

    ppl = math.exp(loss) if loss < 100 else float("inf")
    return {"loss": loss, "ppl": ppl}
