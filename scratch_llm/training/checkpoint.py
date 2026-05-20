"""Checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped by DataParallel/DDP/compile.

    Args:
        model: A torch module, possibly wrapped.

    Returns:
        The module whose state_dict should be saved.
    """

    while hasattr(model, "module"):
        model = model.module

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    return model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save model and optional optimizer state.

    Args:
        path: Output checkpoint file path.
        model: Model whose state_dict will be saved.
        optimizer: Optional optimizer to resume training.
        step: Optional global optimizer step.
        extra: Optional metadata, such as config dicts.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "step": step,
        "extra": {} if extra is None else extra,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model and optional optimizer state.

    Args:
        path: Checkpoint file path.
        model: Model receiving the saved state_dict.
        optimizer: Optional optimizer to restore.
        map_location: Device mapping used by torch.load.
        strict: Whether model.load_state_dict should enforce exact keys.

    Returns:
        Remaining checkpoint metadata, such as step and extra.
    """

    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        model_state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        raise KeyError("Checkpoint does not contain a model state dict")

    load_result = unwrap_model(model).load_state_dict(model_state, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "model", "state_dict", "optimizer_state_dict"}
    }

    if not strict:
        metadata["missing_keys"] = list(load_result.missing_keys)
        metadata["unexpected_keys"] = list(load_result.unexpected_keys)

    return metadata
