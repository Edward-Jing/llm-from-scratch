"""Helpers for reconstructing model config during inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from scratch_llm.config import ModelConfig


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "vocab_size": 6144,
    "dim": 512,
    "n_layers": 8,
    "n_heads": 8,
    "n_kv_heads": None,
    "max_seq_len": 512,
}


def read_checkpoint_model_config(path: str | Path) -> dict[str, Any]:
    """Read saved ModelConfig metadata from a checkpoint when available."""

    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")

    extra = checkpoint.get("extra", {})
    if not isinstance(extra, dict):
        return {}

    model_config = extra.get("model_config", {})
    if not isinstance(model_config, dict):
        return {}

    valid_fields = set(ModelConfig.__dataclass_fields__)
    return {key: value for key, value in model_config.items() if key in valid_fields}


def build_model_config_from_checkpoint(
    path: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ModelConfig:
    """Build ModelConfig from defaults, checkpoint metadata, and overrides."""

    values = DEFAULT_MODEL_CONFIG | read_checkpoint_model_config(path)
    if overrides is not None:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return ModelConfig(**values)
