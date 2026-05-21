"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is optional for this project.
    np = None  # type: ignore[assignment]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch random number generators.

    Args:
        seed: Integer random seed.
        deterministic: If True, request deterministic torch algorithms where
            available.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
