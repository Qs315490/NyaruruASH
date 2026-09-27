"""Determinism helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every RNG we own; torch is seeded only if it is importable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass


def torch_generator(seed: int):
    """A seeded torch.Generator, or None when torch is unavailable."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen
