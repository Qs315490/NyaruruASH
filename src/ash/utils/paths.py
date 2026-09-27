"""Path helpers.

The project must work inside a write-restricted sandbox where the default uv
cache (~/.cache/uv) is read-only, so nothing here ever writes outside the
project unless a caller explicitly asks for an absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | os.PathLike[str], *, root: Path | None = None) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or PROJECT_ROOT) / p


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory (and parents) and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
