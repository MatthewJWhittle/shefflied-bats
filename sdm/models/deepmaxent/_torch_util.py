"""Lazy PyTorch import helpers so base installs can import the package."""

from __future__ import annotations

from typing import Any, Tuple


def require_torch() -> Tuple[Any, Any]:
    """Return ``(torch, torch.nn)`` or raise a clear optional-extra error."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DeepMaxent. Install with: uv sync --extra dl"
        ) from exc
    return torch, nn
