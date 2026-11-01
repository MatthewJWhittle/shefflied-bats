"""Torch device selection: MPS → CUDA → CPU, with config override."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_torch_device(preference: str = "auto") -> str:
    """Return a torch device string.

    Preference ``auto`` (default) picks Apple MPS when available, else CUDA,
    else CPU. Explicit ``mps`` / ``cuda`` / ``cpu`` override that order and
    raise if the requested backend is unavailable.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DeepMaxent. Install with: uv sync --extra dl"
        ) from exc

    pref = (preference or "auto").strip().lower()
    if pref not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError(
            f"Unknown deepmaxent device preference {preference!r}; "
            "use auto, mps, cuda, or cpu"
        )

    def _mps_ok() -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    def _cuda_ok() -> bool:
        return bool(torch.cuda.is_available())

    if pref == "cpu":
        device = "cpu"
    elif pref == "mps":
        if not _mps_ok():
            raise RuntimeError("MPS requested but torch.backends.mps is unavailable")
        device = "mps"
    elif pref == "cuda":
        if not _cuda_ok():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        device = "cuda"
    else:
        if _mps_ok():
            device = "mps"
        elif _cuda_ok():
            device = "cuda"
        else:
            device = "cpu"

    logger.info("DeepMaxent using device: %s (preference=%s)", device, pref)
    return device


def seed_everything(seed: int, *, device: Optional[str] = None) -> None:
    """Seed Python, NumPy, and Torch RNGs; enable CPU determinism when on CPU."""
    import random

    import numpy as np

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DeepMaxent. Install with: uv sync --extra dl"
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" or (device is None and torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are reliable on CPU; GPU/MPS backends vary.
    use_cpu = device == "cpu" or (
        device is None and not torch.cuda.is_available() and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    )
    if use_cpu:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
