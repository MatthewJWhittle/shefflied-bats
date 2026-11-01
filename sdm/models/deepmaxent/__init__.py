"""DeepMaxent: shared multi-output residual MLP with MaxEnt / Poisson loss.

Experimental alternative to per-model elapid MaxEnt. Training uses PyTorch
Lightning. Requires the optional ``dl`` extra (``uv sync --extra dl``).
MaxEnt remains the default trainer.
"""

from __future__ import annotations

from sdm.models.deepmaxent.config import DeepMaxentParams, resolve_deepmaxent_params
from sdm.models.deepmaxent.device import resolve_torch_device
from sdm.models.deepmaxent.training import train_deepmaxent_joint

__all__ = [
    "DeepMaxentParams",
    "resolve_deepmaxent_params",
    "resolve_torch_device",
    "train_deepmaxent_joint",
]
