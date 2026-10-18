"""Project-wide RNG seeding for training and tuning."""

from __future__ import annotations

import random

import numpy as np


def apply_training_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs used by training and tuning code paths."""
    np.random.seed(seed)
    random.seed(seed)
