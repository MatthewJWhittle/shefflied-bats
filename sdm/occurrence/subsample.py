"""Fold-stratified, seeded subsampling of training points.

Used to make model search cheaper while keeping every fold's mix of presence
and background. Draw once per study and reuse, so every trial sees the same rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def stratified_subsample(
    df: pd.DataFrame,
    strata: np.ndarray,
    *,
    presence_fraction: float = 1.0,
    background_fraction: float = 1.0,
    min_presence: int = 0,
    min_background: int = 0,
    random_state: int,
    class_col: str = "class",
) -> np.ndarray:
    """Return a boolean mask (by position) of rows to keep.

    For each class (presence = 1, background = 0) the target size is
    ``max(min, ceil(fraction * n))``, capped at ``n``. A class at or below its
    minimum is kept whole, so rare species keep every record. The target is
    spread across ``strata`` (e.g. fold labels) in proportion to stratum size
    via ``DataFrameGroupBy.sample``.
    """
    for name, frac in (("presence_fraction", presence_fraction), ("background_fraction", background_fraction)):
        if not 0 < frac <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    classes = df[class_col].to_numpy()
    n_by_class = {1: int((classes == 1).sum()), 0: int((classes == 0).sum())}
    settings = {1: (presence_fraction, min_presence), 0: (background_fraction, min_background)}
    effective = {
        c: (min(n, max(m, math.ceil(f * n))) / n if n else 1.0)
        for c, n in n_by_class.items()
        for f, m in [settings[c]]
    }
    keyed = pd.DataFrame(
        {"cls": classes, "stratum": np.asarray(strata)}
    )
    keep = np.zeros(len(df), dtype=bool)
    for cls, frac in effective.items():
        rows = keyed[keyed["cls"] == cls]
        if frac >= 1.0:
            keep[rows.index] = True
        elif len(rows):
            picked = rows.groupby("stratum").sample(frac=frac, random_state=random_state)
            keep[picked.index] = True
    return keep
