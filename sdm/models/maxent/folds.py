"""Selectable geographic cross-validation fold methods for MaxEnt training.

Methods
-------
kmeans
    Current baseline: elapid ``GeographicKFold`` (KMeans on point coordinates).
blocks
    Square spatial blocks assigned to ``k`` folds; assignment chosen to balance
    presence counts while keeping presence and background in every fold.
buffered_loo
    Hold out each presence in turn. Points within ``buffer_m`` of that presence
    are dropped from training. The held-out presence is scored against background
    points that fall inside the same buffer (also held out). Other presences
    inside the buffer are excluded from both train and test for that fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator, Optional

import elapid as ela
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sdm.types import CvConfig
from sdm.utils.rng import apply_training_seed


class FoldMethod(StrEnum):
    KMEANS = "kmeans"
    BLOCKS = "blocks"
    BUFFERED_LOO = "buffered_loo"


DEFAULT_BUFFERED_LOO_WARN_ABOVE = 25


@dataclass(frozen=True)
class FoldParams:
    """Resolved fold settings used for one training run."""

    method: FoldMethod
    n_folds: int
    random_state: int
    block_size_m: float = 10_000.0
    block_n_tries: int = 500
    buffer_m: float = 10_000.0
    buffered_loo_warn_above: int = DEFAULT_BUFFERED_LOO_WARN_ABOVE

    def to_metadata(self) -> dict:
        payload: dict = {
            "method": self.method.value,
            "n_folds": self.n_folds,
            "seed": self.random_state,
        }
        if self.method is FoldMethod.BLOCKS:
            payload["block_size_m"] = self.block_size_m
            payload["block_n_tries"] = self.block_n_tries
        if self.method is FoldMethod.BUFFERED_LOO:
            payload["buffer_m"] = self.buffer_m
            payload["buffered_loo_warn_above"] = self.buffered_loo_warn_above
            # n_folds for LOO is presence count; recorded after split when known.
        return payload


def buffered_loo_presence_warning(
    n_presence: int,
    warn_above: int,
    *,
    latin_name: str,
    activity_type: str,
) -> Optional[str]:
    """Return a warning message when buffered LOO is used on a non-thin species.

    Warning-only guidance: fires when ``n_presence > warn_above``. Does not
    change fold construction or training behaviour.
    """
    if n_presence <= warn_above:
        return None
    return (
        f"{latin_name} - {activity_type}: buffered_loo used with {n_presence} "
        f"presence records (threshold {warn_above}). Buffered leave-one-out is "
        f"intended for thin species; blocks or kmeans is the recommended method "
        f"at this sample size."
    )


def resolve_fold_params(
    cv_config: Optional[CvConfig],
    *,
    model_seed: int,
    method_override: Optional[str] = None,
    n_folds_override: Optional[int] = None,
    block_size_m_override: Optional[float] = None,
    block_n_folds_override: Optional[int] = None,
    buffer_m_override: Optional[float] = None,
    buffered_loo_warn_above_override: Optional[int] = None,
) -> FoldParams:
    """Merge config defaults with optional CLI overrides."""
    cfg = cv_config or CvConfig()
    method = FoldMethod((method_override or cfg.method).lower())
    fold_seed = cfg.seed if cfg.seed is not None else model_seed

    if method is FoldMethod.KMEANS:
        n_folds = n_folds_override if n_folds_override is not None else cfg.n_folds
        return FoldParams(
            method=method,
            n_folds=n_folds,
            random_state=fold_seed,
        )

    if method is FoldMethod.BLOCKS:
        n_folds = (
            block_n_folds_override
            if block_n_folds_override is not None
            else (n_folds_override if n_folds_override is not None else cfg.block_n_folds)
        )
        block_size = (
            block_size_m_override
            if block_size_m_override is not None
            else cfg.block_size_m
        )
        return FoldParams(
            method=method,
            n_folds=n_folds,
            random_state=fold_seed,
            block_size_m=float(block_size),
            block_n_tries=cfg.block_n_tries,
        )

    if method is FoldMethod.BUFFERED_LOO:
        buffer_m = (
            buffer_m_override if buffer_m_override is not None else cfg.buffer_m
        )
        warn_above = (
            buffered_loo_warn_above_override
            if buffered_loo_warn_above_override is not None
            else cfg.buffered_loo_warn_above
        )
        return FoldParams(
            method=method,
            n_folds=0,  # resolved to presence count when splitting
            random_state=fold_seed,
            buffer_m=float(buffer_m),
            buffered_loo_warn_above=int(warn_above),
        )

    raise ValueError(f"unknown fold method: {method!r}")


def _block_keys(gdf: gpd.GeoDataFrame, block_size_m: float) -> np.ndarray:
    xs = gdf.geometry.x.to_numpy(dtype=float)
    ys = gdf.geometry.y.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.floor(xs / block_size_m).astype(np.int64),
            np.floor(ys / block_size_m).astype(np.int64),
        ]
    )


def assign_spatial_block_folds(
    gdf: gpd.GeoDataFrame,
    *,
    n_folds: int,
    block_size_m: float,
    n_tries: int,
    random_state: int,
    class_col: str = "class",
) -> np.ndarray:
    """Assign each point to a fold via square blocks with balanced presences.

    Background points inherit the fold of their block. Among seeded random
    assignments of blocks to folds, choose the one that (1) puts presence and
    background in every fold and (2) minimises the range of presence counts.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for spatial blocks")
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")

    classes = gdf[class_col].to_numpy()
    keys = _block_keys(gdf, block_size_m)
    # Map each unique block to row indices.
    block_index: dict[tuple[int, int], int] = {}
    point_block = np.empty(len(gdf), dtype=np.int64)
    for i, key in enumerate(map(tuple, keys)):
        if key not in block_index:
            block_index[key] = len(block_index)
        point_block[i] = block_index[key]

    n_blocks = len(block_index)
    if n_blocks < n_folds:
        raise ValueError(
            f"Only {n_blocks} spatial blocks of {block_size_m} m for {n_folds} folds; "
            "use a smaller block size or fewer folds"
        )

    presence_per_block = np.zeros(n_blocks, dtype=np.int64)
    background_per_block = np.zeros(n_blocks, dtype=np.int64)
    for block_id in range(n_blocks):
        mask = point_block == block_id
        presence_per_block[block_id] = int(np.sum(classes[mask] == 1))
        background_per_block[block_id] = int(np.sum(classes[mask] == 0))

    rng = np.random.default_rng(random_state)
    best_labels: Optional[np.ndarray] = None
    best_score: tuple[int, float] | None = None
    block_ids = np.arange(n_blocks)

    for _ in range(n_tries):
        # Greedy number-partition: assign blocks in shuffled order to the fold
        # that currently has the fewest presence records (seeded tie-breaks).
        order = block_ids.copy()
        rng.shuffle(order)
        block_to_fold = np.empty(n_blocks, dtype=np.int64)
        presence_counts = np.zeros(n_folds, dtype=np.int64)
        background_counts = np.zeros(n_folds, dtype=np.int64)

        for block_id in order:
            # Prefer folds that still lack presence or background when possible.
            candidates = np.arange(n_folds)
            lacking_presence = candidates[presence_counts == 0]
            if lacking_presence.size and presence_per_block[block_id] > 0:
                candidates = lacking_presence
            else:
                lacking_bg = candidates[background_counts == 0]
                if lacking_bg.size and background_per_block[block_id] > 0:
                    candidates = lacking_bg

            fold_loads = presence_counts[candidates]
            min_load = fold_loads.min()
            tied = candidates[fold_loads == min_load]
            fold = int(rng.choice(tied))
            block_to_fold[block_id] = fold
            presence_counts[fold] += presence_per_block[block_id]
            background_counts[fold] += background_per_block[block_id]

        if np.any(presence_counts == 0) or np.any(background_counts == 0):
            continue

        spread = int(presence_counts.max() - presence_counts.min())
        variance = float(np.var(presence_counts.astype(float)))
        score = (spread, variance)
        if best_score is None or score < best_score:
            best_score = score
            best_labels = block_to_fold[point_block].astype(np.int64)
            if spread == 0:
                break

    if best_labels is None:
        raise ValueError(
            f"Could not find a {n_folds}-fold block assignment with presence and "
            f"background in every fold after {n_tries} tries "
            f"(block_size_m={block_size_m})"
        )
    return best_labels


def _iter_kmeans_splits(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    # Preserve current behaviour: seed global RNGs then use elapid GeographicKFold.
    apply_training_seed(params.random_state)
    gfolds = ela.GeographicKFold(n_splits=params.n_folds)
    for fold_id, (train_idx, test_idx) in enumerate(gfolds.split(gdf)):
        yield fold_id, np.asarray(train_idx), np.asarray(test_idx)


def _iter_block_splits(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    labels = assign_spatial_block_folds(
        gdf,
        n_folds=params.n_folds,
        block_size_m=params.block_size_m,
        n_tries=params.block_n_tries,
        random_state=params.random_state,
    )
    indices = np.arange(len(gdf))
    for fold_id in range(params.n_folds):
        test_idx = indices[labels == fold_id]
        train_idx = indices[labels != fold_id]
        yield fold_id, train_idx, test_idx


def _iter_buffered_loo_splits(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
    *,
    class_col: str = "class",
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Leave-one-presence-out with a spatial buffer (see module docstring)."""
    classes = gdf[class_col].to_numpy()
    presence_idx = np.flatnonzero(classes == 1)
    if presence_idx.size == 0:
        return

    xy = np.column_stack(
        [gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)]
    )
    all_idx = np.arange(len(gdf))

    for fold_id, p_idx in enumerate(presence_idx):
        other = all_idx != p_idx
        other_idx = all_idx[other]
        if other_idx.size == 0:
            yield fold_id, np.array([], dtype=int), np.array([p_idx], dtype=int)
            continue

        deltas = xy[other_idx] - xy[p_idx]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        in_buffer = distances <= params.buffer_m
        buffer_iloc = other_idx[in_buffer]

        buffer_background = buffer_iloc[classes[buffer_iloc] == 0]
        test_idx = np.unique(np.concatenate([[p_idx], buffer_background])).astype(int)

        excluded = np.zeros(len(gdf), dtype=bool)
        excluded[p_idx] = True
        excluded[buffer_iloc] = True
        train_idx = all_idx[~excluded]
        yield fold_id, train_idx, test_idx


def iter_fold_splits(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
    *,
    class_col: str = "class",
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold_id, train_idx, test_idx)`` for the configured method."""
    if params.method is FoldMethod.KMEANS:
        yield from _iter_kmeans_splits(gdf, params)
        return
    if params.method is FoldMethod.BLOCKS:
        yield from _iter_block_splits(gdf, params)
        return
    if params.method is FoldMethod.BUFFERED_LOO:
        yield from _iter_buffered_loo_splits(gdf, params, class_col=class_col)
        return
    raise ValueError(f"unknown fold method: {params.method!r}")


def fold_assignments_frame(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
    *,
    class_col: str = "class",
) -> pd.DataFrame:
    """Per-point fold labels for methods that assign each point to one fold.

    For ``buffered_loo``, each presence is its own fold id; background points
    that fall inside any presence buffer are labelled with the nearest such
    presence's fold (points never held out get fold ``-1``).
    """
    n = len(gdf)
    classes = gdf[class_col].to_numpy()
    labels = np.full(n, -1, dtype=np.int64)

    if params.method is FoldMethod.BLOCKS:
        labels = assign_spatial_block_folds(
            gdf,
            n_folds=params.n_folds,
            block_size_m=params.block_size_m,
            n_tries=params.block_n_tries,
            random_state=params.random_state,
            class_col=class_col,
        )
    elif params.method is FoldMethod.KMEANS:
        for fold_id, _train, test in _iter_kmeans_splits(gdf, params):
            labels[test] = fold_id
    elif params.method is FoldMethod.BUFFERED_LOO:
        presence_idx = np.flatnonzero(classes == 1)
        for fold_id, p_idx in enumerate(presence_idx):
            labels[p_idx] = fold_id
        # Label buffer background with the fold of the nearest presence.
        if presence_idx.size and np.any(classes == 0):
            presence_gdf = gdf.iloc[presence_idx]
            background_mask = classes == 0
            background_gdf = gdf.iloc[background_mask]
            p_xy = np.column_stack(
                [presence_gdf.geometry.x.to_numpy(), presence_gdf.geometry.y.to_numpy()]
            )
            b_xy = np.column_stack(
                [
                    background_gdf.geometry.x.to_numpy(),
                    background_gdf.geometry.y.to_numpy(),
                ]
            )
            tree = cKDTree(p_xy)
            dist, nn = tree.query(b_xy, k=1)
            bg_positions = np.flatnonzero(background_mask)
            in_buffer = dist <= params.buffer_m
            labels[bg_positions[in_buffer]] = nn[in_buffer]
    else:
        raise ValueError(f"unknown fold method: {params.method!r}")

    return pd.DataFrame(
        {
            "point_index": np.arange(n, dtype=int),
            "class": classes.astype(int),
            "fold": labels.astype(int),
        }
    )


def presence_counts_per_fold(
    gdf: gpd.GeoDataFrame,
    params: FoldParams,
    *,
    class_col: str = "class",
) -> list[int]:
    """Count presence records held out in each fold (from split test sets)."""
    counts: list[int] = []
    classes = gdf[class_col].to_numpy()
    for _fold_id, _train, test_idx in iter_fold_splits(gdf, params, class_col=class_col):
        counts.append(int(np.sum(classes[test_idx] == 1)))
    return counts
