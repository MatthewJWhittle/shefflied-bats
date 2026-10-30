"""Partial-dependence / response curves for fitted DeepMaxent packages.

## Implementation notes

Uses ``sklearn.inspection.partial_dependence`` (average PD) and
``PartialDependenceDisplay`` for PNGs. Curves are written under each model
package as run artifacts only — never committed (licensed / site data).

Oversampled or shared background rows are fine as the PD design matrix; the
curves are a post-fit diagnostic, not part of training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from sdm.commands.modelling.utils import get_model_id
from sdm.types import TrainingResults

logger = logging.getLogger(__name__)

PD_SUBDIR = "partial_dependence"
DEFAULT_GRID_RESOLUTION = 51


def _background_frame(
    background: Union[pd.DataFrame, np.ndarray],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    names = list(feature_names)
    if isinstance(background, pd.DataFrame):
        missing = [c for c in names if c not in background.columns]
        if missing:
            raise ValueError(f"Background missing features for PD: {missing}")
        return background[names].copy()
    arr = np.asarray(background, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(
            f"Background array shape {arr.shape} incompatible with "
            f"{len(names)} features"
        )
    return pd.DataFrame(arr, columns=names)


def partial_dependence_for_feature(
    estimator,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    percentiles: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Average PD for one feature via sklearn; columns ``{feature}``, ``suitability``."""
    if feature not in X.columns:
        raise ValueError(f"Feature {feature!r} not in background columns")
    result = partial_dependence(
        estimator,
        X,
        features=[feature],
        kind="average",
        grid_resolution=int(grid_resolution),
        percentiles=percentiles,
    )
    grid = np.asarray(result["grid_values"][0], dtype=float)
    # Classifiers: average is [n_targets, n_grid]; binary → positive class row.
    average = np.asarray(result["average"], dtype=float)
    if average.ndim == 2:
        values = average[-1]
    else:
        values = average.reshape(-1)
    return pd.DataFrame({feature: grid, "suitability": values})


def write_feature_pd_plot(
    estimator,
    X: pd.DataFrame,
    feature: str,
    output_path: Path,
    *,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    percentiles: tuple[float, float] = (0.05, 0.95),
) -> Path:
    """Write a single-feature average PD PNG (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X,
        features=[feature],
        kind="average",
        grid_resolution=int(grid_resolution),
        percentiles=percentiles,
    )
    disp.figure_.suptitle(f"Partial dependence: {feature}")
    disp.figure_.tight_layout()
    disp.figure_.savefig(output_path, dpi=120)
    plt.close(disp.figure_)
    return output_path


def write_model_partial_dependence(
    estimator,
    background: Union[pd.DataFrame, np.ndarray],
    feature_names: Sequence[str],
    output_dir: Path,
    *,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
) -> List[Path]:
    """Write ``pd__{feature}.csv`` and ``.png`` for every feature under ``output_dir``."""
    X = _background_frame(background, feature_names)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for feat in feature_names:
        table = partial_dependence_for_feature(
            estimator, X, feat, grid_resolution=grid_resolution
        )
        csv_path = out / f"pd__{feat}.csv"
        table.to_csv(csv_path, index=False)
        written.append(csv_path)
        png_path = out / f"pd__{feat}.png"
        write_feature_pd_plot(
            estimator, X, feat, png_path, grid_resolution=grid_resolution
        )
        written.append(png_path)
    return written


def _feature_names_from_result(
    tr: TrainingResults,
    fallback: Optional[Sequence[str]],
) -> List[str]:
    if fallback is not None:
        return list(fallback)
    model = tr.final_model
    if model is None:
        return []
    if hasattr(model, "named_steps"):
        head = model.named_steps.get("deepmaxent")
        if head is not None and hasattr(head, "feature_names"):
            return list(head.feature_names)
        if head is not None and hasattr(getattr(head, "ensemble", None), "feature_names"):
            return list(head.ensemble.feature_names)
    if hasattr(model, "feature_names"):
        return list(model.feature_names)
    return []


def write_run_partial_dependence(
    run_dir: Path,
    results: Sequence[TrainingResults],
    background: Union[pd.DataFrame, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
) -> Dict[str, List[Path]]:
    """Write PD CSV+PNG under each model package's ``partial_dependence/`` folder.

    Layout::

        {run_dir}/{model_id}/partial_dependence/pd__{feature}.csv
        {run_dir}/{model_id}/partial_dependence/pd__{feature}.png
    """
    run_dir = Path(run_dir)
    by_model: Dict[str, List[Path]] = {}
    for tr in results:
        if not tr.success or tr.final_model is None:
            continue
        mid = get_model_id([tr.latin_name, tr.activity_type])
        feats = _feature_names_from_result(tr, feature_names)
        if not feats:
            logger.warning("Skipping PD for %s (no feature names)", mid)
            continue
        pkg_pd = run_dir / mid / PD_SUBDIR
        paths = write_model_partial_dependence(
            tr.final_model,
            background,
            feats,
            pkg_pd,
            grid_resolution=grid_resolution,
        )
        by_model[mid] = paths
        logger.info(
            "Wrote DeepMaxent PD curves for %s (%d features) under %s",
            mid,
            len(feats),
            pkg_pd,
        )
    return by_model


def pd_artifacts_relative(feature_names: Sequence[str]) -> Dict[str, Any]:
    """Relative artifact paths to record in package / run metadata."""
    return {
        "partial_dependence_dir": PD_SUBDIR,
        "partial_dependence_csv": [f"{PD_SUBDIR}/pd__{f}.csv" for f in feature_names],
        "partial_dependence_png": [f"{PD_SUBDIR}/pd__{f}.png" for f in feature_names],
    }
