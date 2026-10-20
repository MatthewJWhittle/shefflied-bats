"""Pluggable suitability thresholds from CV validation scores.

Rules fit cut-offs on training folds and are scored on held-out folds so the
evaluation fold never informs the cut-off. When no ``fold`` column is present,
rules fit and evaluate on the same scores (pooled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RuleFn = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], float]


def presence_percentile_cutoff(presence_scores: np.ndarray, percentile: float) -> float:
    """Cut-off at a presence-score percentile (0 = lowest presence)."""
    scores = np.asarray(presence_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot compute threshold: no presence scores")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    return float(np.percentile(scores, percentile))


def presence_capture_cutoff(presence_scores: np.ndarray, capture: float) -> float:
    """Lowest cut-off that retains ``capture`` share of presence scores."""
    if not 0.0 < capture <= 1.0:
        raise ValueError(f"capture must be in (0, 1], got {capture}")
    omit_percentile = 100.0 * (1.0 - capture)
    return presence_percentile_cutoff(presence_scores, omit_percentile)


def max_sens_spec_cutoff(
    presence_scores: np.ndarray,
    background_scores: np.ndarray,
) -> float:
    """Cut-off that maximises sensitivity + specificity via ``roc_curve``."""
    presence = np.asarray(presence_scores, dtype=float)
    background = np.asarray(background_scores, dtype=float)
    if presence.size == 0:
        raise ValueError("Cannot compute threshold: no presence scores")
    if background.size == 0:
        raise ValueError("Cannot compute threshold: no background scores")

    y_true = np.concatenate(
        [np.ones(presence.size, dtype=int), np.zeros(background.size, dtype=int)]
    )
    y_score = np.concatenate([presence, background])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    sens_plus_spec = tpr + (1.0 - fpr)
    best = int(np.argmax(sens_plus_spec))
    cutoff = float(thresholds[best])
    if not np.isfinite(cutoff):
        # sklearn may return +inf for the first ROC point; use a cut just above max.
        cutoff = float(np.nextafter(y_score.max(), np.inf))
    return cutoff


def _rule_presence_percentile(
    presence: np.ndarray,
    background: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    del background
    return presence_percentile_cutoff(presence, float(params.get("percentile", 10.0)))


def _rule_presence_capture(
    presence: np.ndarray,
    background: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    del background
    return presence_capture_cutoff(presence, float(params.get("capture", 0.95)))


def _rule_max_sens_spec(
    presence: np.ndarray,
    background: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    del params
    return max_sens_spec_cutoff(presence, background)


THRESHOLD_RULES: Dict[str, RuleFn] = {
    "presence_percentile": _rule_presence_percentile,
    "max_sens_spec": _rule_max_sens_spec,
    "presence_capture": _rule_presence_capture,
}


def fit_threshold(
    rule: str,
    presence_scores: np.ndarray,
    background_scores: np.ndarray,
    params: Optional[Mapping[str, Any]] = None,
) -> float:
    """Dispatch ``rule`` from the registry and return a cut-off."""
    try:
        fn = THRESHOLD_RULES[rule]
    except KeyError as exc:
        known = ", ".join(sorted(THRESHOLD_RULES))
        raise KeyError(f"unknown threshold rule {rule!r}; known: {known}") from exc
    return fn(
        np.asarray(presence_scores, dtype=float),
        np.asarray(background_scores, dtype=float),
        params or {},
    )


def evaluate_threshold(
    cutoff: float,
    presence_scores: np.ndarray,
    background_scores: np.ndarray,
) -> Dict[str, float]:
    """Metrics for a cut-off on presence / background scores."""
    presence = np.asarray(presence_scores, dtype=float)
    background = np.asarray(background_scores, dtype=float)
    presence_capture = (
        float(np.mean(presence >= cutoff)) if presence.size else float("nan")
    )
    background_suitable = (
        float(np.mean(background >= cutoff)) if background.size else float("nan")
    )
    auc = float("nan")
    if presence.size and background.size:
        y_true = np.concatenate(
            [np.ones(presence.size, dtype=int), np.zeros(background.size, dtype=int)]
        )
        y_score = np.concatenate([presence, background])
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_score))
    return {
        "cutoff": float(cutoff),
        "presence_capture": presence_capture,
        "background_suitable": background_suitable,
        "auc": auc,
    }


def _split_classes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    presence = frame.loc[frame["class"] == 1, "held_out_score"].to_numpy(dtype=float)
    background = frame.loc[frame["class"] == 0, "held_out_score"].to_numpy(dtype=float)
    return presence, background


def _params_for_rule(
    rule: str,
    rule_params: Optional[Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if not rule_params:
        return {}
    return rule_params.get(rule, {})


def summarise_thresholds_for_scores(
    validation_scores: pd.DataFrame,
    *,
    rules: Optional[Iterable[str]] = None,
    rule_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Fit each rule fold-wise (when folds exist) and return a tidy summary.

    For each fold ``k``, the cut-off is fit on scores with ``fold != k`` and
    evaluated on ``fold == k``. Summary rows average cut-offs and metrics across
    folds. Without a ``fold`` column, fit and evaluate are pooled on all scores.
    """
    required = {"class", "held_out_score"}
    missing = required - set(validation_scores.columns)
    if missing:
        raise ValueError(f"validation scores missing columns: {sorted(missing)}")

    selected = list(rules) if rules is not None else list(THRESHOLD_RULES)
    unknown = [name for name in selected if name not in THRESHOLD_RULES]
    if unknown:
        raise KeyError(f"unknown threshold rule(s): {unknown}")

    has_folds = "fold" in validation_scores.columns
    fold_ids = (
        sorted(int(f) for f in validation_scores["fold"].dropna().unique())
        if has_folds
        else []
    )

    rows: list[dict[str, Any]] = []
    for rule in selected:
        params = _params_for_rule(rule, rule_params)
        fold_metrics: list[Dict[str, float]] = []

        if fold_ids:
            for fold in fold_ids:
                train = validation_scores[validation_scores["fold"] != fold]
                test = validation_scores[validation_scores["fold"] == fold]
                train_p, train_b = _split_classes(train)
                test_p, test_b = _split_classes(test)
                if train_p.size == 0 or test_p.size == 0:
                    continue
                if rule == "max_sens_spec" and train_b.size == 0:
                    continue
                cutoff = fit_threshold(rule, train_p, train_b, params=params)
                fold_metrics.append(evaluate_threshold(cutoff, test_p, test_b))
            n_folds_evaluated = len(fold_metrics)
        else:
            presence, background = _split_classes(validation_scores)
            cutoff = fit_threshold(rule, presence, background, params=params)
            fold_metrics.append(evaluate_threshold(cutoff, presence, background))
            n_folds_evaluated = 0

        if not fold_metrics:
            raise ValueError(f"No evaluable folds for rule {rule!r}")

        metrics_frame = pd.DataFrame(fold_metrics)
        row: dict[str, Any] = {
            "rule": rule,
            "cutoff": float(metrics_frame["cutoff"].mean()),
            "presence_capture": float(metrics_frame["presence_capture"].mean()),
            "background_suitable": float(metrics_frame["background_suitable"].mean()),
            "auc": float(metrics_frame["auc"].mean()),
            "n_folds_evaluated": n_folds_evaluated,
            "n_presence": int((validation_scores["class"] == 1).sum()),
            "n_background": int((validation_scores["class"] == 0).sum()),
        }
        # Keep rule parameters in the table for reproducibility.
        for key, value in params.items():
            row[f"param_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def plot_score_spreads(
    validation_scores: pd.DataFrame,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    """Optional presence vs background score histogram."""
    presence, background = _split_classes(validation_scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(background, bins=30, alpha=0.6, label="background", density=True)
    ax.hist(presence, bins=30, alpha=0.6, label="presence", density=True)
    ax.set_xlabel("held-out suitability score")
    ax.set_ylabel("density")
    ax.set_title(title or "Presence vs background scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
