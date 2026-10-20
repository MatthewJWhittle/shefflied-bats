"""Pluggable suitability thresholds from CV validation scores.

Rules fit cut-offs on training folds and are scored on held-out folds so the
evaluation fold never informs the cut-off. When no ``fold`` column is present,
rules fit and evaluate on the same scores (pooled).

The published training-run rule ``p5_median_transfer`` follows
``docs/thresholds/METHOD.md``: 5th-percentile held-out presence cut-offs per
fold, median across folds, transferred onto the final model's training-presence
scores at the same held-out capture rate, with a fold-median fallback when the
transferred cut-off is 0.95 or higher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RuleFn = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], float]

P5_MEDIAN_TRANSFER_RULE = "p5_median_transfer"
TRANSFER_FALLBACK_MIN = 0.95
AUC_MODEL_LIMITED = 0.75
FEW_RECORDS_MAX = 50
RUN_THRESHOLDS_JSON = "thresholds.json"
RUN_THRESHOLDS_CSV = "thresholds.csv"
VALIDATION_SCORES_FILENAME = "validation_scores.parquet"


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


def _oof_auc(validation_scores: pd.DataFrame) -> float:
    presence, background = _split_classes(validation_scores)
    if presence.size == 0 or background.size == 0:
        return float("nan")
    y_true = np.concatenate(
        [np.ones(presence.size, dtype=int), np.zeros(background.size, dtype=int)]
    )
    y_score = np.concatenate([presence, background])
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _threshold_flags(*, n_presence: int, oof_auc: float) -> str:
    parts: list[str] = []
    if np.isfinite(oof_auc) and oof_auc < AUC_MODEL_LIMITED:
        parts.append(f"model-limited (AUC<{AUC_MODEL_LIMITED:g})")
    if n_presence < FEW_RECORDS_MAX:
        parts.append(f"few records (<{FEW_RECORDS_MAX})")
    return "; ".join(parts)


def compute_p5_median_transfer(
    validation_scores: pd.DataFrame,
    *,
    final_presence_scores: np.ndarray,
    map_scores: Optional[np.ndarray] = None,
    transfer_fallback_min: float = TRANSFER_FALLBACK_MIN,
) -> Dict[str, Any]:
    """Publishable cut-off: fold 5th-percentile median, transferred to the final model.

    See ``docs/thresholds/METHOD.md``. When the transferred cut-off is
    ``transfer_fallback_min`` or higher (overfit final model), the fold median
    is used instead.
    """
    required = {"class", "held_out_score", "fold"}
    missing = required - set(validation_scores.columns)
    if missing:
        raise ValueError(f"validation scores missing columns: {sorted(missing)}")

    presence_all, _background_all = _split_classes(validation_scores)
    if presence_all.size == 0:
        raise ValueError("Cannot compute threshold: no presence scores")

    final_presence = np.asarray(final_presence_scores, dtype=float)
    if final_presence.size == 0:
        raise ValueError("Cannot compute threshold: no final presence scores")

    fold_ids = sorted(int(f) for f in validation_scores["fold"].dropna().unique())
    if not fold_ids:
        raise ValueError("Cannot compute threshold: no folds in validation scores")

    fold_cutoffs: list[float] = []
    fold_caught: list[float] = []
    for fold in fold_ids:
        fold_frame = validation_scores[validation_scores["fold"] == fold]
        fold_presence, _ = _split_classes(fold_frame)
        if fold_presence.size == 0:
            continue
        cutoff = presence_percentile_cutoff(fold_presence, 5.0)
        fold_cutoffs.append(cutoff)
        fold_caught.append(float(np.mean(fold_presence >= cutoff)))

    if not fold_cutoffs:
        raise ValueError("Cannot compute threshold: no folds with presence scores")

    fold_median_cutoff = float(np.median(fold_cutoffs))
    heldout_caught = float(np.mean(presence_all >= fold_median_cutoff))
    worst_fold_caught = float(min(fold_caught)) if fold_caught else float("nan")

    # Same capture rate on the final model: cut at the matching presence percentile.
    omit_percentile = 100.0 * (1.0 - heldout_caught)
    transferred = presence_percentile_cutoff(final_presence, omit_percentile)
    if transferred >= transfer_fallback_min:
        threshold = fold_median_cutoff
        fallback_used = True
        method = "fold_median (transfer broken by overfit model)"
    else:
        threshold = float(transferred)
        fallback_used = False
        method = "transfer"

    map_share: Optional[float]
    if map_scores is None:
        map_share = None
    else:
        maps = np.asarray(map_scores, dtype=float)
        map_share = float(np.mean(maps >= threshold)) if maps.size else None

    n_presence = int((validation_scores["class"] == 1).sum())
    oof_auc = _oof_auc(validation_scores)
    flag = _threshold_flags(n_presence=n_presence, oof_auc=oof_auc)

    return {
        "rule": P5_MEDIAN_TRANSFER_RULE,
        "threshold": float(threshold),
        "heldout_caught": heldout_caught,
        "worst_fold_caught": worst_fold_caught,
        "map_share": map_share,
        "fold_median_cutoff": fold_median_cutoff,
        "n_presence": n_presence,
        "oof_auc": oof_auc,
        "flag": flag,
        "fallback_used": fallback_used,
        "method": method,
    }


def write_run_thresholds(
    run_dir: Path,
    *,
    final_presence_by_model: Mapping[str, np.ndarray],
    map_scores_by_model: Optional[Mapping[str, np.ndarray]] = None,
    model_ids: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Write ``thresholds.json`` / ``.csv`` into ``run_dir`` and each package dir.

    Expects ``{run_dir}/{model_id}/validation_scores.parquet``. Does not modify
    ``package.json``. Cut-offs are model outputs only (no locations or site names).
    """
    run_dir = Path(run_dir)
    package_dirs = sorted(
        path.parent for path in run_dir.glob(f"*/{VALIDATION_SCORES_FILENAME}")
    )
    if model_ids is not None:
        wanted = {mid.lower() for mid in model_ids}
        package_dirs = [d for d in package_dirs if d.name.lower() in wanted]

    rows: list[dict[str, Any]] = []
    thresholds_map: dict[str, float] = {}
    for package_dir in package_dirs:
        model_id = package_dir.name
        if model_id not in final_presence_by_model:
            raise KeyError(
                f"missing final presence scores for model {model_id!r}; "
                f"have: {sorted(final_presence_by_model)}"
            )
        validation_scores = pd.read_parquet(package_dir / VALIDATION_SCORES_FILENAME)
        map_scores = None
        if map_scores_by_model is not None:
            map_scores = map_scores_by_model.get(model_id)
        result = compute_p5_median_transfer(
            validation_scores,
            final_presence_scores=final_presence_by_model[model_id],
            map_scores=map_scores,
        )
        row = {"model_id": model_id, **result}
        rows.append(row)
        thresholds_map[model_id] = float(result["threshold"])

        package_payload = _json_safe(
            {
                "rule": P5_MEDIAN_TRANSFER_RULE,
                "model_id": model_id,
                "threshold": float(result["threshold"]),
                "heldout_caught": result["heldout_caught"],
                "map_share": result["map_share"],
                "flag": result["flag"],
                "fallback_used": result["fallback_used"],
                "method": result["method"],
                "fold_median_cutoff": result["fold_median_cutoff"],
                "n_presence": result["n_presence"],
                "oof_auc": result["oof_auc"],
            }
        )
        with open(package_dir / RUN_THRESHOLDS_JSON, "w", encoding="utf-8") as handle:
            json.dump(package_payload, handle, indent=2, allow_nan=False)

    frame = pd.DataFrame(rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_csv(run_dir / RUN_THRESHOLDS_CSV, index=False)

    run_payload = _json_safe(
        {
            "rule": P5_MEDIAN_TRANSFER_RULE,
            "method_doc": "docs/thresholds/METHOD.md",
            "thresholds": thresholds_map,
            "models": rows,
        }
    )
    with open(run_dir / RUN_THRESHOLDS_JSON, "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2, allow_nan=False)

    return rows


def _json_safe(value: Any) -> Any:
    """Replace NaN/Inf with ``None`` so ``json.dump(..., allow_nan=False)`` works."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
