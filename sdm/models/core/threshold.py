"""Suitability threshold computation from held-out cross-validation scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from sdm.types import ThresholdConfig

ThresholdRuleName = Literal["presence_percentile"]


@dataclass(frozen=True)
class ThresholdResult:
    """Computed suitability threshold and uncertainty bounds."""

    value: float
    rule: str
    rule_params: Dict[str, Any]
    source: str
    n_presence_records: int
    bootstrap_low: Optional[float]
    bootstrap_high: Optional[float]


def _presence_scores(validation_scores: pd.DataFrame) -> np.ndarray:
    presence = validation_scores.loc[validation_scores["class"] == 1, "held_out_score"]
    return presence.to_numpy(dtype=float)


def compute_presence_percentile_threshold(
    presence_scores: np.ndarray,
    percentile: float,
) -> float:
    """Return the given percentile of held-out presence suitability scores."""
    if presence_scores.size == 0:
        raise ValueError("Cannot compute threshold: no held-out presence scores")
    return float(np.percentile(presence_scores, percentile))


def bootstrap_percentile_uncertainty(
    presence_scores: np.ndarray,
    percentile: float,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
) -> tuple[Optional[float], Optional[float]]:
    """Bootstrap resampled percentile thresholds for uncertainty bounds."""
    n = presence_scores.size
    if n == 0:
        return None, None
    if n == 1:
        value = float(presence_scores[0])
        return value, value

    rng = np.random.default_rng(random_state)
    boot_thresholds = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(presence_scores, size=n, replace=True)
        boot_thresholds[i] = np.percentile(sample, percentile)

    return (
        float(np.percentile(boot_thresholds, percentile_low)),
        float(np.percentile(boot_thresholds, percentile_high)),
    )


def compute_threshold_from_validation_scores(
    validation_scores: pd.DataFrame,
    config: ThresholdConfig,
) -> ThresholdResult:
    """Apply the configured threshold rule to held-out validation scores."""
    rule = config.rule
    if rule == "presence_percentile":
        return _compute_presence_percentile_threshold(validation_scores, config)
    raise ValueError(f"Unsupported threshold rule: {rule!r}")


def _compute_presence_percentile_threshold(
    validation_scores: pd.DataFrame,
    config: ThresholdConfig,
) -> ThresholdResult:
    presence_scores = _presence_scores(validation_scores)
    threshold_value = compute_presence_percentile_threshold(
        presence_scores,
        config.percentile,
    )
    boot_low, boot_high = bootstrap_percentile_uncertainty(
        presence_scores,
        config.percentile,
        n_bootstrap=config.bootstrap_samples,
        random_state=config.bootstrap_random_state,
        percentile_low=config.bootstrap_percentile_low,
        percentile_high=config.bootstrap_percentile_high,
    )
    return ThresholdResult(
        value=threshold_value,
        rule="presence_percentile",
        rule_params={"percentile": config.percentile},
        source="held-out cross-validation",
        n_presence_records=int(presence_scores.size),
        bootstrap_low=boot_low,
        bootstrap_high=boot_high,
    )


def threshold_result_to_package_dict(result: ThresholdResult) -> Dict[str, Any]:
    """Serialize a threshold result for ``package.json``."""
    return {
        "value": result.value,
        "rule": result.rule,
        "rule_params": result.rule_params,
        "source": result.source,
        "n_presence_records": result.n_presence_records,
        "bootstrap_range": {
            "low": result.bootstrap_low,
            "high": result.bootstrap_high,
            "method": "bootstrap_resample_percentile",
            "bootstrap_samples": None,
        },
    }


def enrich_threshold_package_dict(
    package_threshold: Dict[str, Any],
    config: ThresholdConfig,
) -> Dict[str, Any]:
    """Attach bootstrap metadata that depends on config, not the point scores."""
    package_threshold = dict(package_threshold)
    bootstrap_range = dict(package_threshold.get("bootstrap_range") or {})
    bootstrap_range["bootstrap_samples"] = config.bootstrap_samples
    bootstrap_range["random_state"] = config.bootstrap_random_state
    bootstrap_range["percentile_low"] = config.bootstrap_percentile_low
    bootstrap_range["percentile_high"] = config.bootstrap_percentile_high
    package_threshold["bootstrap_range"] = bootstrap_range
    return package_threshold
