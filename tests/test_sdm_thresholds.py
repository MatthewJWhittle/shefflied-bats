"""Tests for suitability threshold computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdm.commands.modelling.compute_sdm_thresholds import (
    VALIDATION_SCORES_FILENAME,
    compute_threshold_for_package,
)
from sdm.models.core.threshold import (
    bootstrap_percentile_uncertainty,
    compute_presence_percentile_threshold,
    compute_threshold_from_validation_scores,
)
from sdm.types import ThresholdConfig


def _validation_scores(presence_scores: list[float], background_scores: list[float]) -> pd.DataFrame:
    rows = []
    idx = 0
    for score in presence_scores:
        rows.append(
            {"point_index": idx, "class": 1, "fold": idx % 3, "held_out_score": score}
        )
        idx += 1
    for score in background_scores:
        rows.append(
            {"point_index": idx, "class": 0, "fold": idx % 3, "held_out_score": score}
        )
        idx += 1
    return pd.DataFrame(rows)


class TestPresencePercentileThreshold:
    def test_percentile_correctness(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert compute_presence_percentile_threshold(scores, 10) == pytest.approx(0.19)
        assert compute_presence_percentile_threshold(scores, 50) == pytest.approx(0.55)

    def test_bootstrap_is_deterministic(self) -> None:
        scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        first = bootstrap_percentile_uncertainty(
            scores,
            percentile=10,
            n_bootstrap=500,
            random_state=42,
        )
        second = bootstrap_percentile_uncertainty(
            scores,
            percentile=10,
            n_bootstrap=500,
            random_state=42,
        )
        assert first == second

    def test_bootstrap_handles_single_presence(self) -> None:
        scores = np.array([0.75])
        low, high = bootstrap_percentile_uncertainty(scores, percentile=10, random_state=42)
        assert low == pytest.approx(0.75)
        assert high == pytest.approx(0.75)

    def test_compute_threshold_from_validation_scores(self) -> None:
        validation_scores = _validation_scores(
            presence_scores=[0.1, 0.3, 0.5, 0.7, 0.9],
            background_scores=[0.05, 0.15, 0.25],
        )
        config = ThresholdConfig(
            percentile=20,
            bootstrap_samples=200,
            bootstrap_random_state=7,
        )
        result = compute_threshold_from_validation_scores(validation_scores, config)
        assert result.rule == "presence_percentile"
        assert result.source == "held-out cross-validation"
        assert result.n_presence_records == 5
        assert result.value == pytest.approx(0.26)
        assert result.bootstrap_low is not None
        assert result.bootstrap_high is not None
        assert result.bootstrap_low <= result.value <= result.bootstrap_high


class TestThresholdPackageIntegration:
    def test_compute_threshold_for_package_updates_package_json(
        self,
        tmp_path: Path,
    ) -> None:
        package_dir = tmp_path / "test_species_roost"
        package_dir.mkdir()
        validation_scores = _validation_scores(
            presence_scores=[0.2, 0.4, 0.6, 0.8],
            background_scores=[0.1, 0.3],
        )
        validation_scores.to_parquet(package_dir / VALIDATION_SCORES_FILENAME, index=False)
        package = {
            "schema_version": 1,
            "identifier": "Test species_Roost",
            "latin_name": "Test species",
            "activity_type": "Roost",
        }
        (package_dir / "package.json").write_text(json.dumps(package), encoding="utf-8")

        model_config_path = tmp_path / "model_config.yml"
        model_config_path.write_text(
            """
model:
  maxent:
    feature_types: ["linear"]
    beta_multiplier: 1.0
    beta_lqp: 1.0
    beta_hinge: 1.0
    beta_threshold: 1.0
    beta_categorical: 1.0
    n_hinge_features: 5
    n_threshold_features: 5
    clamp: true
    convergence_tolerance: 1.0e-5
    use_lambdas: "best"
    n_lambdas: 100
    class_weights: "balanced"
    tau: 0.5
    transform: "cloglog"
  threshold:
    rule: "presence_percentile"
    percentile: 25
    bootstrap_samples: 100
    bootstrap_random_state: 1
""".strip(),
            encoding="utf-8",
        )

        value = compute_threshold_for_package(
            package_dir,
            model_config_path=model_config_path,
        )
        assert value == pytest.approx(0.35)

        updated = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
        assert "threshold" in updated
        assert updated["threshold"]["rule"] == "presence_percentile"
        assert updated["threshold"]["source"] == "held-out cross-validation"
        assert updated["threshold"]["n_presence_records"] == 4
