"""Tests for pluggable suitability threshold rules (TDD)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdm.models.core.threshold import (
    THRESHOLD_RULES,
    evaluate_threshold,
    fit_threshold,
    max_sens_spec_cutoff,
    presence_capture_cutoff,
    presence_percentile_cutoff,
    summarise_thresholds_for_scores,
)


def _scores(
    presence: list[float],
    background: list[float],
    *,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Build a validation_scores-like table with known fold assignment."""
    rows: list[dict] = []
    idx = 0
    for i, score in enumerate(presence):
        rows.append(
            {
                "point_index": idx,
                "class": 1,
                "fold": i % n_folds,
                "held_out_score": score,
            }
        )
        idx += 1
    for i, score in enumerate(background):
        rows.append(
            {
                "point_index": idx,
                "class": 0,
                "fold": i % n_folds,
                "held_out_score": score,
            }
        )
        idx += 1
    return pd.DataFrame(rows)


class TestPresencePercentileCutoff:
    def test_percentile_zero_is_minimum_presence(self) -> None:
        scores = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        assert presence_percentile_cutoff(scores, 0) == pytest.approx(0.2)

    def test_default_tenth_percentile(self) -> None:
        # Linear 0.1..1.0 → 10th percentile is 0.19 under numpy's default method.
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert presence_percentile_cutoff(scores, 10) == pytest.approx(0.19)

    def test_empty_presence_raises(self) -> None:
        with pytest.raises(ValueError, match="presence"):
            presence_percentile_cutoff(np.array([]), 10)


class TestPresenceCaptureCutoff:
    def test_ninety_five_percent_capture(self) -> None:
        # Catching 95% of 20 equally spaced scores → omit lowest 5% → 5th percentile.
        scores = np.linspace(0.0, 1.0, 20)
        expected = float(np.percentile(scores, 5.0))
        assert presence_capture_cutoff(scores, 0.95) == pytest.approx(expected)

    def test_full_capture_is_minimum(self) -> None:
        scores = np.array([0.3, 0.5, 0.7])
        assert presence_capture_cutoff(scores, 1.0) == pytest.approx(0.3)


class TestMaxSensSpecCutoff:
    def test_perfect_separation(self) -> None:
        presence = np.array([0.8, 0.9, 1.0])
        background = np.array([0.0, 0.1, 0.2])
        cutoff = max_sens_spec_cutoff(presence, background)
        # Any cut between 0.2 and 0.8 maximises sens+spec; sklearn's roc_curve
        # returns an operating point that separates the two clouds.
        assert 0.2 < cutoff <= 0.8
        metrics = evaluate_threshold(cutoff, presence, background)
        assert metrics["presence_capture"] == pytest.approx(1.0)
        assert metrics["background_suitable"] == pytest.approx(0.0)

    def test_uses_roc_curve_operating_point(self) -> None:
        presence = np.array([0.1, 0.4, 0.6, 0.9])
        background = np.array([0.0, 0.2, 0.3, 0.5])
        cutoff = max_sens_spec_cutoff(presence, background)
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_score = np.concatenate([presence, background])
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        sens_plus_spec = tpr + (1.0 - fpr)
        best = int(np.argmax(sens_plus_spec))
        expected = float(thresholds[best])
        if not np.isfinite(expected):
            expected = float(np.nextafter(y_score.max(), np.inf))
        assert cutoff == pytest.approx(expected)


class TestRuleRegistry:
    def test_registry_contains_required_rules(self) -> None:
        assert set(THRESHOLD_RULES) >= {
            "presence_percentile",
            "max_sens_spec",
            "presence_capture",
        }

    def test_fit_threshold_dispatches(self) -> None:
        presence = np.array([0.2, 0.4, 0.6, 0.8])
        background = np.array([0.1, 0.3])
        cut = fit_threshold(
            "presence_percentile",
            presence,
            background,
            params={"percentile": 25},
        )
        assert cut == pytest.approx(float(np.percentile(presence, 25)))

    def test_unknown_rule_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown"):
            fit_threshold("not_a_rule", np.array([1.0]), np.array([0.0]))


class TestFoldWiseNoLeakage:
    def test_cutoff_fit_on_other_folds_only(self) -> None:
        # Fold 0 presence scores are uniquely high; if fold 0 leaked into the
        # fit for evaluating fold 0, the cutoff would be pulled upward.
        scores = _scores(
            presence=[0.95, 0.96, 0.97, 0.10, 0.12, 0.14, 0.20, 0.22, 0.24],
            background=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            n_folds=3,
        )
        summary = summarise_thresholds_for_scores(
            scores,
            rules=["presence_percentile"],
            rule_params={"presence_percentile": {"percentile": 10}},
        )
        row = summary.iloc[0]
        # Fold-wise: when evaluating fold 0, fit uses folds 1+2 only
        # (presence ~0.10–0.24), so cutoffs stay low; mean cutoff << 0.95.
        assert row["cutoff"] < 0.3
        assert "presence_capture" in summary.columns
        assert "background_suitable" in summary.columns
        assert "auc" in summary.columns

    def test_pooled_when_no_fold_column(self) -> None:
        scores = _scores(
            presence=[0.2, 0.4, 0.6, 0.8],
            background=[0.05, 0.15],
            n_folds=2,
        ).drop(columns=["fold"])
        summary = summarise_thresholds_for_scores(
            scores,
            rules=["presence_percentile"],
            rule_params={"presence_percentile": {"percentile": 25}},
        )
        assert len(summary) == 1
        assert summary.iloc[0]["cutoff"] == pytest.approx(
            float(np.percentile([0.2, 0.4, 0.6, 0.8], 25))
        )
        assert summary.iloc[0]["n_folds_evaluated"] == 0


class TestCommandIntegration:
    def test_compute_writes_tidy_table(self, tmp_path: Path) -> None:
        from sdm.commands.modelling.compute_sdm_thresholds import (
            compute_sdm_thresholds,
        )

        models_dir = tmp_path / "models"
        pkg = models_dir / "synthetic_species_roost"
        pkg.mkdir(parents=True)
        scores = _scores(
            presence=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
            background=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        )
        scores.to_parquet(pkg / "validation_scores.parquet", index=False)
        (pkg / "package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "identifier": "Synthetic species_Roost",
                    "model_id": "synthetic_species_roost",
                    "latin_name": "Synthetic species",
                    "activity_type": "Roost",
                    "artifacts": {
                        "model_pickle": "model.pkl",
                        "validation_scores": "validation_scores.parquet",
                    },
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "threshold_out"
        result = compute_sdm_thresholds(
            models_dir=models_dir,
            output_dir=out,
            rules=["presence_percentile", "max_sens_spec", "presence_capture"],
            presence_percentile=10.0,
            presence_capture=0.95,
        )
        assert (out / "threshold_summary.csv").is_file()
        table = pd.read_csv(out / "threshold_summary.csv")
        assert set(table["rule"]) == {
            "presence_percentile",
            "max_sens_spec",
            "presence_capture",
        }
        assert (table["model_id"] == "synthetic_species_roost").all()
        # Must not mutate published package metadata.
        package = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
        assert "threshold" not in package
        assert result is not None
        assert len(result) == 3
