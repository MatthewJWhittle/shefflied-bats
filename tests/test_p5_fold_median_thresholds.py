"""TDD: pooled OOF-percentile cut-offs written into the training run folder (#67)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from sdm.commands.modelling.compute_sdm_thresholds import publish_run_thresholds
from sdm.commands.modelling.train_sdm_models import (
    save_models,
    save_run_thresholds,
    save_training_data,
    train_single_model,
)
from sdm.models.core.threshold import (
    compute_pooled_oof_percentile,
    write_run_thresholds,
)
from sdm.models.maxent.folds import FoldMethod, FoldParams
from sdm.models.maxent.maxent_model import DefaultMaxentConfig
from sdm.types import TrainingData


def _fold_scores(
    fold_presence: dict[int, list[float]],
    fold_background: dict[int, list[float]] | None = None,
) -> pd.DataFrame:
    """Build validation_scores with explicit per-fold presence/background scores."""
    rows: list[dict] = []
    idx = 0
    for fold, scores in fold_presence.items():
        for score in scores:
            rows.append(
                {
                    "point_index": idx,
                    "class": 1,
                    "fold": fold,
                    "held_out_score": score,
                }
            )
            idx += 1
    background = fold_background or {
        fold: [0.01 * (i + 1) for i in range(len(scores))]
        for fold, scores in fold_presence.items()
    }
    for fold, scores in background.items():
        for score in scores:
            rows.append(
                {
                    "point_index": idx,
                    "class": 0,
                    "fold": fold,
                    "held_out_score": score,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _kmeans_fold_params(seed: int) -> FoldParams:
    return FoldParams(method=FoldMethod.KMEANS, n_folds=3, random_state=seed)


class TestPooledOofPercentile:
    def test_threshold_is_percentile_of_pooled_held_out_presence(self) -> None:
        fold_presence = {
            0: list(np.linspace(0.10, 0.50, 20)),
            1: list(np.linspace(0.20, 0.60, 20)),
            2: list(np.linspace(0.30, 0.70, 20)),
        }
        scores = _fold_scores(fold_presence)
        all_presence = np.concatenate([np.asarray(v) for v in fold_presence.values()])
        # 60 presences → 5th percentile of the pool (not median of per-fold p5).
        expected = float(np.percentile(all_presence, 5.0))
        fold_cutoffs = [
            float(np.percentile(fold_presence[f], 5.0)) for f in (0, 1, 2)
        ]
        median_of_fold_p5 = float(np.median(fold_cutoffs))

        result = compute_pooled_oof_percentile(scores)
        assert result["threshold"] == pytest.approx(expected)
        assert result["threshold"] != pytest.approx(median_of_fold_p5)
        assert result["rule"] == "pooled_oof_percentile"
        assert result["percentile"] == pytest.approx(5.0)
        assert result["heldout_caught"] == pytest.approx(
            float(np.mean(all_presence >= expected))
        )
        assert result["no_cutoff_reason"] is None
        assert "fold_median_cutoff" not in result

    def test_uses_fifth_percentile_below_fifty_records(self) -> None:
        """Few-record models use the same 5th percentile (no special rule)."""
        fold_presence = {
            0: list(np.linspace(0.10, 0.90, 10)),
            1: list(np.linspace(0.10, 0.90, 10)),
            2: list(np.linspace(0.10, 0.90, 10)),
        }
        scores = _fold_scores(fold_presence)
        all_presence = np.concatenate([np.asarray(v) for v in fold_presence.values()])
        assert len(all_presence) == 30
        expected = float(np.percentile(all_presence, 5.0))

        result = compute_pooled_oof_percentile(scores)
        assert result["n_presence"] == 30
        assert result["percentile"] == pytest.approx(5.0)
        assert result["threshold"] == pytest.approx(expected)
        assert "few records" in result["flag"]

    def test_model_limited_publishes_no_cutoff(self) -> None:
        fold_presence = {
            0: list(np.linspace(0.2, 0.9, 12)),
            1: list(np.linspace(0.2, 0.9, 12)),
            2: list(np.linspace(0.2, 0.9, 12)),
        }
        scores = _fold_scores(fold_presence)
        map_scores = np.array([0.1, 0.5, 0.9])
        result = compute_pooled_oof_percentile(
            scores,
            map_scores=map_scores,
            model_limited=True,
        )
        assert result["threshold"] is None
        assert result["map_share"] is None
        assert result["heldout_caught"] is None
        assert result["model_limited"] is True
        assert result["no_cutoff_reason"] == "model_limited"
        assert "no cut-off" in result["flag"]

    def test_flags_for_low_auc_and_few_records(self) -> None:
        fold_presence = {
            0: [0.4, 0.5, 0.6],
            1: [0.45, 0.55, 0.65],
            2: [0.35, 0.5, 0.7],
        }
        fold_background = {
            0: [0.35, 0.55, 0.75],
            1: [0.4, 0.5, 0.7],
            2: [0.3, 0.6, 0.8],
        }
        scores = _fold_scores(fold_presence, fold_background)
        result = compute_pooled_oof_percentile(scores)
        assert result["n_presence"] < 50
        assert "few records" in result["flag"]
        assert result["oof_auc"] < 0.75
        assert "model-limited" in result["flag"]
        assert result["threshold"] is not None

    def test_map_share_from_optional_map_scores(self) -> None:
        fold_presence = {
            0: list(np.linspace(0.2, 0.9, 20)),
            1: list(np.linspace(0.2, 0.9, 20)),
            2: list(np.linspace(0.2, 0.9, 20)),
        }
        scores = _fold_scores(fold_presence)
        map_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        result = compute_pooled_oof_percentile(scores, map_scores=map_scores)
        assert result["map_share"] == pytest.approx(
            float(np.mean(map_scores >= result["threshold"]))
        )

    def test_map_share_null_when_no_map_scores(self) -> None:
        fold_presence = {
            0: list(np.linspace(0.2, 0.9, 10)),
            1: list(np.linspace(0.2, 0.9, 10)),
            2: list(np.linspace(0.2, 0.9, 10)),
        }
        scores = _fold_scores(fold_presence)
        result = compute_pooled_oof_percentile(scores)
        assert result["map_share"] is None or (
            isinstance(result["map_share"], float) and np.isnan(result["map_share"])
        )

    def test_synthetic_before_after_vs_fold_median_rule(self) -> None:
        """Fixture-only before/after: pooled p5 vs former median-of-fold-p5."""
        fold_presence = {
            0: list(np.linspace(0.02, 0.20, 8)),
            1: list(np.linspace(0.30, 0.80, 8)),
            2: list(np.linspace(0.35, 0.85, 8)),
        }
        scores = _fold_scores(fold_presence)
        all_presence = np.concatenate([np.asarray(v) for v in fold_presence.values()])
        fold_p5 = [float(np.percentile(fold_presence[f], 5.0)) for f in (0, 1, 2)]
        old_rule = float(np.median(fold_p5))
        new_rule = float(np.percentile(all_presence, 5.0))

        result = compute_pooled_oof_percentile(scores)
        assert result["threshold"] == pytest.approx(new_rule)
        assert old_rule != pytest.approx(new_rule)


class TestWriteRunThresholds:
    def test_writes_thresholds_next_to_model_packages(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "demo"
        pkg = run_dir / "demo_species_roost"
        pkg.mkdir(parents=True)
        fold_presence = {
            0: list(np.linspace(0.1, 0.9, 20)),
            1: list(np.linspace(0.1, 0.9, 20)),
            2: list(np.linspace(0.1, 0.9, 20)),
        }
        scores = _fold_scores(fold_presence)
        scores.to_parquet(pkg / "validation_scores.parquet", index=False)
        (pkg / "package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "identifier": "Demo species_Roost",
                    "model_id": "demo_species_roost",
                    "latin_name": "Demo species",
                    "activity_type": "Roost",
                    "model_limited": False,
                    "metrics": {"mean_cv_auc": 0.85, "n_presence": 60},
                    "artifacts": {
                        "model_pickle": "model.pkl",
                        "validation_scores": "validation_scores.parquet",
                    },
                }
            ),
            encoding="utf-8",
        )

        rows = write_run_thresholds(run_dir)

        assert (run_dir / "thresholds.json").is_file()
        assert (run_dir / "thresholds.csv").is_file()
        assert (pkg / "thresholds.json").is_file()

        payload = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        assert payload["rule"] == "pooled_oof_percentile"
        assert "thresholds" in payload
        assert "demo_species_roost" in payload["thresholds"]
        assert payload["thresholds"]["demo_species_roost"] is not None
        assert len(rows) == 1
        assert rows[0]["model_id"] == "demo_species_roost"
        for key in ("threshold", "rule", "heldout_caught", "map_share", "flag"):
            assert key in rows[0]

        package = json.loads((pkg / "package.json").read_text(encoding="utf-8"))
        assert "threshold" not in package
        assert "thresholds" not in package

    def test_model_limited_package_writes_null_threshold(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "limited"
        pkg = run_dir / "myotis_mystacinus_in_flight"
        pkg.mkdir(parents=True)
        fold_presence = {
            0: list(np.linspace(0.1, 0.9, 12)),
            1: list(np.linspace(0.1, 0.9, 12)),
            2: list(np.linspace(0.1, 0.9, 12)),
        }
        scores = _fold_scores(fold_presence)
        scores.to_parquet(pkg / "validation_scores.parquet", index=False)
        (pkg / "package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "myotis_mystacinus_in_flight",
                    "model_limited": True,
                    "artifacts": {"validation_scores": "validation_scores.parquet"},
                }
            ),
            encoding="utf-8",
        )

        rows = write_run_thresholds(run_dir)
        assert rows[0]["threshold"] is None
        assert rows[0]["no_cutoff_reason"] == "model_limited"

        payload = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        assert payload["thresholds"]["myotis_mystacinus_in_flight"] is None
        pkg_payload = json.loads((pkg / "thresholds.json").read_text(encoding="utf-8"))
        assert pkg_payload["threshold"] is None
        assert pkg_payload["no_cutoff_reason"] == "model_limited"

    def test_refuses_overwrite_when_run_thresholds_exist(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "existing"
        pkg = run_dir / "demo_species_roost"
        pkg.mkdir(parents=True)
        fold_presence = {
            0: list(np.linspace(0.1, 0.9, 12)),
            1: list(np.linspace(0.1, 0.9, 12)),
            2: list(np.linspace(0.1, 0.9, 12)),
        }
        scores = _fold_scores(fold_presence)
        scores.to_parquet(pkg / "validation_scores.parquet", index=False)
        prior = '{"rule":"prior","thresholds":{}}'
        (run_dir / "thresholds.json").write_text(prior, encoding="utf-8")

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_run_thresholds(run_dir)

        assert (run_dir / "thresholds.json").read_text(encoding="utf-8") == prior
        assert not (run_dir / "thresholds.csv").exists()
        assert not (pkg / "thresholds.json").exists()

    def test_refuses_overwrite_when_package_thresholds_exist(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "runs" / "pkg-existing"
        pkg = run_dir / "demo_species_roost"
        pkg.mkdir(parents=True)
        fold_presence = {
            0: list(np.linspace(0.1, 0.9, 12)),
            1: list(np.linspace(0.1, 0.9, 12)),
            2: list(np.linspace(0.1, 0.9, 12)),
        }
        scores = _fold_scores(fold_presence)
        scores.to_parquet(pkg / "validation_scores.parquet", index=False)
        prior_pkg = '{"model_id":"demo_species_roost","threshold":0.5}'
        (pkg / "thresholds.json").write_text(prior_pkg, encoding="utf-8")

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            write_run_thresholds(run_dir)

        assert (pkg / "thresholds.json").read_text(encoding="utf-8") == prior_pkg
        assert not (run_dir / "thresholds.json").exists()


class TestSaveRunThresholdsFromTraining:
    def test_train_helper_writes_run_thresholds(self, tmp_path: Path) -> None:
        feature_names = [f"ev_{idx}" for idx in range(6)]
        rng = np.random.default_rng(0)
        rows = 80
        data = {name: rng.random(rows) for name in feature_names}
        data["class"] = np.array([1] * 40 + [0] * 40)
        data["sample_weight"] = rng.random(rows)
        data["geometry"] = [
            Point(x, y) for x, y in zip(rng.random(rows), rng.random(rows))
        ]
        frame = gpd.GeoDataFrame(data, crs="EPSG:27700")
        training_data = TrainingData(
            latin_name="Species gamma",
            activity_type="Roost",
            occurrence=frame,
            maxent_config=DefaultMaxentConfig(),
            model_features=feature_names[:4],
        )
        result = train_single_model(
            training_data,
            max_threads_per_model=1,
            seed=42,
            fold_params=_kmeans_fold_params(42),
        )
        if not result.success:
            pytest.skip(result.error)

        run_dir = tmp_path / "run"
        save_models([result], [training_data], run_dir)
        save_training_data([training_data], run_dir)
        rows_out = save_run_thresholds([result], [training_data], run_dir)

        assert (run_dir / "thresholds.json").is_file()
        assert (run_dir / "thresholds.csv").is_file()
        model_id = "species_gamma_roost"
        assert (run_dir / model_id / "thresholds.json").is_file()
        assert len(rows_out) == 1
        assert rows_out[0]["rule"] == "pooled_oof_percentile"
        payload = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        assert model_id in payload["thresholds"]
        package = json.loads(
            (run_dir / model_id / "package.json").read_text(encoding="utf-8")
        )
        assert package["model_limited"] is False

    def test_publish_run_thresholds_from_packages(self, tmp_path: Path) -> None:
        feature_names = [f"ev_{idx}" for idx in range(6)]
        rng = np.random.default_rng(1)
        rows = 80
        data = {name: rng.random(rows) for name in feature_names}
        data["class"] = np.array([1] * 40 + [0] * 40)
        data["sample_weight"] = rng.random(rows)
        data["geometry"] = [
            Point(x, y) for x, y in zip(rng.random(rows), rng.random(rows))
        ]
        frame = gpd.GeoDataFrame(data, crs="EPSG:27700")
        training_data = TrainingData(
            latin_name="Species delta",
            activity_type="In flight",
            occurrence=frame,
            maxent_config=DefaultMaxentConfig(),
            model_features=feature_names[:4],
            model_limited=True,
        )
        result = train_single_model(
            training_data,
            max_threads_per_model=1,
            seed=7,
            fold_params=_kmeans_fold_params(7),
        )
        if not result.success:
            pytest.skip(result.error)

        run_dir = tmp_path / "run"
        save_models([result], [training_data], run_dir)

        published = publish_run_thresholds(run_dir)
        assert published is not None
        assert (run_dir / "thresholds.json").is_file()
        assert len(published) == 1
        assert published[0]["model_id"] == "species_delta_in_flight"
        assert published[0]["threshold"] is None
        assert published[0]["no_cutoff_reason"] == "model_limited"

        prior = (run_dir / "thresholds.json").read_text(encoding="utf-8")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            publish_run_thresholds(run_dir)
        assert (run_dir / "thresholds.json").read_text(encoding="utf-8") == prior
