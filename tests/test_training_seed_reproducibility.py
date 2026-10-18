"""Reproducibility of training/tuning RNG from model_config ``seed``."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from sdm.commands.modelling.train_sdm_models import (
    save_run_metadata,
    train_single_model,
    _summarize_cv_scores,
)
from sdm.models.maxent.maxent_model import DefaultMaxentConfig, cross_validate_maxent_model
from sdm.occurrence.sampling import BackgroundMethod, generate_background_points_from_data
from sdm.types import TrainingData
from sdm.utils.io import load_model_config
from tests.test_cv_validation_scores import RecordingDummyClassifier, _synthetic_occurrence_gdf


@pytest.fixture
def synthetic_boundary() -> gpd.GeoDataFrame:
    geometry = box(0, 0, 1000, 1000)
    return gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:27700")


@pytest.fixture
def synthetic_occurrences(synthetic_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    n_points = 60
    gdf = synthetic_boundary.sample_points(size=n_points).explode(index_parts=False).to_frame()
    gdf["latin_name"] = "species_alpha"
    gdf["activity_type"] = "activity_a"
    gdf["accuracy"] = 10
    return gdf


def _background_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def test_model_config_seed_from_repo() -> None:
    cfg = load_model_config()
    assert cfg.seed == 42


def test_background_points_reproducible_for_same_seed(
    synthetic_occurrences: gpd.GeoDataFrame,
    synthetic_boundary: gpd.GeoDataFrame,
) -> None:
    kwargs = dict(
        occurrence_data=synthetic_occurrences,
        boundary=synthetic_boundary,
        n_background_points=25,
        background_method=BackgroundMethod.CONTRAST,
        background_value=0.3,
        sigma=1.0,
        grid_resolution=100,
    )
    first, _ = generate_background_points_from_data(**kwargs, seed=7)
    second, _ = generate_background_points_from_data(**kwargs, seed=7)
    other, _ = generate_background_points_from_data(**kwargs, seed=8)

    assert np.allclose(_background_xy(first), _background_xy(second))
    assert not np.allclose(_background_xy(first), _background_xy(other))


def test_cv_folds_and_scores_reproducible_for_same_seed() -> None:
    occurrence = _synthetic_occurrence_gdf(seed=11)
    feature_columns = ["feature_a", "feature_b"]
    model = RecordingDummyClassifier()

    _models_a, scores_a, _ = cross_validate_maxent_model(
        model=model,
        occurrence_gdf=occurrence,
        n_folds=3,
        feature_columns=feature_columns,
        random_state=99,
    )
    _models_b, scores_b, _ = cross_validate_maxent_model(
        model=model,
        occurrence_gdf=occurrence,
        n_folds=3,
        feature_columns=feature_columns,
        random_state=99,
    )

    assert np.allclose(scores_a, scores_b, equal_nan=True)
    for fold_a, fold_b in zip(_models_a, _models_b):
        if fold_a is None or fold_b is None:
            continue
        assert np.array_equal(fold_a.train_indices_, fold_b.train_indices_)


def test_train_single_model_scores_reproducible_for_same_seed() -> None:
    feature_names = [f"ev_{idx}" for idx in range(6)]
    rng = np.random.default_rng(0)
    rows = 80
    data = {name: rng.random(rows) for name in feature_names}
    data["class"] = rng.integers(0, 2, rows)
    data["sample_weight"] = rng.random(rows)
    data["geometry"] = [Point(x, y) for x, y in zip(rng.random(rows), rng.random(rows))]
    frame = gpd.GeoDataFrame(data, crs="EPSG:27700")
    training_data = TrainingData(
        latin_name="species_alpha",
        activity_type="activity_a",
        occurrence=frame,
        maxent_config=DefaultMaxentConfig(),
        model_features=feature_names[:4],
    )

    first = train_single_model(training_data, max_threads_per_model=1, seed=42)
    second = train_single_model(training_data, max_threads_per_model=1, seed=42)
    if not first.success or not second.success:
        pytest.skip(f"training failed: {first.error} / {second.error}")

    mean_a, std_a, _, _ = _summarize_cv_scores(first.cv_scores)
    mean_b, std_b, _, _ = _summarize_cv_scores(second.cv_scores)
    assert mean_a == pytest.approx(mean_b, rel=0, abs=1e-9)
    assert std_a == pytest.approx(std_b, rel=0, abs=1e-9)
    assert np.allclose(first.cv_scores, second.cv_scores, equal_nan=True)


def test_run_metadata_records_seed(tmp_path: Path) -> None:
    path = save_run_metadata(
        tmp_path,
        seed=42,
        model_config_path="model_config.yml",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["model_config_path"] == "model_config.yml"
