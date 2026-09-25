"""Tests for held-out cross-validation score capture."""

from __future__ import annotations

import inspect

import elapid as ela
import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier

from sdm.models.maxent.maxent_model import (
    cross_validate_maxent_model,
    evaluate_and_train_maxent_model,
)

_GEOGRAPHIC_KFOLD_SUPPORTS_RANDOM_STATE = (
    "random_state" in inspect.signature(ela.GeographicKFold.__init__).parameters
)


def _synthetic_occurrence_gdf(
    n_presence: int = 18,
    n_background: int = 36,
    seed: int = 0,
) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_presence + n_background
    classes = np.array([1] * n_presence + [0] * n_background)
    data = {
        "feature_a": rng.normal(size=n_total),
        "feature_b": rng.normal(size=n_total),
        "class": classes,
        "sample_weight": np.ones(n_total),
        "geometry": [
            Point(x, y)
            for x, y in zip(rng.uniform(0, 100, n_total), rng.uniform(0, 100, n_total))
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


class RecordingDummyClassifier(DummyClassifier):
    """Dummy classifier that records which training indices each fold saw."""

    def __init__(self) -> None:
        super().__init__(strategy="prior")
        self.train_indices_: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001
        if hasattr(X, "index"):
            self.train_indices_ = np.asarray(X.index)
        else:
            self.train_indices_ = np.arange(len(y))
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture
def occurrence_gdf() -> gpd.GeoDataFrame:
    return _synthetic_occurrence_gdf()


@pytest.fixture
def feature_columns() -> list[str]:
    return ["feature_a", "feature_b"]


class TestCollectCvValidationScores:
    def test_each_presence_gets_one_held_out_score(
        self,
        occurrence_gdf: gpd.GeoDataFrame,
        feature_columns: list[str],
    ) -> None:
        model = RecordingDummyClassifier()
        _models, _scores, validation_scores = cross_validate_maxent_model(
            model=model,
            occurrence_gdf=occurrence_gdf,
            n_folds=3,
            feature_columns=feature_columns,
            random_state=42,
            collect_validation_scores=True,
        )

        assert validation_scores is not None
        presence = validation_scores[validation_scores["class"] == 1]
        n_presence = int((occurrence_gdf["class"] == 1).sum())
        assert len(presence) == n_presence
        assert presence["point_index"].nunique() == n_presence
        assert len(presence) == len(presence.drop_duplicates("point_index"))

    def test_held_out_score_not_from_training_fold(
        self,
        occurrence_gdf: gpd.GeoDataFrame,
        feature_columns: list[str],
    ) -> None:
        model = RecordingDummyClassifier()
        cv_models, _scores, validation_scores = cross_validate_maxent_model(
            model=model,
            occurrence_gdf=occurrence_gdf,
            n_folds=3,
            feature_columns=feature_columns,
            random_state=42,
            collect_validation_scores=True,
        )

        assert validation_scores is not None
        for fold_idx, fold_model in enumerate(cv_models):
            if fold_model is None:
                continue
            fold_scores = validation_scores[validation_scores["fold"] == fold_idx]
            trained_on = set(fold_model.train_indices_.tolist())
            for point_idx in fold_scores["point_index"]:
                assert point_idx not in trained_on

    @pytest.mark.skipif(
        not _GEOGRAPHIC_KFOLD_SUPPORTS_RANDOM_STATE,
        reason="elapid<1.0.4 does not support GeographicKFold random_state",
    )
    def test_geographic_kfold_is_seeded(
        self,
        occurrence_gdf: gpd.GeoDataFrame,
        feature_columns: list[str],
    ) -> None:
        model = DummyClassifier(strategy="prior")

        def run(seed: int) -> pd.DataFrame:
            _models, _scores, scores = cross_validate_maxent_model(
                model=clone(model),
                occurrence_gdf=occurrence_gdf,
                n_folds=3,
                feature_columns=feature_columns,
                random_state=seed,
                collect_validation_scores=True,
            )
            assert scores is not None
            return scores.sort_values("point_index").reset_index(drop=True)

        first = run(42)
        second = run(42)
        third = run(99)

        pd.testing.assert_frame_equal(first, second)
        assert not first.equals(third)

    def test_evaluate_and_train_returns_validation_scores(
        self,
        occurrence_gdf: gpd.GeoDataFrame,
        feature_columns: list[str],
    ) -> None:
        model = DummyClassifier(strategy="prior")
        final_model, cv_models, cv_scores, validation_scores = evaluate_and_train_maxent_model(
            model=model,
            occurrence_gdf=occurrence_gdf,
            n_cv_folds=3,
            feature_columns=feature_columns,
            cv_random_state=42,
            collect_validation_scores=True,
        )

        assert final_model is not None
        assert len(cv_models) > 0
        assert len(cv_scores) > 0
        assert validation_scores is not None
        assert set(validation_scores.columns) == {
            "point_index",
            "class",
            "fold",
            "held_out_score",
        }
