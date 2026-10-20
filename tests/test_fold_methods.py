"""Synthetic tests for selectable CV fold methods (#63)."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
from sklearn.dummy import DummyClassifier

from sdm.models.maxent.folds import (
    DEFAULT_BUFFERED_LOO_WARN_ABOVE,
    FoldMethod,
    assign_spatial_block_folds,
    buffered_loo_presence_warning,
    fold_assignments_frame,
    iter_fold_splits,
    resolve_fold_params,
)
from sdm.models.maxent.maxent_model import cross_validate_maxent_model
from sdm.types import CvConfig


def _clustered_occurrence(
    *,
    n_clusters: int = 5,
    presence_per_cluster: int = 8,
    background_per_cluster: int = 12,
    cluster_spacing_m: float = 30_000.0,
    jitter_m: float = 1_000.0,
    seed: int = 0,
) -> gpd.GeoDataFrame:
    """Synthetic points in well-separated clusters (British National Grid metres).

    Cluster centres sit in the middle of 10 km blocks so default block CV keeps
    each cluster inside one block.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for cluster_i in range(n_clusters):
        # Offset by half a block so jitter cannot cross a 10 km boundary.
        cx = 5_000.0 + cluster_i * cluster_spacing_m
        cy = 5_000.0 + (cluster_i % 2) * cluster_spacing_m
        for _ in range(presence_per_cluster):
            rows.append(
                {
                    "feature_a": float(rng.normal()),
                    "feature_b": float(rng.normal()),
                    "class": 1,
                    "sample_weight": 1.0,
                    "geometry": Point(
                        cx + float(rng.uniform(-jitter_m, jitter_m)),
                        cy + float(rng.uniform(-jitter_m, jitter_m)),
                    ),
                }
            )
        for _ in range(background_per_cluster):
            rows.append(
                {
                    "feature_a": float(rng.normal()),
                    "feature_b": float(rng.normal()),
                    "class": 0,
                    "sample_weight": 1.0,
                    "geometry": Point(
                        cx + float(rng.uniform(-jitter_m, jitter_m)),
                        cy + float(rng.uniform(-jitter_m, jitter_m)),
                    ),
                }
            )
    return gpd.GeoDataFrame(rows, crs="EPSG:27700")


def _default_kmeans_occurrence(seed: int = 0) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    n_presence, n_background = 18, 36
    n_total = n_presence + n_background
    return gpd.GeoDataFrame(
        {
            "feature_a": rng.normal(size=n_total),
            "feature_b": rng.normal(size=n_total),
            "class": np.array([1] * n_presence + [0] * n_background),
            "sample_weight": np.ones(n_total),
            "geometry": [
                Point(x, y)
                for x, y in zip(rng.uniform(0, 100, n_total), rng.uniform(0, 100, n_total))
            ],
        },
        crs="EPSG:27700",
    )


class TestResolveFoldParams:
    def test_default_is_kmeans_with_three_folds(self) -> None:
        params = resolve_fold_params(CvConfig(), model_seed=42)
        assert params.method == FoldMethod.KMEANS
        assert params.n_folds == 3
        assert params.random_state == 42

    def test_blocks_defaults(self) -> None:
        params = resolve_fold_params(
            CvConfig(method="blocks"),
            model_seed=7,
        )
        assert params.method == FoldMethod.BLOCKS
        assert params.n_folds == 5
        assert params.block_size_m == 10_000.0
        assert params.random_state == 7


class TestSpatialBlockFolds:
    def test_blocks_balance_presence_counts(self) -> None:
        gdf = _clustered_occurrence(n_clusters=5, presence_per_cluster=10)
        labels = assign_spatial_block_folds(
            gdf,
            n_folds=5,
            block_size_m=10_000.0,
            n_tries=200,
            random_state=11,
            class_col="class",
        )
        presence = gdf["class"].to_numpy() == 1
        counts = [int(np.sum(presence & (labels == fold))) for fold in range(5)]
        assert min(counts) >= 1
        # Five equal clusters → one cluster per fold when balance works.
        assert max(counts) - min(counts) <= 2

    def test_blocks_never_single_class(self) -> None:
        gdf = _clustered_occurrence(n_clusters=5, presence_per_cluster=8)
        labels = assign_spatial_block_folds(
            gdf,
            n_folds=5,
            block_size_m=10_000.0,
            n_tries=200,
            random_state=3,
            class_col="class",
        )
        for fold in range(5):
            fold_classes = set(gdf.loc[labels == fold, "class"].tolist())
            assert fold_classes == {0, 1}

    def test_blocks_background_inherits_block_fold(self) -> None:
        gdf = _clustered_occurrence(n_clusters=4, presence_per_cluster=6)
        labels = assign_spatial_block_folds(
            gdf,
            n_folds=4,
            block_size_m=10_000.0,
            n_tries=100,
            random_state=5,
            class_col="class",
        )
        xs = gdf.geometry.x.to_numpy()
        ys = gdf.geometry.y.to_numpy()
        block_ids = np.column_stack(
            [np.floor(xs / 10_000.0).astype(int), np.floor(ys / 10_000.0).astype(int)]
        )
        # Points in the same block share a fold label.
        for block in np.unique(block_ids, axis=0):
            mask = (block_ids == block).all(axis=1)
            assert len(set(labels[mask].tolist())) == 1

    def test_blocks_seed_reproducible(self) -> None:
        gdf = _clustered_occurrence(seed=9)
        a = assign_spatial_block_folds(
            gdf, n_folds=5, block_size_m=10_000.0, n_tries=80, random_state=21
        )
        b = assign_spatial_block_folds(
            gdf, n_folds=5, block_size_m=10_000.0, n_tries=80, random_state=21
        )
        c = assign_spatial_block_folds(
            gdf, n_folds=5, block_size_m=10_000.0, n_tries=80, random_state=22
        )
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestBufferedLooPresenceWarning:
    def test_warns_above_threshold_not_at_or_below(self) -> None:
        threshold = DEFAULT_BUFFERED_LOO_WARN_ABOVE
        below = buffered_loo_presence_warning(
            threshold - 1,
            threshold,
            latin_name="Syntheticus testus",
            activity_type="In flight",
        )
        at = buffered_loo_presence_warning(
            threshold,
            threshold,
            latin_name="Syntheticus testus",
            activity_type="In flight",
        )
        above = buffered_loo_presence_warning(
            threshold + 1,
            threshold,
            latin_name="Syntheticus testus",
            activity_type="In flight",
        )
        assert below is None
        assert at is None
        assert above is not None
        assert str(threshold + 1) in above
        assert str(threshold) in above
        assert "blocks" in above
        assert "kmeans" in above

    def test_resolve_fold_params_carries_warn_threshold(self) -> None:
        params = resolve_fold_params(
            CvConfig(method="buffered_loo", buffered_loo_warn_above=40),
            model_seed=1,
        )
        assert params.buffered_loo_warn_above == 40
        assert params.to_metadata()["buffered_loo_warn_above"] == 40

        overridden = resolve_fold_params(
            CvConfig(method="buffered_loo", buffered_loo_warn_above=40),
            model_seed=1,
            buffered_loo_warn_above_override=10,
        )
        assert overridden.buffered_loo_warn_above == 10

    def test_train_single_model_logs_warning_above_threshold(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warning fires per model above threshold; not at/below. Training continues."""
        from sdm.commands.modelling.train_sdm_models import train_single_model
        from sdm.types import TrainingData

        class _StubMaxent:
            feature_types = ["linear"]

        def _fake_evaluate_and_train(**_kwargs):
            return object(), [], np.array([0.5]), None

        monkeypatch.setattr(
            "sdm.commands.modelling.train_sdm_models.create_maxent_pipeline",
            lambda **_kwargs: object(),
        )
        monkeypatch.setattr(
            "sdm.commands.modelling.train_sdm_models.evaluate_and_train_maxent_model",
            _fake_evaluate_and_train,
        )

        threshold = 5
        params = resolve_fold_params(
            CvConfig(method="buffered_loo", buffered_loo_warn_above=threshold),
            model_seed=0,
        )

        def _occurrence(n_presence: int) -> gpd.GeoDataFrame:
            n_bg = max(n_presence, 4)
            rng = np.random.default_rng(0)
            n_total = n_presence + n_bg
            return gpd.GeoDataFrame(
                {
                    "feature_a": rng.normal(size=n_total),
                    "feature_b": rng.normal(size=n_total),
                    "class": [1] * n_presence + [0] * n_bg,
                    "sample_weight": np.ones(n_total),
                    "geometry": [
                        Point(float(i * 1000), 0.0) for i in range(n_total)
                    ],
                },
                crs="EPSG:27700",
            )

        for n_presence, expect_warn in ((threshold, False), (threshold + 1, True)):
            data = TrainingData(
                latin_name="Syntheticus testus",
                activity_type="In flight",
                occurrence=_occurrence(n_presence),
                maxent_config=_StubMaxent(),
                model_features=["feature_a", "feature_b"],
            )
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                result = train_single_model(
                    data,
                    max_threads_per_model=1,
                    seed=0,
                    fold_params=params,
                )
            assert result.success
            messages = [
                r.message
                for r in caplog.records
                if "buffered_loo used with" in r.message
            ]
            if expect_warn:
                assert messages, f"expected warning for n_presence={n_presence}"
                assert str(n_presence) in messages[0]
                assert str(threshold) in messages[0]
                assert "blocks" in messages[0] and "kmeans" in messages[0]
            else:
                assert not messages, f"unexpected warning for n_presence={n_presence}"


class TestBufferedLoo:
    def test_buffer_excludes_nearby_training_points(self) -> None:
        # Presence at origin; nearby background inside buffer; far background outside.
        gdf = gpd.GeoDataFrame(
            {
                "feature_a": [0.0, 0.0, 0.0],
                "feature_b": [0.0, 0.0, 0.0],
                "class": [1, 0, 0],
                "sample_weight": [1.0, 1.0, 1.0],
                "geometry": [
                    Point(0.0, 0.0),
                    Point(100.0, 0.0),  # inside 10 km buffer
                    Point(20_000.0, 0.0),  # outside
                ],
            },
            crs="EPSG:27700",
        )
        splits = list(
            iter_fold_splits(
                gdf,
                resolve_fold_params(
                    CvConfig(method="buffered_loo", buffer_m=10_000.0),
                    model_seed=0,
                ),
            )
        )
        assert len(splits) == 1
        fold_id, train_idx, test_idx = splits[0]
        assert fold_id == 0
        assert set(test_idx.tolist()) == {0, 1}
        assert set(train_idx.tolist()) == {2}
        assert 1 not in train_idx  # nearby background excluded from training

    def test_buffered_loo_seed_stable_split_order(self) -> None:
        gdf = _clustered_occurrence(n_clusters=3, presence_per_cluster=3, seed=4)
        params = resolve_fold_params(
            CvConfig(method="buffered_loo", buffer_m=5_000.0),
            model_seed=42,
        )
        a = [(t.copy(), v.copy()) for _, t, v in iter_fold_splits(gdf, params)]
        b = [(t.copy(), v.copy()) for _, t, v in iter_fold_splits(gdf, params)]
        assert len(a) == len(b)
        for (ta, va), (tb, vb) in zip(a, b):
            assert np.array_equal(ta, tb)
            assert np.array_equal(va, vb)


class TestKmeansDefaultUnchanged:
    def test_default_matches_legacy_geographic_kfold_indices(self) -> None:
        import elapid as ela

        from sdm.utils.rng import apply_training_seed

        gdf = _default_kmeans_occurrence(seed=3)
        apply_training_seed(42)
        legacy = list(ela.GeographicKFold(n_splits=3).split(gdf))

        params = resolve_fold_params(CvConfig(), model_seed=42)
        modern = list(iter_fold_splits(gdf, params))

        assert len(modern) == 3
        for (legacy_train, legacy_test), (_fold, train, test) in zip(legacy, modern):
            assert np.array_equal(legacy_train, train)
            assert np.array_equal(legacy_test, test)


class TestBufferedLooThresholdCompatibility:
    def test_p5_fold_median_is_median_of_held_out_presence_scores(self) -> None:
        """With one presence per fold, fold p5 equals that score; median is overall median."""
        import pandas as pd

        from sdm.models.core.threshold import compute_p5_fold_median

        presence_scores = [0.10, 0.20, 0.30, 0.40, 0.50]
        rows = []
        for fold, score in enumerate(presence_scores):
            rows.append(
                {
                    "point_index": fold,
                    "class": 1,
                    "fold": fold,
                    "held_out_score": score,
                }
            )
            rows.append(
                {
                    "point_index": 100 + fold,
                    "class": 0,
                    "fold": fold,
                    "held_out_score": 0.01,
                }
            )
        result = compute_p5_fold_median(pd.DataFrame(rows))
        assert result["threshold"] == pytest.approx(float(np.median(presence_scores)))


class TestCrossValidateWarningsAndAssignments:
    def test_skip_logs_warning_with_fold_and_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        # All background in one tight cluster far from a single presence cluster so
        # KMeans can produce a single-class fold; force a tiny, skewed layout.
        gdf = gpd.GeoDataFrame(
            {
                "feature_a": np.linspace(0, 1, 12),
                "feature_b": np.linspace(1, 0, 12),
                "class": [1, 1, 1] + [0] * 9,
                "sample_weight": np.ones(12),
                "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]
                + [Point(100_000 + i, 0) for i in range(9)],
            },
            crs="EPSG:27700",
        )
        model = DummyClassifier(strategy="prior")
        with caplog.at_level(logging.WARNING):
            cross_validate_maxent_model(
                model=model,
                occurrence_gdf=gdf,
                n_folds=3,
                feature_columns=["feature_a", "feature_b"],
                collect_validation_scores=True,
                random_state=0,
                fold_method=FoldMethod.KMEANS,
            )
        skip_messages = [r.message for r in caplog.records if "Skipping fold" in r.message]
        assert skip_messages, "expected a warning when a fold is skipped"
        assert any("fold" in msg.lower() for msg in skip_messages)

    def test_fold_assignments_saved_for_blocks(self) -> None:
        gdf = _clustered_occurrence(n_clusters=5, seed=2)
        params = resolve_fold_params(CvConfig(method="blocks"), model_seed=1)
        frame = fold_assignments_frame(gdf, params)
        assert set(frame.columns) >= {"point_index", "class", "fold"}
        assert len(frame) == len(gdf)
        assert frame["fold"].nunique() == 5

    def test_cv_seed_reproducible_for_blocks(self) -> None:
        gdf = _clustered_occurrence(seed=8)
        model = DummyClassifier(strategy="prior")
        kwargs = dict(
            model=model,
            occurrence_gdf=gdf,
            n_folds=5,
            feature_columns=["feature_a", "feature_b"],
            collect_validation_scores=True,
            random_state=17,
            fold_method=FoldMethod.BLOCKS,
            block_size_m=10_000.0,
        )
        _m1, s1, v1 = cross_validate_maxent_model(**kwargs)
        _m2, s2, v2 = cross_validate_maxent_model(**kwargs)
        assert np.allclose(s1, s2, equal_nan=True)
        assert v1 is not None and v2 is not None
        assert v1["fold"].tolist() == v2["fold"].tolist()
        assert v1["point_index"].tolist() == v2["point_index"].tolist()
