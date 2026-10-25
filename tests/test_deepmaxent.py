"""Synthetic CPU tests for experimental DeepMaxent (optional ``dl`` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score

torch = pytest.importorskip("torch")

from sdm.models.deepmaxent.config import DeepMaxentParams
from sdm.models.deepmaxent.data import build_joint_presence_background_table
from sdm.models.deepmaxent.device import resolve_torch_device
from sdm.models.deepmaxent.estimator import wrap_output_pipeline
from sdm.models.deepmaxent.loss import deepmaxent_loss
from sdm.models.deepmaxent.training import (
    fit_ensemble,
    train_deepmaxent_joint,
)
from sdm.models.maxent.folds import FoldMethod, FoldParams
from sdm.models.maxent.maxent_model import (
    DefaultMaxentConfig,
    create_maxent_pipeline,
)
from sdm.models.core.threshold import compute_pooled_oof_percentile
from sdm.types import TrainingData


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_params(**overrides) -> DeepMaxentParams:
    base = dict(
        device="cpu",
        hidden_size=16,
        n_hidden_layers=2,
        learning_rate=2e-3,
        batch_size=32,
        weight_decay=3e-4,
        epochs=40,
        n_seeds=2,
        epsilon=1e-6,
        use_species_weights=False,
        transform="cloglog",
        early_stopping_patience=None,
        seed=0,
    )
    base.update(overrides)
    return DeepMaxentParams(**base)


def _synthetic_species_gdf(
    rng: np.random.Generator,
    *,
    n_presence: int,
    n_background: int,
    coef: np.ndarray,
    feature_names: list[str],
    x0: float = 0.0,
) -> gpd.GeoDataFrame:
    """Presence-background table with a linear logit habitat signal."""
    n = n_presence + n_background
    X = rng.normal(size=(n, len(feature_names)))
    # Presence rows drawn toward high linear predictor.
    scores = X @ coef
    order = np.argsort(scores)[::-1]
    X = X[order]
    y = np.zeros(n, dtype=int)
    y[:n_presence] = 1
    data = {name: X[:, i] for i, name in enumerate(feature_names)}
    data["class"] = y
    data["sample_weight"] = np.ones(n)
    data["geometry"] = [
        Point(x0 + i * 100.0, (i % 7) * 100.0) for i in range(n)
    ]
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


@pytest.fixture
def two_species_training_data() -> list[TrainingData]:
    rng = np.random.default_rng(7)
    feats = ["f1", "f2", "f3"]
    coef_a = np.array([1.5, -0.8, 0.2])
    coef_b = np.array([1.2, -0.5, 0.1])
    gdf_a = _synthetic_species_gdf(
        rng, n_presence=40, n_background=120, coef=coef_a, feature_names=feats, x0=0
    )
    gdf_b = _synthetic_species_gdf(
        rng, n_presence=35, n_background=120, coef=coef_b, feature_names=feats, x0=50_000
    )
    cfg = DefaultMaxentConfig(feature_types=["linear"], beta_multiplier=1.0)
    return [
        TrainingData(
            latin_name="Demo one",
            activity_type="Roost",
            occurrence=gdf_a,
            maxent_config=cfg,
            model_features=feats,
            model_limited=False,
        ),
        TrainingData(
            latin_name="Demo two",
            activity_type="In flight",
            occurrence=gdf_b,
            maxent_config=cfg,
            model_features=feats,
            model_limited=False,
        ),
    ]


def test_resolve_device_cpu():
    assert resolve_torch_device("cpu") == "cpu"


def test_deepmaxent_loss_shapes_and_finite():
    logits = torch.randn(20, 3, dtype=torch.float32)
    targets = torch.zeros(20, 3)
    targets[0, 0] = 1
    targets[5, 1] = 1
    # species 2 absent from batch — epsilon path
    loss = deepmaxent_loss(logits, targets, epsilon=1e-6)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_estimator_fits_and_predicts_unit_interval(two_species_training_data):
    table = build_joint_presence_background_table(two_species_training_data)
    params = _tiny_params(epochs=25, n_seeds=1)
    ensemble, _states, device = fit_ensemble(table, params, n_seeds=1)
    assert device == "cpu"
    proba = ensemble.predict_proba_multi(table.features)
    assert proba.shape == (len(table.features), len(table.output_names))
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    pipe = wrap_output_pipeline(ensemble, table.output_names[0])
    out = pipe.predict_proba(table.features)
    assert out.shape == (len(table.features), 2)
    assert np.all(out[:, 1] >= 0.0) and np.all(out[:, 1] <= 1.0)


def test_linear_synthetic_ranks_close_to_elapid_maxent(two_species_training_data):
    """On a linear habitat, DeepMaxent and elapid MaxEnt rankings should agree."""
    data = two_species_training_data[0]
    feats = data.model_features
    occ = data.occurrence
    X = occ[feats]
    y = occ["class"].to_numpy()
    w = occ["sample_weight"].to_numpy()

    maxent = create_maxent_pipeline(
        feature_names=feats,
        maxent_n_jobs=1,
        model_config=DefaultMaxentConfig(
            feature_types=["linear"],
            beta_multiplier=1.0,
            n_hinge_features=0,
            n_threshold_features=0,
        ),
    )
    maxent.fit(X, y, maxent__sample_weight=w)
    maxent_scores = maxent.predict_proba(X)[:, 1]

    table = build_joint_presence_background_table([data])
    params = _tiny_params(epochs=60, n_seeds=1, hidden_size=8, learning_rate=5e-3)
    ensemble, _, _ = fit_ensemble(table, params, n_seeds=1)
    dm_scores = ensemble.predict_proba_multi(table.features)[:, 0]

    # Align rows: joint table aggregates; score on the per-model occurrence frame
    # via the wrapped pipeline instead.
    pipe = wrap_output_pipeline(ensemble, table.output_names[0])
    dm_on_occ = pipe.predict_proba(X)[:, 1]
    corr, _p = spearmanr(maxent_scores, dm_on_occ)
    assert corr > 0.5, f"Spearman r={corr:.3f} too low vs elapid MaxEnt"


def test_joint_cv_threshold_predict_end_to_end(two_species_training_data, tmp_path: Path):
    params = _tiny_params(epochs=20, n_seeds=1, batch_size=64)
    fold_params = FoldParams(
        method=FoldMethod.BLOCKS,
        n_folds=3,
        random_state=0,
        block_size_m=500.0,
        block_n_tries=50,
    )
    result = train_deepmaxent_joint(
        two_species_training_data, params, fold_params=fold_params
    )
    assert len(result.results) == 2
    assert result.device == "cpu"
    for tr in result.results:
        assert tr.success
        assert tr.final_model is not None
        assert tr.validation_scores is not None
        assert len(tr.validation_scores) > 0
        # Pooled 10th-percentile cut-off path (same rule as MaxEnt).
        cut = compute_pooled_oof_percentile(
            tr.validation_scores, percentile=10.0
        )
        assert "threshold" in cut
        if cut["threshold"] is not None:
            assert np.isfinite(cut["threshold"])
        vs = tr.validation_scores
        if vs["class"].nunique() > 1:
            auc = roc_auc_score(vs["class"], vs["held_out_score"])
            assert 0.0 <= auc <= 1.0

    # Tiny "stack": score a grid of synthetic cells with the packaged estimator.
    grid = result.joint_table.features.copy()
    for tr in result.results:
        pred = tr.final_model.predict_proba(grid)[:, 1]
        assert pred.shape == (len(grid),)
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)

    # Run metadata shape used by train save path.
    meta = {
        "trainer": "deepmaxent",
        "device": result.device,
        "deepmaxent": result.params.to_metadata(),
    }
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2))
    assert json.loads(path.read_text())["device"] == "cpu"
