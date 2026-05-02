"""Pickle portability: MaxEnt pipelines should not depend on ``sdm`` for the estimator."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from elapid.models import MaxentModel as ElapidMaxentModel

from sdm.models.core.pipeline_features import pipeline_selected_feature_names
from sdm.models.maxent.maxent_model import (
    DefaultMaxentConfig,
    create_maxent_pipeline,
    elapid_maxent_from_config,
)
from sdm.types import ModelConfig


def _pickle_refs_sdm_maxent_wrapper(blob: bytes) -> bool:
    """True if blob encodes the ``sdm`` MaxEnt wrapper class (GLOBAL / STACK_GLOBAL style)."""
    return b"sdm.models.maxent.maxent_model\nMaxentModel" in blob


@pytest.fixture
def tiny_xy():
    rng = np.random.default_rng(0)
    n = 40
    cols = ["a", "b", "c"]
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    y = (X["a"] + X["b"] > 0).astype(int)
    w = np.ones(n)
    return X, y, w, cols


def test_create_maxent_pipeline_uses_elapid_estimator(tiny_xy):
    X, y, w, cols = tiny_xy
    pipe = create_maxent_pipeline(
        feature_names=["a", "b"],
        maxent_n_jobs=1,
        model_config=DefaultMaxentConfig(),
    )
    assert isinstance(pipe.named_steps["maxent"], ElapidMaxentModel)
    pipe.fit(X, y, maxent__sample_weight=w)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_elapid_maxent_from_config_matches_wrapper_from_config():
    cfg = DefaultMaxentConfig(beta_lqp=0.7, beta_threshold=2.3, clamp=False)
    direct = elapid_maxent_from_config(cfg, n_cpus=1)
    from sdm.models.maxent.maxent_model import MaxentModel

    wrapped = MaxentModel.from_config(cfg, n_cpus=1)
    for attr in (
        "feature_types",
        "tau",
        "transform",
        "clamp",
        "beta_multiplier",
        "beta_lqp",
        "beta_hinge",
        "beta_categorical",
        "n_hinge_features",
        "n_threshold_features",
        "use_lambdas",
        "n_lambdas",
        "class_weights",
    ):
        assert getattr(direct, attr) == getattr(wrapped, attr)
    assert direct.beta_threshold == wrapped.beta_threshold
    assert direct.beta_threshold == 2.3
    assert wrapped.beta_threshold == 2.3


def test_default_maxent_config_from_model_config_preserves_clamp():
    cfg = ModelConfig(
        maxent={
            "feature_types": ["linear", "hinge"],
            "beta_multiplier": 1.5,
            "beta_lqp": 0.8,
            "beta_hinge": 1.2,
            "beta_threshold": 2.4,
            "beta_categorical": 1.1,
            "n_hinge_features": 10,
            "n_threshold_features": 0,
            "clamp": False,
            "convergence_tolerance": 1e-5,
            "use_lambdas": "best",
            "n_lambdas": 100,
            "class_weights": 100,
            "tau": 0.5,
            "transform": "cloglog",
        }
    )

    maxent_config = DefaultMaxentConfig.from_config(cfg)

    assert maxent_config.clamp is False
    assert maxent_config.beta_threshold == 2.4


def test_fitted_pipeline_pickle_roundtrip(tiny_xy):
    X, y, w, cols = tiny_xy
    pipe = create_maxent_pipeline(
        feature_names=cols,
        maxent_n_jobs=1,
        model_config=DefaultMaxentConfig(),
    )
    pipe.fit(X, y, maxent__sample_weight=w)
    blob = pickle.dumps(pipe)
    loaded = pickle.loads(blob)
    assert pipeline_selected_feature_names(loaded) == cols
    np.testing.assert_allclose(
        pipe.predict_proba(X), loaded.predict_proba(X), rtol=1e-10, atol=1e-10
    )


def test_fitted_pipeline_pickle_avoids_sdm_maxent_wrapper_class(tiny_xy):
    X, y, w, cols = tiny_xy
    pipe = create_maxent_pipeline(feature_names=cols, maxent_n_jobs=1)
    pipe.fit(X, y, maxent__sample_weight=w)
    blob = pickle.dumps(pipe)
    assert not _pickle_refs_sdm_maxent_wrapper(blob)
