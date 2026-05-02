"""
Tests for explain_sdm_models: save/load explainer artifacts.
"""

import importlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sdm.models.core.feature_subsetter import FeatureSubsetter
from sdm.commands.modelling.explain_sdm_models import (
    save_explainer_artifacts,
    load_explainer_artifacts,
)


@pytest.fixture
def minimal_model_and_data(tmp_path):
    """Minimal pipeline and EV-like DataFrame for artifact tests."""
    feature_names = ["a", "b", "c"]
    np.random.seed(42)
    n = 50
    X = pd.DataFrame(
        np.random.randn(n, 3).astype(np.float32),
        columns=feature_names,
    )
    y = (X["a"] + X["b"] > 0).astype(int)
    pipe = Pipeline([
        ("feature_selection", FeatureSubsetter(feature_names)),
        ("clf", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X, y)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    return pipe, X, feature_names, model_path


def test_save_and_load_explainer_artifacts_roundtrip(minimal_model_and_data, tmp_path):
    """Save model path + background; load reconstructs explainer and shap_values(point) works."""
    pipe, X, feature_names, model_path = minimal_model_and_data
    background = X.sample(n=10, random_state=42)

    artifacts_dir = tmp_path / "artifacts"
    save_explainer_artifacts(artifacts_dir, model_path, background, feature_names)

    assert (artifacts_dir / "meta.json").exists()
    assert (artifacts_dir / "background.parquet").exists()
    assert (artifacts_dir / "feature_names.json").exists()
    assert not (artifacts_dir / "explainer.pkl").exists()

    loaded = load_explainer_artifacts(artifacts_dir)
    assert loaded["feature_names"] == feature_names
    explainer = loaded["explainer"]

    point = X.head(1)
    shap_vals = explainer.shap_values(point)
    assert shap_vals.shape == (1, len(feature_names))


def test_load_explainer_artifacts_missing_dir_raises():
    """Missing artifacts directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Explainer artifacts directory not found"):
        load_explainer_artifacts(Path("/nonexistent/shap/Plecotus_auritus/Roost"))


def test_explain_sdm_models_uses_auto_worker_count(monkeypatch, tmp_path):
    """When n_jobs=None, the computed auto worker count should be passed to Joblib."""

    explain_mod = importlib.import_module("sdm.commands.modelling.explain_sdm_models")
    captured: dict[str, int] = {}

    class FakeParallel:
        def __init__(self, n_jobs, verbose=0):
            captured["n_jobs"] = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        return lambda **kwargs: lambda: func(**kwargs)

    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(explain_mod, "Parallel", FakeParallel)
    monkeypatch.setattr(explain_mod, "delayed", fake_delayed)
    monkeypatch.setattr(
        explain_mod,
        "load_model_index",
        lambda _models_dir: pd.DataFrame(
            [{"latin_name": "Pipistrellus pipistrellus", "activity_type": "Roost"}]
        ),
    )
    monkeypatch.setattr(explain_mod, "filter_models", lambda model_index, *_args: model_index)
    monkeypatch.setattr(explain_mod, "load_environmental_variables", lambda _ev_path: (object(), None))
    monkeypatch.setattr(
        explain_mod,
        "sample_points_from_xarray_dataset",
        lambda **_kwargs: pd.DataFrame({"a": [1.0]}),
    )
    monkeypatch.setattr(
        explain_mod,
        "process_single_model",
        lambda **_kwargs: {
            "latin_name": "Pipistrellus pipistrellus",
            "activity_type": "Roost",
            "model_id": "Pipistrellus_pipistrellus_Roost",
            "success": True,
            "n_features": 1,
            "n_explain": 1,
            "plot_paths": {},
            "yaml_path": None,
            "error": None,
        },
    )

    results = explain_mod.explain_sdm_models(
        ev_path=tmp_path / "ev.tif",
        models_dir=tmp_path,
        output_dir=tmp_path,
        n_jobs=None,
    )

    assert captured["n_jobs"] == 4
    assert results["success"].tolist() == [True]
