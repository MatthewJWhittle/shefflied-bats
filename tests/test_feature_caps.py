"""Per-variable caps from per-species settings (synthetic data only)."""
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sdm.commands.modelling.train_sdm_models import save_run_metadata
from sdm.models.maxent.maxent_model import (
    create_maxent_pipeline,
    make_capping_transformer,
    resolve_feature_caps,
)
from sdm.types import FeatureCapConfig, VariablesConfig


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 100, n), "b": rng.normal(0, 1, n)})
    y = (X["a"] + rng.normal(0, 20, n) > 60).astype(int)
    return X, y


def test_cap_config_needs_exactly_one_of_quantile_or_value():
    FeatureCapConfig(quantile=0.99)
    FeatureCapConfig(value=45)
    with pytest.raises(ValidationError):
        FeatureCapConfig()
    with pytest.raises(ValidationError):
        FeatureCapConfig(quantile=0.9, value=1)
    with pytest.raises(ValidationError):
        FeatureCapConfig(quantile=1.5)


def test_variables_config_parses_caps_and_rejects_caps_on_unused_variables():
    cfg = VariablesConfig.model_validate(
        {"variables": ["a", "b"], "caps": {"a": {"quantile": 0.99}}}
    )
    assert cfg.validate_features(["a", "b", "c"]).caps["a"].quantile == 0.99
    assert VariablesConfig.model_validate({"variables": ["a"]}).caps == {}
    bad = VariablesConfig.model_validate({"variables": ["a"], "caps": {"b": {"value": 1}}})
    with pytest.raises(ValueError):
        bad.validate_features(["a", "b"])


def test_resolve_quantile_and_fixed_caps_from_training_points():
    X, _ = _synthetic()
    caps = resolve_feature_caps(
        {"a": FeatureCapConfig(quantile=0.9), "b": FeatureCapConfig(value=0.5)}, X
    )
    assert caps["a"] == pytest.approx(np.quantile(X["a"], 0.9))
    assert caps["b"] == 0.5


def test_capping_transformer_clips_only_capped_columns():
    X = pd.DataFrame({"a": [1.0, 50.0, 99.0], "b": [-5.0, 0.0, 500.0]})
    out = make_capping_transformer(["a", "b"], {"a": 40.0}).fit_transform(X)
    assert list(out["a"]) == [1.0, 40.0, 40.0]
    assert list(out["b"]) == [-5.0, 0.0, 500.0]


def test_training_and_prediction_use_the_same_cap():
    X, y = _synthetic()
    cap = resolve_feature_caps({"a": FeatureCapConfig(quantile=0.8)}, X)
    pipe = create_maxent_pipeline(["a", "b"], feature_caps=cap)
    assert "capping" in pipe.named_steps
    pipe.fit(X, y)
    capped_train = pipe[:2].transform(X)
    assert capped_train.iloc[:, 0].max() == pytest.approx(cap["a"])
    # Values above the cap predict exactly like the cap itself
    probe = pd.DataFrame({"a": [cap["a"], cap["a"] + 10, 1e6], "b": [0.0, 0.0, 0.0]})
    p = pipe.predict_proba(probe)[:, 1]
    assert p[0] == pytest.approx(p[1]) and p[0] == pytest.approx(p[2])


def test_pipeline_without_caps_has_no_capping_step():
    assert "capping" not in create_maxent_pipeline(["a", "b"]).named_steps


def test_run_metadata_records_resolved_caps(tmp_path):
    path = save_run_metadata(
        tmp_path, seed=1, model_config_path="model_config.yml",
        feature_caps={"species_x_roost": {"a": 12.5}},
    )
    meta = json.loads(path.read_text())
    assert meta["feature_caps"] == {"species_x_roost": {"a": 12.5}}
    plain = json.loads(save_run_metadata(tmp_path, seed=1, model_config_path="m.yml").read_text())
    assert "feature_caps" not in plain
