"""Seed trial and block re-score options for tuning (synthetic data only)."""

import json

import numpy as np
import optuna
import pytest

from sdm.commands.modelling import tune_hyperparameters as th
from sdm.models.maxent.folds import FoldMethod, resolve_fold_params
from sdm.types import CvConfig, ModelConfig
from tests.test_fold_methods import _clustered_occurrence

FEATURES = ["feature_a", "feature_b"]


def _model_config() -> ModelConfig:
    return ModelConfig(
        seed=42,
        maxent={
            "feature_types": ["linear"],
            "beta_multiplier": 1.5,
            "beta_lqp": 1.0,
            "beta_hinge": 1.0,
            "beta_threshold": 1.0,
            "beta_categorical": 1.0,
            "n_hinge_features": 10,
            "n_threshold_features": 0,
            "clamp": True,
            "convergence_tolerance": 1e-5,
            "use_lambdas": "best",
            "n_lambdas": 100,
            "class_weights": 100,
            "tau": 0.5,
            "transform": "cloglog",
        },
    )


def test_load_seed_features_yaml_and_json(tmp_path):
    y = tmp_path / "s.yml"
    y.write_text("sp_roost:\n  - feature_a\n")
    j = tmp_path / "s.json"
    j.write_text(json.dumps({"sp_roost": ["feature_b"]}))
    assert th.load_seed_features(y) == {"sp_roost": ["feature_a"]}
    assert th.load_seed_features(j) == {"sp_roost": ["feature_b"]}
    bad = tmp_path / "bad.yml"
    bad.write_text("- a\n")
    with pytest.raises(ValueError):
        th.load_seed_features(bad)


def test_build_seed_trial_params_switches_only_listed_layers():
    params = th.build_seed_trial_params(["feature_a"], FEATURES)
    assert params["feature_feature_a"] is True
    assert params["feature_feature_b"] is False
    assert params["beta_multiplier"] == th.get_default_trial_params([])["beta_multiplier"]
    with pytest.raises(ValueError):
        th.build_seed_trial_params(["missing"], FEATURES)


def test_seed_trial_is_first_trial_run():
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(th.build_seed_trial_params(["feature_a"], FEATURES))

    def objective(trial):
        th.suggest_maxent_hyperparameters(trial)
        return float(len(th.pick_features(trial, FEATURES)))

    study.optimize(objective, n_trials=1)
    first = study.trials[0].params
    assert first["feature_feature_a"] is True
    assert first["feature_feature_b"] is False
    assert first["beta_multiplier"] == 1.5


def test_trial_params_to_config_round_trip():
    params = th.build_seed_trial_params(["feature_b"], FEATURES)
    maxent, features = th.trial_params_to_config(params, _model_config(), FEATURES)
    assert features == ["feature_b"]
    assert "linear" in maxent["feature_types"]
    assert maxent["n_hinge_features"] == 10


def test_rescore_uses_training_block_folds():
    folds = resolve_fold_params(CvConfig(), model_seed=42)
    assert folds.method is FoldMethod.BLOCKS
    assert folds.n_folds == 5 and folds.block_size_m == 10_000.0

    gdf = _clustered_occurrence(n_clusters=6, seed=1)
    cfg = _model_config()
    study = optuna.create_study(direction="maximize")
    scores = [0.9, 0.5, 0.1]
    for layers in [["feature_a"], FEATURES, ["feature_b"]]:
        study.enqueue_trial(th.build_seed_trial_params(layers, FEATURES))

    def objective(trial):
        th.suggest_maxent_hyperparameters(trial)
        th.pick_features(trial, FEATURES)
        value = scores[trial.number]
        trial.set_user_attr("mean_auc", value)
        trial.set_user_attr("std_auc", 0.0)
        return value

    study.optimize(objective, n_trials=3)
    df = th.rescore_top_trials(
        study, gdf, cfg, FEATURES, "Myotis test", "Roost", top_k=2
    )
    assert list(df["trial_number"]) == [0, 1]
    assert list(df["tuning_mean_auc"]) == [0.9, 0.5]
    assert np.isfinite(df["block_mean_auc"]).all()
    assert "blocks" in df["block_cv"].iloc[0]


def test_prepare_tuning_folds_matches_training_blocks_and_thins_background():
    from sdm.models.maxent.folds import assign_spatial_block_folds

    gdf = _clustered_occurrence(n_clusters=6, background_per_cluster=40, seed=2)
    cfg = _model_config()
    full, labels = th.prepare_tuning_folds(gdf, cfg)
    folds = resolve_fold_params(CvConfig(), model_seed=42)
    expected = assign_spatial_block_folds(
        gdf,
        n_folds=folds.n_folds,
        block_size_m=folds.block_size_m,
        n_tries=folds.block_n_tries,
        random_state=folds.random_state,
    )
    assert len(full) == len(gdf)
    np.testing.assert_array_equal(labels, expected)

    sub, sub_labels = th.prepare_tuning_folds(gdf, cfg, background_fraction=0.25)
    again, _ = th.prepare_tuning_folds(gdf, cfg, background_fraction=0.25)
    assert list(sub.index) == list(again.index)  # same subset every call
    assert (sub["class"] == 1).sum() == (gdf["class"] == 1).sum()
    for f in np.unique(expected):
        n_bg_full = ((expected == f) & (gdf["class"].to_numpy() == 0)).sum()
        n_bg_sub = ((sub_labels == f) & (sub["class"].to_numpy() == 0)).sum()
        assert n_bg_sub == round(n_bg_full * 0.25)

    old, old_labels = th.prepare_tuning_folds(gdf, cfg, cv_method="kmeans")
    assert old_labels is None and len(old) == len(gdf)


def test_per_fold_pruning_stops_weak_trial():
    gdf = _clustered_occurrence(n_clusters=6, seed=3)
    cfg = _model_config()
    _, labels = th.prepare_tuning_folds(gdf, cfg)
    seen = []
    params = th.build_seed_trial_params(FEATURES, FEATURES)
    params["feature_type_product"] = False
    trial = optuna.trial.FixedTrial(params)
    trial.report = lambda value, step: seen.append(step)
    trial.should_prune = lambda: len(seen) >= 1
    with pytest.raises(optuna.TrialPruned):
        th.objective_train_model(
            trial,
            species_training_gdf=gdf,
            base_model_config=cfg,
            available_features=FEATURES,
            latin_name="Myotis test",
            activity_type="Roost",
            min_presence=1,
            fold_labels=labels,
            prune_per_fold=True,
        )
    assert seen == [0]
