"""Registry and MaxEnt adapter contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from sdm.models.adapters import (
    ADAPTERS,
    DEFAULT_TRAINER,
    MaxEntAdapter,
    TrainBundle,
    get_adapter,
    require_tune_support,
    resolve_run_trainer,
    resolve_trainer,
)
from sdm.types import ModelConfig, TrainingData, TrainingResults


def test_get_adapter_default_maxent() -> None:
    adapter = get_adapter(None)
    assert adapter.name == "maxent"
    assert isinstance(adapter, MaxEntAdapter)
    assert "maxent" in ADAPTERS


def test_get_adapter_unknown_type_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown model.type"):
        get_adapter("not-a-real-trainer")


def test_model_config_type_defaults_to_maxent() -> None:
    cfg = ModelConfig(
        seed=1,
        maxent={
            "feature_types": ["linear"],
            "beta_multiplier": 1.0,
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
            "class_weights": "balanced",
            "tau": 0.5,
            "transform": "cloglog",
        },
    )
    assert cfg.type == "maxent"


def test_model_config_type_normalizes_case() -> None:
    cfg = ModelConfig.model_validate(
        {
            "seed": 1,
            "type": " MaxEnt ",
            "maxent": {
                "feature_types": ["linear"],
                "beta_multiplier": 1.0,
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
                "class_weights": "balanced",
                "tau": 0.5,
                "transform": "cloglog",
            },
        }
    )
    assert cfg.type == "maxent"


def test_resolve_trainer_defaults_missing_to_maxent() -> None:
    assert resolve_trainer() == DEFAULT_TRAINER
    assert resolve_trainer(package={}) == DEFAULT_TRAINER
    assert resolve_trainer(package={"trainer": "deepmaxent"}) == "deepmaxent"
    assert resolve_trainer(run_metadata={"trainer": "maxent"}) == "maxent"


def test_resolve_run_trainer_prefers_metadata_then_package(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert resolve_run_trainer(run) == "maxent"

    (run / "run_metadata.json").write_text(
        json.dumps({"schema_version": 1, "trainer": "maxent"}),
        encoding="utf-8",
    )
    assert resolve_run_trainer(run) == "maxent"

    pkg = run / "some_model"
    pkg.mkdir()
    (pkg / "package.json").write_text(
        json.dumps({"trainer": "maxent", "schema_version": 1}),
        encoding="utf-8",
    )
    # Missing trainer on old runs:
    old = tmp_path / "old_run"
    old.mkdir()
    old_pkg = old / "legacy_model"
    old_pkg.mkdir()
    (old_pkg / "package.json").write_text(
        json.dumps({"schema_version": 1, "model_id": "legacy_model"}),
        encoding="utf-8",
    )
    assert resolve_run_trainer(old) == "maxent"


def test_require_tune_support_refuses_non_tune_adapter() -> None:
    class StubAdapter:
        name = "stub"
        supports_tune = False

        def train(
            self,
            training_data: Sequence[TrainingData],
            *,
            seed: int,
            fold_params: Any = None,
            max_threads_per_model: int = 1,
            n_jobs: Optional[int] = None,
        ) -> TrainBundle:
            return TrainBundle(results=[], training_data=list(training_data))

        def load_predictor(self, model_path: Path) -> Any:
            raise NotImplementedError

        def suggest_hyperparameters(self, trial: Any) -> Dict[str, Any]:
            raise NotImplementedError

        def eval_trial(
            self,
            data: TrainingData,
            *,
            seed: int,
            max_threads_per_model: int = 1,
            n_cv_folds: int = 3,
        ) -> Tuple[float, float]:
            raise NotImplementedError

    with pytest.raises(ValueError, match="not supported"):
        require_tune_support(StubAdapter())


def test_stub_second_adapter_registers_without_cli_branch() -> None:
    """Adding a method is a class + registry entry (no CLI if-ladder)."""

    class EchoAdapter:
        name = "echo"
        supports_tune = False

        def train(
            self,
            training_data: Sequence[TrainingData],
            *,
            seed: int,
            fold_params: Any = None,
            max_threads_per_model: int = 1,
            n_jobs: Optional[int] = None,
        ) -> TrainBundle:
            results: List[TrainingResults] = [
                TrainingResults(
                    latin_name=d.latin_name,
                    activity_type=d.activity_type,
                    success=True,
                )
                for d in training_data
            ]
            return TrainBundle(results=results, training_data=list(training_data))

        def load_predictor(self, model_path: Path) -> Any:
            return object()

        def suggest_hyperparameters(self, trial: Any) -> Dict[str, Any]:
            raise NotImplementedError

        def eval_trial(
            self,
            data: TrainingData,
            *,
            seed: int,
            max_threads_per_model: int = 1,
            n_cv_folds: int = 3,
        ) -> Tuple[float, float]:
            raise NotImplementedError

    previous = ADAPTERS.get("echo")
    try:
        ADAPTERS["echo"] = EchoAdapter()
        adapter = get_adapter("echo")
        assert adapter.name == "echo"
        bundle = adapter.train([], seed=0)
        assert bundle.results == []
        with pytest.raises(ValueError, match="not supported"):
            require_tune_support(adapter)
    finally:
        if previous is None:
            ADAPTERS.pop("echo", None)
        else:
            ADAPTERS["echo"] = previous


def test_save_models_writes_trainer_maxent(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    from sdm.commands.modelling.train_sdm_models import save_models
    from sdm.models.maxent.maxent_model import DefaultMaxentConfig

    occurrence = gpd.GeoDataFrame(
        {
            "class": [1, 0, 1, 0],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        crs="EPSG:27700",
    )
    data = TrainingData(
        latin_name="Test species",
        activity_type="Roost",
        occurrence=occurrence,
        maxent_config=DefaultMaxentConfig(),
        model_features=["ev1"],
    )
    result = TrainingResults(
        latin_name="Test species",
        activity_type="Roost",
        final_model={"stub": True},
        cv_scores=None,
        success=True,
    )
    paths = save_models([result], [data], tmp_path, trainer="maxent")
    assert paths
    pkg = json.loads(
        (next(iter(paths.values())).parent / "package.json").read_text(encoding="utf-8")
    )
    assert pkg["trainer"] == "maxent"
