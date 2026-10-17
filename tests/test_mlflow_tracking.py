"""Tests for MLflow experiment tracking helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import geopandas as gpd
import mlflow
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sdm.types import (
    MlflowConfig,
    PathsConfig,
    ProjectConfig,
    SpatialConfig,
    TrainingData,
    TrainingResults,
)
from sdm.utils.mlflow_tracking import (
    TrainingSessionMetadata,
    configure_mlflow,
    log_models_to_mlflow,
    resolve_mlflow_config,
)


def _sample_project_config() -> ProjectConfig:
    return ProjectConfig(
        paths=PathsConfig(
            raw_data="data/raw",
            processed_data="data/processed",
            models="data/sdm_models",
            predictions="data/sdm_predictions",
            model_config="model_config.yml",
            variables_config="variables_config.yml",
            tuning_dir="data/sdm_tuning",
            config_dir="data/sdm_config",
            occurence_data="data/processed/bats-tidy.geojson",
            background_points="data/processed/background-points.geojson",
            boundary="data/processed/boundary.geojson",
            grid_points="data/evs/grid-points.parquet",
            evs="data/evs",
            ev_tiff="data/evs/evs-to-model.tif",
        ),
        spatial=SpatialConfig(
            top=646900.0,
            left=313000.0,
            crs="EPSG:27700",
            resolution=100,
            study_area_buffer=7000.0,
        ),
        crs="EPSG:27700",
        mlflow=MlflowConfig(
            enabled=True,
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="unit_test_experiment",
        ),
    )


class TestResolveMlflowConfig:
    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./custom_mlruns")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from_env")

        cfg = resolve_mlflow_config(_sample_project_config())

        assert cfg.enabled is False
        assert cfg.tracking_uri == "file:./custom_mlruns"
        assert cfg.experiment_name == "from_env"


class TestConfigureMlflow:
    def test_disabled_skips_client_setup(self) -> None:
        assert configure_mlflow(MlflowConfig(enabled=False)) is False

    def test_sqlite_store_succeeds(self, tmp_path) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        ok = configure_mlflow(
            MlflowConfig(
                enabled=True,
                tracking_uri=tracking_uri,
                experiment_name="configure_test",
            )
        )
        assert ok is True


class TestLogModelsToMlflow:
    @pytest.fixture
    def tracking_uri(self, tmp_path):
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("log_models_test")
        return uri

    def test_logs_metrics_params_and_model(self, tracking_uri, tmp_path) -> None:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression()),
            ]
        )
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = np.array([1, 0, 1, 0])
        pipeline.fit(X, y)

        gdf = gpd.GeoDataFrame(
            {
                "class": [1, 0],
                "sample_weight": [1.0, 1.0],
                "ev1": [1.0, 2.0],
                "ev2": [3.0, 4.0],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:27700",
        )
        training_data = [
            TrainingData(
                latin_name="Myotis daubentonii",
                activity_type="In flight",
                occurrence=gdf,
                maxent_config=MagicMock(),
                model_features=["ev1", "ev2"],
            )
        ]
        results = [
            TrainingResults(
                latin_name="Myotis daubentonii",
                activity_type="In flight",
                final_model=pipeline,
                cv_models=None,
                cv_scores=np.array([0.81, 0.79, 0.83]),
                success=True,
            )
        ]
        results_df = pd.DataFrame(
            [
                {
                    "identifier": "Myotis daubentonii_In flight",
                    "mean_cv_score": 0.81,
                    "std_cv_score": 0.02,
                }
            ]
        )

        occurrence_path = tmp_path / "occurrence.geojson"
        occurrence_path.write_text("{}", encoding="utf-8")

        session = TrainingSessionMetadata(
            occurrence_data=str(occurrence_path),
            ev_data="data/evs/evs-to-model.tif",
            model_config_path="model_config.yml",
            model_run_uri="data/sdm_models/runs/test-run",
        )

        log_models_to_mlflow(
            results,
            training_data,
            results_df,
            session_metadata=session,
            maxent_config_to_dict=lambda _cfg: {"beta_multiplier": 1.0},
        )

        experiment = mlflow.get_experiment_by_name("log_models_test")
        assert experiment is not None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 2

        nested = runs[runs["tags.mlflow.parentRunId"].notna()]
        assert len(nested) >= 1
        assert nested["metrics.mean_auc"].notna().any()
        assert nested["params.latin_name"].iloc[0] == "Myotis daubentonii"


class TestTrainingPipelineSkipsMlflowWhenDisabled:
    def test_configure_false_does_not_require_mlflow_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        cfg = resolve_mlflow_config(_sample_project_config())
        assert configure_mlflow(cfg) is False
