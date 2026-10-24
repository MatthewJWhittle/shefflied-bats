"""Tests for shared MLflow run tracking (Neon backend + GCS artifacts design)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
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
    build_tracking_uri,
    configure_mlflow,
    fetch_database_password,
    load_mlflow_settings,
    log_models_to_mlflow,
    resolve_mlflow_config,
)


def _sample_project_config(**mlflow_overrides) -> ProjectConfig:
    mlflow_cfg = {
        "enabled": True,
        "settings_file": "mlflow.settings.yml",
        "experiment_name": "unit_test_experiment",
    }
    mlflow_cfg.update(mlflow_overrides)
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
        mlflow=MlflowConfig(**mlflow_cfg),
    )


def _write_settings(path: Path, **overrides) -> Path:
    data = {
        "enabled": True,
        "experiment_name": "settings_experiment",
        "database": {
            "host": "ep-example.eu-west-2.aws.neon.tech",
            "port": 5432,
            "name": "mlflow",
            "user": "mlflow_tracking",
            "sslmode": "require",
        },
        "password": {
            "secret_manager": {
                "project_id": "example-project",
                "secret_id": "mlflow-tracking-db-password",
                "version": "latest",
            }
        },
        "artifact_root": "gs://example-bucket/mlflow-artifacts",
    }
    data.update(overrides)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestLoadMlflowSettings:
    def test_loads_example_shaped_file(self, tmp_path: Path) -> None:
        settings_path = _write_settings(tmp_path / "mlflow.settings.yml")
        settings = load_mlflow_settings(settings_path)
        assert settings.enabled is True
        assert settings.database.host.endswith("neon.tech")
        assert settings.database.user == "mlflow_tracking"
        assert settings.artifact_root.startswith("gs://")
        assert "password" not in yaml.safe_load(settings_path.read_text())["database"]


class TestFetchDatabasePassword:
    def test_env_override_wins_for_cloud_agents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MLFLOW_DB_PASSWORD", "from-cursor-secret")
        settings = load_mlflow_settings(_write_settings(tmp_path / "s.yml"))
        assert fetch_database_password(settings) == "from-cursor-secret"

    def test_secret_manager_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MLFLOW_DB_PASSWORD", raising=False)
        settings = load_mlflow_settings(_write_settings(tmp_path / "s.yml"))
        with patch(
            "sdm.utils.mlflow_tracking._access_secret_manager_version",
            return_value="from-gsm",
        ) as mock_gsm:
            assert fetch_database_password(settings) == "from-gsm"
            mock_gsm.assert_called_once_with(
                project_id="example-project",
                secret_id="mlflow-tracking-db-password",
                version="latest",
            )


class TestBuildTrackingUri:
    def test_builds_postgres_uri_without_embedding_in_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MLFLOW_DB_PASSWORD", "s3cret")
        settings = load_mlflow_settings(_write_settings(tmp_path / "s.yml"))
        uri = build_tracking_uri(settings)
        assert uri.startswith("postgresql+psycopg://mlflow_tracking:")
        assert "s3cret" in uri
        assert "ep-example.eu-west-2.aws.neon.tech" in uri
        assert "sslmode=require" in uri
        # Real settings file on disk must not contain the password value.
        assert "s3cret" not in (tmp_path / "s.yml").read_text()


class TestResolveMlflowConfig:
    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from_env")
        cfg = resolve_mlflow_config(_sample_project_config())
        assert cfg.enabled is False
        assert cfg.tracking_uri == "sqlite:///custom.db"
        assert cfg.experiment_name == "from_env"


class TestConfigureMlflow:
    def test_disabled_skips_client_setup(self) -> None:
        assert configure_mlflow(MlflowConfig(enabled=False)) is False

    def test_sqlite_store_with_local_artifact_root(self, tmp_path: Path) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        artifact_root = str(tmp_path / "artifacts")
        ok = configure_mlflow(
            MlflowConfig(
                enabled=True,
                tracking_uri=tracking_uri,
                experiment_name="configure_test",
                artifact_root=artifact_root,
            )
        )
        assert ok is True
        experiment = mlflow.get_experiment_by_name("configure_test")
        assert experiment is not None
        assert artifact_root in (experiment.artifact_location or "")


class TestTwoWritersSameBackend:
    """Done-check: log a run from two places and read both back (#93)."""

    def test_logs_from_two_places_and_reads_both(self, tmp_path: Path) -> None:
        shared_db = tmp_path / "shared.db"
        tracking_uri = f"sqlite:///{shared_db}"
        artifact_root = str(tmp_path / "artifacts")
        experiment_name = "dual_writer_experiment"

        def _log_from(writer_id: str, auc: float) -> str:
            # Fresh client configuration, as if on a separate computer.
            mlflow.set_tracking_uri("")
            ok = configure_mlflow(
                MlflowConfig(
                    enabled=True,
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    artifact_root=artifact_root,
                )
            )
            assert ok is True
            with mlflow.start_run(run_name=f"run_{writer_id}") as run:
                mlflow.set_tag("writer", writer_id)
                mlflow.log_metric("mean_auc", auc)
                mlflow.log_param("latin_name", "Myotis daubentonii")
                marker = tmp_path / f"{writer_id}.txt"
                marker.write_text(writer_id, encoding="utf-8")
                mlflow.log_artifact(str(marker), artifact_path="plots")
                return run.info.run_id

        run_a = _log_from("computer_a", 0.91)
        run_b = _log_from("computer_b", 0.87)

        # Third client reads both back from the shared store.
        mlflow.set_tracking_uri("")
        configure_mlflow(
            MlflowConfig(
                enabled=True,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                artifact_root=artifact_root,
            )
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        writers = set(runs["tags.writer"].dropna())
        assert writers == {"computer_a", "computer_b"}
        assert {run_a, run_b} <= set(runs["run_id"])
        by_writer = {
            row["tags.writer"]: row["metrics.mean_auc"]
            for _, row in runs.iterrows()
            if pd.notna(row.get("tags.writer"))
        }
        assert by_writer["computer_a"] == pytest.approx(0.91)
        assert by_writer["computer_b"] == pytest.approx(0.87)


class TestLogModelsToMlflow:
    @pytest.fixture
    def tracking_uri(self, tmp_path: Path):
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        artifact_root = str(tmp_path / "artifacts")
        configure_mlflow(
            MlflowConfig(
                enabled=True,
                tracking_uri=uri,
                experiment_name="log_models_test",
                artifact_root=artifact_root,
            )
        )
        return uri

    def test_logs_metrics_params_and_model(self, tracking_uri, tmp_path: Path) -> None:
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
    def test_configure_false_does_not_require_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ENABLED", "false")
        cfg = resolve_mlflow_config(_sample_project_config())
        assert configure_mlflow(cfg) is False


class TestExampleSettingsCommitted:
    def test_example_file_has_empty_values_and_no_secrets(self) -> None:
        example = Path("mlflow.settings.example.yml")
        assert example.is_file()
        text = example.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        assert data["database"]["host"] in ("", None)
        assert data["database"]["user"] in ("", None)
        assert data["database"]["name"] in ("", None)
        assert "password" not in data.get("database", {})
        assert data["password"]["secret_manager"]["project_id"] in ("", None)
        assert data["artifact_root"] in ("", None)
        # No accidental credential-looking tokens.
        for banned in ("neon.tech", "AKIA", "postgresql://"):
            assert banned not in text
