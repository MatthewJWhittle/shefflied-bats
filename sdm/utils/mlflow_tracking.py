"""MLflow experiment tracking helpers for SDM training and tuning."""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

import mlflow
from mlflow.sklearn import log_model

from sdm.types import MlflowConfig, ProjectConfig

if TYPE_CHECKING:
    from sdm.types import TrainingData, TrainingResults

logger = logging.getLogger(__name__)

_ENV_ENABLED = "MLFLOW_ENABLED"
_ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
_ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_mlflow_config(project_config: ProjectConfig) -> MlflowConfig:
    """Merge ``config.yml`` MLflow settings with environment overrides."""
    cfg = project_config.mlflow.model_copy(deep=True)

    if _ENV_ENABLED in os.environ:
        cfg.enabled = _parse_bool(os.environ[_ENV_ENABLED])

    if _ENV_TRACKING_URI in os.environ:
        cfg.tracking_uri = os.environ[_ENV_TRACKING_URI].strip()

    if _ENV_EXPERIMENT_NAME in os.environ:
        cfg.experiment_name = os.environ[_ENV_EXPERIMENT_NAME].strip()

    return cfg


def configure_mlflow(config: MlflowConfig) -> bool:
    """Configure the MLflow client. Returns False if tracking is disabled or unavailable."""
    if not config.enabled:
        logger.info("MLflow tracking is disabled (config or %s).", _ENV_ENABLED)
        return False

    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        logger.debug(
            "MLflow tracking URI: %s; experiment: %s",
            config.tracking_uri,
            config.experiment_name,
        )
        return True
    except Exception as exc:
        logger.warning(
            "MLflow tracking unavailable (%s); continuing without experiment logging.",
            exc,
        )
        return False


def get_git_commit() -> Optional[str]:
    """Return the current git commit SHA when running inside a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _path_version_hint(path: Optional[str]) -> Optional[str]:
    """Best-effort dataset version hint from local file mtime (ISO UTC)."""
    if not path:
        return None
    if path.startswith("gs://") or "://" in path:
        return None
    local = Path(path)
    if not local.is_file():
        return None
    mtime = datetime.fromtimestamp(local.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat()


@dataclass
class TrainingSessionMetadata:
    """Context logged once per training or tuning session."""

    occurrence_data: Optional[str] = None
    ev_data: Optional[str] = None
    model_config_path: Optional[str] = None
    variables_config_path: Optional[str] = None
    model_run_uri: Optional[str] = None
    extra_tags: Dict[str, str] = field(default_factory=dict)

    def as_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self.occurrence_data:
            params["occurrence_data"] = self.occurrence_data
            version = _path_version_hint(self.occurrence_data)
            if version:
                params["occurrence_data_version"] = version
        if self.ev_data:
            params["ev_data"] = self.ev_data
            version = _path_version_hint(self.ev_data)
            if version:
                params["ev_data_version"] = version
        if self.model_config_path:
            params["model_config_path"] = self.model_config_path
        if self.variables_config_path:
            params["variables_config_path"] = self.variables_config_path
        if self.model_run_uri:
            params["model_run_uri"] = self.model_run_uri
        return params

    def as_tags(self) -> Dict[str, str]:
        tags = dict(self.extra_tags)
        commit = get_git_commit()
        if commit:
            tags["git_commit"] = commit
            tags["mlflow.source.git.commit"] = commit
        return tags


def _summarize_cv_scores(cv_scores: Any) -> tuple[float, float, int]:
    """Local copy of CV summary for MLflow metrics (avoids circular imports)."""
    import math

    import numpy as np

    if cv_scores is None or len(cv_scores) == 0:
        return float("nan"), float("nan"), 0
    arr = np.asarray(cv_scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    mean = float(valid.mean())
    std = float(valid.std())
    if math.isnan(mean):
        return float("nan"), float("nan"), 0
    return mean, std, len(valid)


def log_models_to_mlflow(
    models: List[TrainingResults],
    training_data: List[TrainingData],
    results_df: Any,
    *,
    session_metadata: Optional[TrainingSessionMetadata] = None,
    maxent_config_to_dict: Optional[Any] = None,
) -> None:
    """
    Log trained models and summary results to MLflow (nested runs per species–activity).

    No-op when MLflow is disabled or the tracking backend is unreachable.
    """
    if not models:
        return

    data_by_id = {item.identifier(): item for item in training_data}
    active_run = mlflow.active_run()

    if active_run is None:
        parent_run_name = f"SDM_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_context: Any = mlflow.start_run(run_name=parent_run_name)
        logger.debug("Started MLflow parent run: %s", parent_run_name)
    else:
        logger.debug("Using existing MLflow run: %s", active_run.info.run_id)
        run_context = nullcontext()

    with run_context:
        if session_metadata is not None:
            for key, value in session_metadata.as_tags().items():
                mlflow.set_tag(key, value)
            safe_params = {
                k: str(v)[:500] for k, v in session_metadata.as_params().items()
            }
            if safe_params:
                mlflow.log_params(safe_params)

        mlflow.log_table(data=results_df, artifact_file="model_results.parquet")
        logger.debug("Logging %d models to MLflow...", len(models))

        for model in models:
            data = data_by_id.get(model.identifier())
            if data is None:
                logger.warning(
                    "Skipping MLflow logging for %s (no matching TrainingData)",
                    model.identifier(),
                )
                continue

            model_identifier = model.identifier()
            with mlflow.start_run(nested=True, run_name=model_identifier):
                logger.debug("Logging model: %s", model_identifier)

                mlflow.log_params(
                    {
                        "latin_name": model.latin_name,
                        "activity_type": model.activity_type,
                        "n_presence": len(
                            data.occurrence[data.occurrence["class"] == 1]
                        ),
                        "n_background": len(
                            data.occurrence[data.occurrence["class"] == 0]
                        ),
                    }
                )

                if maxent_config_to_dict is not None:
                    mlflow.log_dict(
                        {"features": list(data.model_features)},
                        "model_features.json",
                    )
                    mlflow.log_dict(
                        maxent_config_to_dict(data.maxent_config),
                        "maxent_config.json",
                    )

                if model.final_model is not None:
                    try:
                        model_params = model.final_model.get_params()
                        flat_params = {
                            k: str(v)[:250]
                            for k, v in model_params.items()
                            if not callable(v)
                        }
                        if flat_params:
                            mlflow.log_params(flat_params)
                    except Exception as exc:
                        logger.debug(
                            "Could not log sklearn params for %s: %s",
                            model_identifier,
                            exc,
                        )
                else:
                    logger.error("Model is None for %s", model_identifier)

                mlflow.set_tag("latin_name", model.latin_name)
                parts = model.latin_name.split(" ", 1)
                if len(parts) == 2:
                    genus, species = parts[0], parts[1]
                    mlflow.set_tag(
                        "species_code", f"{genus[:3]}_{species[:3]}"
                    )
                mlflow.set_tag("activity_type", model.activity_type)

                cv_mean, cv_std, n_valid = _summarize_cv_scores(model.cv_scores)
                if n_valid > 0:
                    mlflow.log_metric("mean_auc", cv_mean)
                    mlflow.log_metric("std_auc", cv_std)
                else:
                    logger.info(
                        "Skipping MLflow metrics for %s (no valid CV scores)",
                        model_identifier,
                    )

                if model.final_model is not None:
                    import pandas as pd

                    occurrence = data.occurrence
                    x_sample = occurrence.drop(
                        columns=["geometry", "class", "sample_weight"]
                    )
                    x_sample = x_sample.iloc[0]
                    input_example = pd.DataFrame(x_sample).T
                    artifact_path = f"{model_identifier}_final_model"
                    try:
                        model_info = log_model(
                            model.final_model,
                            name=artifact_path,
                            input_example=input_example,
                        )
                        logger.debug(
                            "Model logged to MLflow: %s", model_info.model_uri
                        )
                    except Exception as exc:
                        if "UNIQUE constraint" in str(
                            exc
                        ) or "duplicate" in str(exc).lower():
                            logger.warning(
                                "MLflow metric conflict for %s: %s",
                                model_identifier,
                                exc,
                            )
                        else:
                            logger.error(
                                "Failed to log model %s to MLflow: %s",
                                model_identifier,
                                exc,
                            )
                            raise

        active_run_after = mlflow.active_run()
        parent_run_id = (
            active_run_after.info.run_id if active_run_after else "N/A"
        )
        logger.debug(
            "MLflow logging complete - Run ID: %s, %d models logged",
            parent_run_id,
            len(models),
        )


@contextmanager
def mlflow_session_run(
    run_name: str,
    *,
    enabled: bool,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[None]:
    """Start a top-level MLflow run when tracking is enabled."""
    if not enabled:
        yield
        return

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(
                {k: str(v)[:500] for k, v in params.items()}
            )
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        yield


def log_tuning_session_tags(metadata: TrainingSessionMetadata) -> None:
    """Apply session tags and params to the active MLflow run (tuning parent run)."""
    for key, value in metadata.as_tags().items():
        mlflow.set_tag(key, value)
    safe_params = {k: str(v)[:500] for k, v in metadata.as_params().items()}
    if safe_params:
        mlflow.log_params(safe_params)
