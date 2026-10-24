"""MLflow experiment tracking helpers for SDM training and tuning.

Design (#93): no tracking server. Each computer talks to a shared Neon
Postgres backend for run metadata, and stores plots / model files under a
dedicated private GCS prefix. The database password comes from Google Secret
Manager (or ``MLFLOW_DB_PASSWORD`` for cloud agents), never from git.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
from urllib.parse import quote_plus, unquote_plus

import mlflow
import numpy as np
import pandas as pd
import yaml
from google.cloud import secretmanager
from mlflow.sklearn import log_model
from pydantic import BaseModel, Field

from sdm.types import MlflowConfig, ProjectConfig

if TYPE_CHECKING:
    from sdm.types import TrainingData, TrainingResults

logger = logging.getLogger(__name__)

_ENV_ENABLED = "MLFLOW_ENABLED"
_ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
_ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
_ENV_DB_PASSWORD = "MLFLOW_DB_PASSWORD"
_ENV_ARTIFACT_ROOT = "MLFLOW_ARTIFACT_ROOT"

DEFAULT_SETTINGS_FILE = "mlflow.settings.yml"

# Redact userinfo passwords in URIs that may appear in driver/SQLAlchemy errors.
_URI_PASSWORD_RE = re.compile(r"(://[^:/@\s]+):([^@/\s]+)@")


def redact_secrets(text: str, *extra_secrets: str) -> str:
    """Return ``text`` with URI passwords and any known secrets replaced by ``***``."""
    redacted = _URI_PASSWORD_RE.sub(r"\1:***@", text)
    for secret in extra_secrets:
        if secret:
            redacted = redacted.replace(secret, "***")
            # URL-encoded form may appear in connection strings.
            encoded = quote_plus(secret)
            if encoded != secret:
                redacted = redacted.replace(encoded, "***")
    return redacted


def _password_from_tracking_uri(uri: Optional[str]) -> Optional[str]:
    """Extract the userinfo password from a SQLAlchemy / Neon tracking URI."""
    if not uri:
        return None
    match = _URI_PASSWORD_RE.search(uri)
    if match is None:
        return None
    return unquote_plus(match.group(2))


class SecretManagerRef(BaseModel):
    project_id: str = ""
    secret_id: str = ""
    version: str = "latest"


class PasswordConfig(BaseModel):
    secret_manager: SecretManagerRef = Field(default_factory=SecretManagerRef)


class DatabaseConfig(BaseModel):
    host: str = ""
    port: int = 5432
    name: str = ""
    user: str = ""
    sslmode: str = "require"


class MlflowSettings(BaseModel):
    """Local settings loaded from the gitignored settings file."""

    enabled: bool = True
    experiment_name: str = "species_sdm_experiment"
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    password: PasswordConfig = Field(default_factory=PasswordConfig)
    artifact_root: str = ""


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def load_mlflow_settings(path: Union[str, Path]) -> MlflowSettings:
    """Load MLflow settings YAML (no secrets). Missing file → empty defaults."""
    settings_path = Path(path)
    if not settings_path.is_file():
        logger.debug("MLflow settings file not found at %s", settings_path)
        return MlflowSettings(enabled=False)
    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return MlflowSettings.model_validate(raw)


def _access_secret_manager_version(
    *, project_id: str, secret_id: str, version: str
) -> str:
    """Fetch a secret payload using Application Default Credentials."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()


def fetch_database_password(settings: MlflowSettings) -> str:
    """Resolve the tracking DB password.

    Precedence:
    1. ``MLFLOW_DB_PASSWORD`` (Cursor / cloud-agent secrets; never commit).
    2. Google Secret Manager via the computer's existing Google credentials.
    """
    env_password = os.environ.get(_ENV_DB_PASSWORD)
    if env_password:
        return env_password.strip()

    ref = settings.password.secret_manager
    if not ref.project_id or not ref.secret_id:
        raise ValueError(
            "MLflow database password not available: set "
            f"{_ENV_DB_PASSWORD} or configure password.secret_manager "
            "in the local settings file."
        )
    return _access_secret_manager_version(
        project_id=ref.project_id,
        secret_id=ref.secret_id,
        version=ref.version or "latest",
    )


def build_tracking_uri(settings: MlflowSettings, password: Optional[str] = None) -> str:
    """Build a SQLAlchemy Postgres tracking URI for Neon (no tracking server)."""
    db = settings.database
    if not db.host or not db.name or not db.user:
        raise ValueError(
            "MLflow database host, name, and user must be set in the local settings file."
        )
    if password is None:
        password = fetch_database_password(settings)
    user = quote_plus(db.user)
    pwd = quote_plus(password)
    # psycopg3 dialect (package ``psycopg``); Neon requires TLS via sslmode.
    return (
        f"postgresql+psycopg://{user}:{pwd}@{db.host}:{db.port}/{db.name}"
        f"?sslmode={quote_plus(db.sslmode)}"
    )


def resolve_mlflow_config(
    project_config: ProjectConfig,
    *,
    repo_root: Optional[Path] = None,
) -> MlflowConfig:
    """Merge ``config.yml``, local settings file, and environment overrides.

    The database password is fetched only when tracking is enabled and no
    tracking URI was supplied via the environment. With tracking off, Secret
    Manager is never contacted.
    """
    cfg = project_config.mlflow.model_copy(deep=True)
    root = repo_root or Path.cwd()
    settings_path = Path(cfg.settings_file)
    if not settings_path.is_absolute():
        settings_path = root / settings_path

    settings = load_mlflow_settings(settings_path)
    if settings_path.is_file():
        cfg.enabled = settings.enabled and cfg.enabled
        if settings.experiment_name:
            cfg.experiment_name = settings.experiment_name
        if settings.artifact_root:
            cfg.artifact_root = settings.artifact_root

    # Resolve enabled / non-secret overrides before any password fetch.
    if _ENV_ENABLED in os.environ:
        cfg.enabled = _parse_bool(os.environ[_ENV_ENABLED])

    if _ENV_EXPERIMENT_NAME in os.environ:
        cfg.experiment_name = os.environ[_ENV_EXPERIMENT_NAME].strip()

    if _ENV_ARTIFACT_ROOT in os.environ:
        cfg.artifact_root = os.environ[_ENV_ARTIFACT_ROOT].strip()

    if _ENV_TRACKING_URI in os.environ:
        cfg.tracking_uri = os.environ[_ENV_TRACKING_URI].strip()
    elif cfg.enabled and settings_path.is_file() and not cfg.tracking_uri:
        try:
            cfg.tracking_uri = build_tracking_uri(settings)
        except ValueError as exc:
            logger.warning(
                "MLflow settings incomplete (%s); tracking stays off.",
                redact_secrets(str(exc)),
            )
            cfg.enabled = False

    return cfg


def _ensure_experiment(name: str, artifact_root: Optional[str]) -> None:
    """Create or select an experiment, pinning artifact_location when provided."""
    existing = mlflow.get_experiment_by_name(name)
    if existing is None:
        if artifact_root:
            mlflow.create_experiment(name, artifact_location=artifact_root)
        else:
            mlflow.create_experiment(name)
    mlflow.set_experiment(name)


def configure_mlflow(config: MlflowConfig) -> bool:
    """Configure the MLflow client. Returns False if tracking is disabled or unavailable.

    Never logs the tracking URI (it may embed the database password).
    """
    if not config.enabled:
        logger.info("MLflow tracking is disabled (config or %s).", _ENV_ENABLED)
        return False

    if not config.tracking_uri:
        logger.info("MLflow tracking URI not configured; skipping experiment logging.")
        return False

    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        _ensure_experiment(config.experiment_name, config.artifact_root)
        logger.debug(
            "MLflow client configured; experiment=%s; artifact_root=%s",
            config.experiment_name,
            config.artifact_root or "(default)",
        )
        return True
    except Exception as exc:
        uri_password = _password_from_tracking_uri(config.tracking_uri)
        safe_exc = (
            redact_secrets(str(exc), uri_password)
            if uri_password
            else redact_secrets(str(exc))
        )
        logger.warning(
            "MLflow tracking unavailable (%s); continuing without experiment logging.",
            safe_exc,
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
            ).strip()
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
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
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
