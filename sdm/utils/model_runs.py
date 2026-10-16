"""Model training run output paths under ``runs/<UTC>-<id>/`` (local or ``gs://``)."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from sdm.utils.fs import join_uri_path, resolve_under_env_root, staged_directory_context

MODEL_OUTPUT_ROOT_ENV = "MODEL_OUTPUT_ROOT"
INPUT_DATA_ROOT_ENV = "INPUT_DATA_ROOT"
RUNS_DIR_NAME = "runs"
DEFAULT_GCS_BUCKET = "gs://habitat-suitability-modelling-data"
DEFAULT_INPUTS_PREFIX = f"{DEFAULT_GCS_BUCKET}/inputs"
DEFAULT_RUNS_PREFIX = f"{DEFAULT_GCS_BUCKET}/runs"


def resolve_data_path(config_or_cli_path: str | Path) -> str:
    return resolve_under_env_root(config_or_cli_path, INPUT_DATA_ROOT_ENV)


def resolve_output_root(
    *,
    cli_output_dir: Path | None,
    config_models_path: str | Path,
) -> str:
    env_root = os.environ.get(MODEL_OUTPUT_ROOT_ENV, "").strip()
    if env_root:
        return env_root.rstrip("/")
    if cli_output_dir is not None:
        return str(cli_output_dir)
    return str(config_models_path)


def generate_run_segment(
    *,
    timestamp: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    moment = timestamp or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    suffix = random_suffix or secrets.token_hex(4)
    return f"{stamp}-{suffix}"


def build_run_output_uri(
    output_root: str,
    *,
    run_segment: str | None = None,
) -> str:
    root = output_root.rstrip("/")
    segment = run_segment or generate_run_segment()
    if "://" in root:
        return f"{root}/{RUNS_DIR_NAME}/{segment}/"
    return str(Path(root) / RUNS_DIR_NAME / segment)


def join_run_artifact(run_uri: str, relative_path: str | Path) -> str:
    rel = Path(relative_path).as_posix().lstrip("/")
    return join_uri_path(run_uri, rel)


def artifact_parent_uri(artifact_uri: str) -> str:
    if "://" in artifact_uri:
        head, _sep, _tail = artifact_uri.rpartition("/")
        return f"{head}/" if head else artifact_uri
    return str(Path(artifact_uri).parent)


model_run_write_context = staged_directory_context
