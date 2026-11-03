"""Model training run output paths under ``runs/<UTC>-<id>/`` (local or ``gs://``)."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from sdm.utils.fs import join_uri_path, resolve_under_env_root, staged_directory_context

MODEL_OUTPUT_ROOT_ENV = "MODEL_OUTPUT_ROOT"
INPUT_DATA_ROOT_ENV = "INPUT_DATA_ROOT"
RUNS_DIR_NAME = "runs"
MODEL_INDEX_FILENAME = "model_results.csv"

logger = logging.getLogger(__name__)


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


def run_segment_from_uri(run_uri: str) -> str:
    """Return the ``runs/<segment>`` directory name from a run URI or path."""
    cleaned = run_uri.rstrip("/")
    if "://" in cleaned:
        segment = cleaned.rsplit("/", 1)[-1]
    else:
        segment = Path(cleaned).name
    if not segment or segment == RUNS_DIR_NAME:
        raise ValueError(f"Could not parse run id from {run_uri!r}")
    return segment


def list_local_run_ids(models_root: Path) -> list[str]:
    """Sorted run directory names under ``models_root/runs/`` (lexicographic)."""
    runs_dir = Path(models_root) / RUNS_DIR_NAME
    if not runs_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in runs_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def resolve_run_prefix(run_ids: list[str], prefix: str) -> str:
    """Match *prefix* to exactly one run id; raise if none or ambiguous."""
    matches = [run_id for run_id in run_ids if run_id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(run_ids) if run_ids else "(none)"
    if not matches:
        raise FileNotFoundError(
            f"No run id starts with {prefix!r}. Available runs: {available}"
        )
    matched = ", ".join(matches)
    raise ValueError(
        f"Run prefix {prefix!r} matches more than one run: {matched}. "
        f"Available runs: {available}"
    )


def resolve_predict_models_dir(
    models_dir: Path | str,
    *,
    run: str | None = None,
) -> Path:
    """Resolve the directory that holds ``model_results.csv`` for predict.

    - If *models_dir* already contains ``model_results.csv``, use it (explicit
      run folder or legacy flat layout). ``--run`` is not allowed in that case.
    - Else treat *models_dir* as a models root and select under ``runs/``:
      newest when *run* is None, or the unique prefix match when *run* is set.
    """
    root = Path(models_dir)
    index_path = root / MODEL_INDEX_FILENAME
    if index_path.is_file():
        if run is not None:
            raise ValueError(
                f"--run cannot be combined with a models directory that already "
                f"contains {MODEL_INDEX_FILENAME} ({root}). "
                "Omit --run, or pass the models root and --run <id>."
            )
        return root

    run_ids = list_local_run_ids(root)
    runs_dir = root / RUNS_DIR_NAME

    if run is not None:
        if not run_ids:
            raise FileNotFoundError(
                f"No trained model runs found under {runs_dir}. "
                "Run `sdm train` first, or pass --models-dir to a folder with "
                f"{MODEL_INDEX_FILENAME}."
            )
        chosen = resolve_run_prefix(run_ids, run)
        resolved = runs_dir / chosen
        logger.info("Using model run %s (%s)", chosen, resolved)
        return resolved

    if run_ids:
        chosen = run_ids[-1]
        resolved = runs_dir / chosen
        logger.info("Using newest model run %s (%s)", chosen, resolved)
        return resolved

    flat_pkls = sorted(root.glob("*.pkl"))
    if flat_pkls:
        raise FileNotFoundError(
            f"{root} has .pkl files but no {MODEL_INDEX_FILENAME}. "
            "Pass --models-dir to a training run folder "
            f"({RUNS_DIR_NAME}/<run_id>) that contains {MODEL_INDEX_FILENAME}, "
            "or re-train so a run is written under "
            f"{runs_dir}/."
        )

    raise FileNotFoundError(
        f"No trained model runs found under {runs_dir}. "
        "Run `sdm train` first, pass --run <run_id>, or pass --models-dir "
        f"to a folder with {MODEL_INDEX_FILENAME}."
    )


model_run_write_context = staged_directory_context
