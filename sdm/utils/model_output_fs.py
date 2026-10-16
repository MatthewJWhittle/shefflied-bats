"""fsspec-backed model run output locations (local paths or ``gs://`` URIs)."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import fsspec

logger = logging.getLogger(__name__)

MODEL_OUTPUT_ROOT_ENV = "MODEL_OUTPUT_ROOT"
INPUT_DATA_ROOT_ENV = "INPUT_DATA_ROOT"
RUNS_DIR_NAME = "runs"
DEFAULT_GCS_BUCKET = "gs://habitat-suitability-modelling-data"
DEFAULT_INPUTS_PREFIX = f"{DEFAULT_GCS_BUCKET}/inputs"
DEFAULT_RUNS_PREFIX = f"{DEFAULT_GCS_BUCKET}/runs"


def join_uri_path(base: str | Path, *parts: str) -> str:
    base_str = str(base).rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    if not suffix:
        return base_str
    if "://" in base_str:
        return f"{base_str}/{suffix}"
    return str(Path(base_str) / suffix)


def uri_exists(uri_or_path: str | Path) -> bool:
    ref = str(uri_or_path)
    if "://" not in ref:
        return Path(ref).exists()
    fs, path = filesystem_for_uri(ref)
    return bool(fs.exists(path))


def resolve_data_path(config_or_cli_path: str | Path) -> str:
    """Resolve a config path, optionally under ``INPUT_DATA_ROOT`` or pass-through ``gs://``."""
    path_str = str(config_or_cli_path)
    if "://" in path_str:
        return path_str
    root = os.environ.get(INPUT_DATA_ROOT_ENV, "").strip()
    if root:
        return f"{root.rstrip('/')}/{path_str.lstrip('/')}"
    return path_str


def uri_for_rasterio(uri_or_path: str | Path) -> str:
    """Map ``gs://`` URIs to GDAL ``/vsigs/`` paths for rasterio/rioxarray."""
    ref = str(uri_or_path)
    if ref.startswith("gs://"):
        return f"/vsigs/{ref[len('gs://'):]}"
    return ref


def join_run_artifact(run_uri: str, relative_path: str | Path) -> str:
    rel = Path(relative_path).as_posix().lstrip("/")
    if "://" in run_uri:
        return f"{run_uri.rstrip('/')}/{rel}"
    return str(Path(run_uri.rstrip("/")) / rel)


def artifact_parent_uri(artifact_uri: str) -> str:
    if "://" in artifact_uri:
        head, _sep, _tail = artifact_uri.rpartition("/")
        return f"{head}/" if head else artifact_uri
    return str(Path(artifact_uri).parent)


def resolve_output_root(
    *,
    cli_output_dir: Path | None,
    config_models_path: str | Path,
) -> str:
    """Return the configured model output root (directory or ``gs://`` bucket prefix)."""
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


def filesystem_for_uri(uri: str) -> tuple[fsspec.AbstractFileSystem, str]:
    return fsspec.core.url_to_fs(uri.rstrip("/"))


def _protocol_name(fs: fsspec.AbstractFileSystem) -> str:
    protocol = fs.protocol
    if isinstance(protocol, (tuple, list)):
        return str(protocol[0])
    return str(protocol)


def is_local_uri(uri: str) -> bool:
    fs, _ = filesystem_for_uri(uri)
    return _protocol_name(fs) == "file"


def ensure_run_directory(run_uri: str) -> None:
    fs, path = filesystem_for_uri(run_uri)
    fs.makedirs(path, exist_ok=True)


@contextmanager
def local_materialized_path(uri_or_path: str | Path) -> Iterator[Path]:
    """Yield a local ``Path`` (download remote objects to a temporary file first)."""
    ref = str(uri_or_path)
    if "://" not in ref:
        yield Path(ref)
        return

    fs, remote_path = filesystem_for_uri(ref)
    if _protocol_name(fs) == "file":
        yield Path(remote_path)
        return

    suffix = Path(remote_path).suffix or ".dat"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
    try:
        fs.get(remote_path, str(temp_path))
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def put_local_tree(fs: fsspec.AbstractFileSystem, local_dir: Path, remote_dir: str) -> None:
    remote_base = remote_dir.rstrip("/")
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        remote_path = f"{remote_base}/{rel}"
        parent = remote_path.rsplit("/", 1)[0]
        if parent and parent != remote_base:
            fs.makedirs(parent, exist_ok=True)
        fs.put(str(file_path), remote_path)


@contextmanager
def model_run_write_context(
    run_uri: str,
) -> Iterator[tuple[Path, str]]:
    """
    Yield a local ``Path`` for writers and upload to ``run_uri`` when remote.

    Returns ``(write_dir, run_uri)``.
    """
    ensure_run_directory(run_uri)
    if is_local_uri(run_uri):
        _fs, path = filesystem_for_uri(run_uri)
        write_dir = Path(path)
        write_dir.mkdir(parents=True, exist_ok=True)
        yield write_dir, run_uri
        return

    fs, remote_path = filesystem_for_uri(run_uri)
    with tempfile.TemporaryDirectory(prefix="sdm-model-run-") as tmp:
        staging = Path(tmp)
        yield staging, run_uri
        put_local_tree(fs, staging, remote_path)
        logger.info("Uploaded model run outputs to %s", run_uri)
