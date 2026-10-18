"""Storage-agnostic I/O via fsspec (local paths, ``gs://``, in-memory tests)."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fsspec

logger = logging.getLogger(__name__)


def join_uri_path(base: str | Path, *parts: str) -> str:
    base_str = str(base).rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    if not suffix:
        return base_str
    if "://" in base_str:
        return f"{base_str}/{suffix}"
    return str(Path(base_str) / suffix)


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


def uri_exists(uri_or_path: str | Path) -> bool:
    ref = str(uri_or_path)
    if "://" not in ref:
        return Path(ref).exists()
    fs, path = filesystem_for_uri(ref)
    return bool(fs.exists(path))


def uri_for_rasterio(uri_or_path: str | Path) -> str:
    """Map ``gs://`` URIs to GDAL ``/vsigs/`` paths for rasterio/rioxarray."""
    ref = str(uri_or_path)
    if ref.startswith("gs://"):
        return f"/vsigs/{ref[len('gs://'):]}"
    return ref


def resolve_under_env_root(config_or_cli_path: str | Path, env_var: str) -> str:
    """Return ``gs://``/absolute paths unchanged; otherwise prefix with ``env_var`` when set."""
    path_str = str(config_or_cli_path)
    if "://" in path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    root = os.environ.get(env_var, "").strip()
    if not root:
        return path_str
    if "://" in root:
        return join_uri_path(root, path_str)
    return join_uri_path(os.path.expanduser(root), path_str)


def ensure_directory(uri: str) -> None:
    fs, path = filesystem_for_uri(uri)
    fs.makedirs(path, exist_ok=True)


def read_bytes(uri_or_path: str | Path) -> bytes:
    ref = str(uri_or_path)
    if "://" not in ref:
        return Path(ref).read_bytes()
    fs, path = filesystem_for_uri(ref)
    return fs.cat_file(path) if hasattr(fs, "cat_file") else fs.cat(path)  # type: ignore[arg-type]


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


def _write_bytes_idempotent_local(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        return
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _remote_partial_path(remote_path: str) -> str:
    return f"{remote_path}.partial-{secrets.token_hex(8)}"


def _write_bytes_idempotent_remote(
    fs: fsspec.AbstractFileSystem, remote_path: str, data: bytes
) -> None:
    parent = remote_path.rsplit("/", 1)[0]
    if parent:
        fs.makedirs(parent, exist_ok=True)
    if fs.exists(remote_path):
        existing = (
            fs.cat_file(remote_path)
            if hasattr(fs, "cat_file")
            else fs.cat(remote_path)
        )
        if existing == data:
            return
    partial = _remote_partial_path(remote_path)
    try:
        with fs.open(partial, "wb") as handle:
            handle.write(data)
        if fs.exists(remote_path):
            fs.rm(remote_path)
        if hasattr(fs, "mv"):
            fs.mv(partial, remote_path)
        else:
            fs.put(partial, remote_path)
            fs.rm(partial)
    finally:
        if fs.exists(partial):
            fs.rm(partial)


def write_bytes_idempotent(uri_or_path: str | Path, data: bytes) -> None:
    """Write bytes once; skip when the target already has identical content."""
    ref = str(uri_or_path)
    if "://" not in ref:
        _write_bytes_idempotent_local(Path(ref), data)
        return
    fs, remote_path = filesystem_for_uri(ref)
    if _protocol_name(fs) == "file":
        _write_bytes_idempotent_local(Path(remote_path), data)
        return
    _write_bytes_idempotent_remote(fs, remote_path, data)


def write_text_idempotent(
    uri_or_path: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    write_bytes_idempotent(uri_or_path, text.encode(encoding))


def put_local_file_idempotent(
    fs: fsspec.AbstractFileSystem, local_path: Path, remote_path: str
) -> None:
    data = local_path.read_bytes()
    if _protocol_name(fs) == "file":
        _write_bytes_idempotent_local(Path(remote_path), data)
        return
    _write_bytes_idempotent_remote(fs, remote_path, data)


def put_local_tree(
    fs: fsspec.AbstractFileSystem, local_dir: Path, remote_dir: str
) -> None:
    remote_base = remote_dir.rstrip("/")
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        remote_path = f"{remote_base}/{rel}"
        parent = remote_path.rsplit("/", 1)[0]
        if parent and parent != remote_base:
            fs.makedirs(parent, exist_ok=True)
        put_local_file_idempotent(fs, file_path, remote_path)


@contextmanager
def staged_directory_context(target_uri: str) -> Iterator[tuple[Path, str]]:
    """
    Yield a local directory for writers; upload to ``target_uri`` when remote.

    Returns ``(write_dir, target_uri)``. Remote uploads use idempotent per-file writes.
    """
    ensure_directory(target_uri)
    if is_local_uri(target_uri):
        _fs, path = filesystem_for_uri(target_uri)
        write_dir = Path(path)
        write_dir.mkdir(parents=True, exist_ok=True)
        yield write_dir, target_uri
        return

    fs, remote_path = filesystem_for_uri(target_uri)
    with tempfile.TemporaryDirectory(prefix="sdm-staged-") as tmp:
        staging = Path(tmp)
        yield staging, target_uri
        put_local_tree(fs, staging, remote_path)
        logger.info("Published staged outputs to %s", target_uri)
