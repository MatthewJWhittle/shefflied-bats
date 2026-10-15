"""
Upload a local model-run output directory to Google Cloud Storage.

Uses Application Default Credentials (optionally service-account impersonation),
creates a unique run prefix under ``runs/``, verifies uploads, then removes the
local directory on success.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

logger = logging.getLogger(__name__)

MODEL_OUTPUT_BUCKET_ENV = "MODEL_OUTPUT_BUCKET"
MODEL_RUN_IMPERSONATE_SA_ENV = "MODEL_RUN_IMPERSONATE_SA"

GCS_RW_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"


class UploadConflictError(RuntimeError):
    """An object already exists at the destination (if_generation_match=0 failed)."""


class UploadVerificationError(RuntimeError):
    """Uploaded objects do not match the local manifest."""


@dataclass(frozen=True)
class LocalObject:
    relative_path: str
    absolute_path: Path
    size: int
    md5_hash_b64: str


@dataclass(frozen=True)
class UploadResult:
    bucket_name: str
    run_prefix: str
    relative_paths: tuple[str, ...]
    project_id: str | None


class BlobLike(Protocol):
    name: str
    size: int
    md5_hash: str

    def upload_from_filename(
        self, filename: str, *, if_generation_match: int | None = None
    ) -> None: ...

    def reload(self) -> None: ...


class BucketLike(Protocol):
    def blob(self, name: str) -> BlobLike: ...

    def list_blobs(self, prefix: str = "") -> Iterable[BlobLike]: ...


def generate_run_prefix(
    *,
    timestamp: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    """Return ``runs/<UTC timestamp>-<short id>/`` (trailing slash included)."""
    moment = timestamp or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    suffix = random_suffix or secrets.token_hex(4)
    return f"runs/{stamp}-{suffix}/"


def md5_hash_b64_for_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def collect_local_objects(local_dir: Path) -> list[LocalObject]:
    root = local_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local output directory not found: {root}")

    objects: list[LocalObject] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        objects.append(
            LocalObject(
                relative_path=rel,
                absolute_path=path,
                size=size,
                md5_hash_b64=md5_hash_b64_for_file(path),
            )
        )
    return objects


def get_gcs_credentials_and_project() -> tuple[object, str | None]:
    import google.auth
    from google.auth import impersonated_credentials

    credentials, project = google.auth.default(scopes=[GCS_RW_SCOPE])
    impersonate_sa = os.environ.get(MODEL_RUN_IMPERSONATE_SA_ENV)
    if impersonate_sa:
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=impersonate_sa,
            target_scopes=[GCS_RW_SCOPE],
        )
    return credentials, project


def resolve_bucket_name(bucket_name: str | None) -> str:
    resolved = bucket_name or os.environ.get(MODEL_OUTPUT_BUCKET_ENV)
    if not resolved:
        raise ValueError(
            f"GCS bucket not configured; set {MODEL_OUTPUT_BUCKET_ENV} or pass bucket_name."
        )
    return resolved


def _upload_objects(
    bucket: BucketLike,
    run_prefix: str,
    local_objects: Sequence[LocalObject],
) -> None:
    for obj in local_objects:
        blob_name = f"{run_prefix}{obj.relative_path}"
        blob = bucket.blob(blob_name)
        try:
            blob.upload_from_filename(
                str(obj.absolute_path),
                if_generation_match=0,
            )
        except PreconditionFailed as exc:
            raise UploadConflictError(
                f"Refusing to overwrite existing object gs://{getattr(bucket, 'name', '?')}/{blob_name}"
            ) from exc


def _remote_manifest(
    bucket: BucketLike,
    run_prefix: str,
) -> dict[str, tuple[int, str]]:
    remote: dict[str, tuple[int, str]] = {}
    for blob in bucket.list_blobs(prefix=run_prefix):
        if not blob.name.startswith(run_prefix):
            continue
        relative = blob.name[len(run_prefix) :]
        if not relative:
            continue
        blob.reload()
        if blob.md5_hash is None:
            raise UploadVerificationError(
                f"Remote object missing MD5 hash: {blob.name}"
            )
        remote[relative] = (blob.size, blob.md5_hash)
    return remote


def verify_upload(
    bucket: BucketLike,
    run_prefix: str,
    local_objects: Sequence[LocalObject],
) -> None:
    remote = _remote_manifest(bucket, run_prefix)
    local_by_path = {obj.relative_path: obj for obj in local_objects}

    if set(remote.keys()) != set(local_by_path.keys()):
        raise UploadVerificationError(
            "Object count mismatch after upload "
            f"(local={len(local_by_path)}, remote={len(remote)})."
        )

    for rel, obj in local_by_path.items():
        remote_size, remote_md5 = remote[rel]
        if remote_size != obj.size:
            raise UploadVerificationError(
                f"Size mismatch for {rel}: local={obj.size}, remote={remote_size}"
            )
        if remote_md5 != obj.md5_hash_b64:
            raise UploadVerificationError(f"MD5 mismatch for {rel}")


def print_upload_summary(result: UploadResult) -> None:
    base_uri = f"gs://{result.bucket_name}/{result.run_prefix}"
    print(f"Run folder: {base_uri}")
    for rel in sorted(result.relative_paths):
        print(f"  gs://{result.bucket_name}/{result.run_prefix}{rel}")
    folder_path = f"{result.bucket_name}/{result.run_prefix.rstrip('/')}"
    console_url = (
        f"https://console.cloud.google.com/storage/browser/{folder_path}"
    )
    if result.project_id:
        console_url = f"{console_url}?project={result.project_id}"
    print(f"Console: {console_url}")


def upload_model_run_outputs(
    local_dir: Path,
    *,
    bucket_name: str | None = None,
    run_prefix: str | None = None,
    delete_local: bool = True,
    storage_client: storage.Client | None = None,
    bucket_handle: BucketLike | None = None,
) -> UploadResult:
    """
    Upload ``local_dir`` to GCS under a unique run prefix, verify, optionally delete local.

    ``bucket_handle`` supports tests; when omitted, ``storage_client`` or ADC is used.
    """
    resolved_bucket = resolve_bucket_name(bucket_name)
    local_objects = collect_local_objects(local_dir)
    prefix = run_prefix or generate_run_prefix()

    if bucket_handle is None:
        if storage_client is None:
            credentials, project_id = get_gcs_credentials_and_project()
            storage_client = storage.Client(credentials=credentials, project=project_id)
        else:
            _, project_id = get_gcs_credentials_and_project()
        bucket_handle = storage_client.bucket(resolved_bucket)
    else:
        project_id = None
        if storage_client is not None:
            project_id = storage_client.project

    logger.info(
        "Uploading %d file(s) from %s to gs://%s/%s",
        len(local_objects),
        local_dir,
        resolved_bucket,
        prefix,
    )

    _upload_objects(bucket_handle, prefix, local_objects)
    verify_upload(bucket_handle, prefix, local_objects)

    relative_paths = tuple(obj.relative_path for obj in local_objects)
    result = UploadResult(
        bucket_name=resolved_bucket,
        run_prefix=prefix,
        relative_paths=relative_paths,
        project_id=project_id,
    )
    print_upload_summary(result)

    if delete_local:
        shutil.rmtree(local_dir.resolve())
        logger.info("Removed local output directory: %s", local_dir)

    return result
