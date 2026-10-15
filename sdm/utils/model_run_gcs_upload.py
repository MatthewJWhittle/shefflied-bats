"""
Upload and download model-run output directories in Google Cloud Storage.

Run identity is always a configured service account (never end-user ADC alone).
Auth: impersonation via ``MODEL_RUN_IMPERSONATE_SA`` + ADC, or a matching
``GOOGLE_APPLICATION_CREDENTIALS`` service-account key file.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
MODEL_RUN_SERVICE_ACCOUNT_ENV = "MODEL_RUN_SERVICE_ACCOUNT"
RUNS_ROOT_PREFIX = "runs/"

GCS_RW_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"


class UploadConflictError(RuntimeError):
    """An object already exists at the destination (if_generation_match=0 failed)."""


class UploadVerificationError(RuntimeError):
    """Uploaded objects do not match the local manifest."""


class DownloadLocalConflictError(RuntimeError):
    """Refusing to overwrite an existing local file."""


class RunCredentialsConfigurationError(ValueError):
    """Run service account or credentials are not configured correctly."""


# Backward-compatible alias for earlier upload-only errors.
ImpersonationRequiredError = RunCredentialsConfigurationError


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


@dataclass(frozen=True)
class ModelRunFolder:
    run_id: str
    prefix: str
    gs_uri: str


@dataclass(frozen=True)
class DownloadResult:
    bucket_name: str
    run_prefix: str
    dest_dir: Path
    relative_paths: tuple[str, ...]


class BlobLike(Protocol):
    name: str
    size: int
    md5_hash: str

    def upload_from_filename(
        self, filename: str, *, if_generation_match: int | None = None
    ) -> None: ...

    def download_to_filename(self, filename: str) -> None: ...


class BucketLike(Protocol):
    name: str

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
    return f"{RUNS_ROOT_PREFIX}{stamp}-{suffix}/"


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


def resolve_expected_run_service_account() -> str:
    service_account = os.environ.get(MODEL_RUN_SERVICE_ACCOUNT_ENV, "").strip()
    if service_account:
        return service_account
    impersonate = os.environ.get(MODEL_RUN_IMPERSONATE_SA_ENV, "").strip()
    if impersonate:
        return impersonate
    raise RunCredentialsConfigurationError(
        f"Set {MODEL_RUN_SERVICE_ACCOUNT_ENV} or {MODEL_RUN_IMPERSONATE_SA_ENV} to the "
        "service account email used for model-run GCS access."
    )


def _load_service_account_key_metadata(key_path: Path) -> tuple[str, str | None]:
    try:
        with key_path.open(encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunCredentialsConfigurationError(
            f"Could not read service account key at {key_path}"
        ) from exc

    if info.get("type") != "service_account":
        raise RunCredentialsConfigurationError(
            "GOOGLE_APPLICATION_CREDENTIALS must point to a service_account JSON key."
        )
    client_email = str(info.get("client_email", "")).strip()
    if not client_email:
        raise RunCredentialsConfigurationError(
            "Service account key JSON is missing client_email."
        )
    project_id = info.get("project_id")
    project = str(project_id).strip() if project_id else None
    return client_email, project


def _credentials_from_service_account_key(
    expected_sa: str,
) -> tuple[object, str | None] | None:
    key_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not key_env:
        return None

    key_path = Path(key_env)
    if not key_path.is_file():
        raise RunCredentialsConfigurationError(
            f"GOOGLE_APPLICATION_CREDENTIALS file not found: {key_path}"
        )

    client_email, project_id = _load_service_account_key_metadata(key_path)
    if client_email != expected_sa:
        raise RunCredentialsConfigurationError(
            "Service account key client_email does not match the expected run "
            f"account ({expected_sa!r})."
        )

    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=[GCS_RW_SCOPE],
    )
    logger.info(
        "Using service account key credentials for run account %s",
        expected_sa,
    )
    return credentials, project_id


def get_gcs_credentials_and_project() -> tuple[object, str | None]:
    expected_sa = resolve_expected_run_service_account()

    key_credentials = _credentials_from_service_account_key(expected_sa)
    if key_credentials is not None:
        return key_credentials

    impersonate_sa = os.environ.get(MODEL_RUN_IMPERSONATE_SA_ENV, "").strip()
    if not impersonate_sa:
        raise RunCredentialsConfigurationError(
            "No valid run credentials configured. Either set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service account key whose "
            f"client_email is {expected_sa!r}, or set {MODEL_RUN_IMPERSONATE_SA_ENV} "
            "and authenticate ADC with "
            "`gcloud auth application-default login` for impersonation."
        )
    if impersonate_sa != expected_sa:
        raise RunCredentialsConfigurationError(
            f"{MODEL_RUN_IMPERSONATE_SA_ENV} ({impersonate_sa!r}) must match the "
            f"expected run account ({expected_sa!r})."
        )

    import google.auth
    from google.auth import impersonated_credentials

    source_credentials, project = google.auth.default(scopes=[GCS_RW_SCOPE])
    credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=impersonate_sa,
        target_scopes=[GCS_RW_SCOPE],
    )
    logger.info("Using impersonated credentials for run account %s", impersonate_sa)
    return credentials, project


def resolve_bucket_name(bucket_name: str | None) -> str:
    resolved = bucket_name or os.environ.get(MODEL_OUTPUT_BUCKET_ENV)
    if not resolved:
        raise ValueError(
            f"GCS bucket not configured; set {MODEL_OUTPUT_BUCKET_ENV} or pass bucket_name."
        )
    return resolved


def open_gcs_bucket(
    bucket_name: str | None,
    *,
    storage_client: storage.Client | None = None,
    bucket_handle: BucketLike | None = None,
) -> tuple[BucketLike, str, str | None]:
    resolved_bucket = resolve_bucket_name(bucket_name)
    if bucket_handle is not None:
        project_id = storage_client.project if storage_client else None
        return bucket_handle, resolved_bucket, project_id

    credentials, project_id = get_gcs_credentials_and_project()
    client = storage_client or storage.Client(
        credentials=credentials, project=project_id
    )
    return client.bucket(resolved_bucket), resolved_bucket, project_id


def parse_run_location(
    run_id_or_gs_path: str,
    bucket_name: str | None,
) -> tuple[str, str]:
    reference = run_id_or_gs_path.strip()
    if reference.startswith("gs://"):
        without_scheme = reference[len("gs://") :]
        bucket, separator, object_path = without_scheme.partition("/")
        if not bucket or not separator:
            raise ValueError(
                f"Invalid GCS URI (expected gs://bucket/{RUNS_ROOT_PREFIX}…): {reference!r}"
            )
        prefix = object_path if object_path.endswith("/") else f"{object_path}/"
        if not prefix.startswith(RUNS_ROOT_PREFIX):
            raise ValueError(
                f"GCS run path must be under {RUNS_ROOT_PREFIX}: {reference!r}"
            )
        return bucket, prefix

    resolved_bucket = resolve_bucket_name(bucket_name)
    if reference.startswith(RUNS_ROOT_PREFIX):
        prefix = reference if reference.endswith("/") else f"{reference}/"
    else:
        prefix = f"{RUNS_ROOT_PREFIX}{reference.strip('/')}/"
    return resolved_bucket, prefix


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
                f"Refusing to overwrite existing object gs://{bucket.name}/{blob_name}"
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


def verify_downloaded_files(
    dest_dir: Path,
    remote_manifest: dict[str, tuple[int, str]],
) -> None:
    for rel, (remote_size, remote_md5) in remote_manifest.items():
        local_path = dest_dir / rel
        if not local_path.is_file():
            raise UploadVerificationError(f"Downloaded file missing: {local_path}")
        local_size = local_path.stat().st_size
        if local_size != remote_size:
            raise UploadVerificationError(
                f"Size mismatch for {rel}: local={local_size}, remote={remote_size}"
            )
        if md5_hash_b64_for_file(local_path) != remote_md5:
            raise UploadVerificationError(f"MD5 mismatch for {rel}")


def list_model_run_folders(
    *,
    bucket_name: str | None = None,
    bucket_handle: BucketLike | None = None,
    storage_client: storage.Client | None = None,
) -> list[ModelRunFolder]:
    bucket, resolved_bucket, _ = open_gcs_bucket(
        bucket_name,
        storage_client=storage_client,
        bucket_handle=bucket_handle,
    )
    run_ids: set[str] = set()
    for blob in bucket.list_blobs(prefix=RUNS_ROOT_PREFIX):
        if not blob.name.startswith(RUNS_ROOT_PREFIX):
            continue
        remainder = blob.name[len(RUNS_ROOT_PREFIX) :]
        if not remainder:
            continue
        run_id = remainder.split("/", 1)[0]
        if run_id:
            run_ids.add(run_id)

    folders: list[ModelRunFolder] = []
    for run_id in sorted(run_ids):
        prefix = f"{RUNS_ROOT_PREFIX}{run_id}/"
        folders.append(
            ModelRunFolder(
                run_id=run_id,
                prefix=prefix,
                gs_uri=f"gs://{resolved_bucket}/{prefix}",
            )
        )
    return folders


def print_model_run_list(folders: Sequence[ModelRunFolder]) -> None:
    if not folders:
        print(f"No run folders found under {RUNS_ROOT_PREFIX}")
        return
    for folder in folders:
        print(folder.gs_uri)


def download_model_run_outputs(
    run_id_or_gs_path: str,
    dest_dir: Path,
    *,
    bucket_name: str | None = None,
    bucket_handle: BucketLike | None = None,
    storage_client: storage.Client | None = None,
) -> DownloadResult:
    target_bucket, run_prefix = parse_run_location(run_id_or_gs_path, bucket_name)
    bucket, resolved_bucket, _ = open_gcs_bucket(
        target_bucket,
        storage_client=storage_client,
        bucket_handle=bucket_handle,
    )

    remote = _remote_manifest(bucket, run_prefix)
    if not remote:
        raise FileNotFoundError(
            f"No objects found at gs://{resolved_bucket}/{run_prefix}"
        )

    destination = dest_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for rel in sorted(remote):
        local_path = destination / rel
        if local_path.exists():
            raise DownloadLocalConflictError(
                f"Refusing to overwrite existing file: {local_path}"
            )

    for rel in sorted(remote):
        local_path = destination / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(f"{run_prefix}{rel}")
        blob.download_to_filename(str(local_path))

    verify_downloaded_files(destination, remote)

    relative_paths = tuple(sorted(remote.keys()))
    print(f"Downloaded gs://{resolved_bucket}/{run_prefix} to {destination}")
    for rel in relative_paths:
        print(f"  {destination / rel}")

    return DownloadResult(
        bucket_name=resolved_bucket,
        run_prefix=run_prefix,
        dest_dir=destination,
        relative_paths=relative_paths,
    )


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

    ``bucket_handle`` supports tests; when omitted, configured run credentials are used.
    """
    bucket, resolved_bucket, project_id = open_gcs_bucket(
        bucket_name,
        storage_client=storage_client,
        bucket_handle=bucket_handle,
    )
    local_objects = collect_local_objects(local_dir)
    prefix = run_prefix or generate_run_prefix()

    logger.info(
        "Uploading %d file(s) from %s to gs://%s/%s",
        len(local_objects),
        local_dir,
        resolved_bucket,
        prefix,
    )

    _upload_objects(bucket, prefix, local_objects)
    verify_upload(bucket, prefix, local_objects)

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
