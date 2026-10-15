"""Tests for GCS model-run output upload, list, and download."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

from sdm.utils.model_run_gcs_upload import (
    MODEL_RUN_IMPERSONATE_SA_ENV,
    MODEL_RUN_SERVICE_ACCOUNT_ENV,
    DownloadLocalConflictError,
    RunCredentialsConfigurationError,
    UploadConflictError,
    UploadVerificationError,
    collect_local_objects,
    download_model_run_outputs,
    generate_run_prefix,
    get_gcs_credentials_and_project,
    list_model_run_folders,
    md5_hash_b64_for_file,
    upload_model_run_outputs,
    verify_upload,
)


class FakeBlob:
    def __init__(self, name: str, store: dict[str, dict[str, object]]) -> None:
        self.name = name
        self._store = store
        self.size = 0
        self.md5_hash: str | None = None

    def upload_from_filename(
        self, filename: str, *, if_generation_match: int | None = None
    ) -> None:
        if if_generation_match == 0 and self.name in self._store:
            raise PreconditionFailed("object already exists")
        path = Path(filename)
        payload = path.read_bytes()
        import base64
        import hashlib

        digest = base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")
        self.size = len(payload)
        self.md5_hash = digest
        self._store[self.name] = {
            "size": self.size,
            "md5": digest,
            "data": payload,
        }

    def download_to_filename(self, filename: str) -> None:
        entry = self._store.get(self.name)
        if entry is None:
            raise FileNotFoundError(self.name)
        Path(filename).write_bytes(bytes(entry["data"]))

    def reload(self) -> None:
        raise AssertionError("reload() must not be called during upload verification")


class FakeBucket:
    name = "test-bucket"

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, object]] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self._objects)

    def list_blobs(self, prefix: str = ""):
        for name in sorted(self._objects):
            if name.startswith(prefix):
                entry = self._objects[name]
                blob = FakeBlob(name, self._objects)
                blob.size = int(entry["size"])
                blob.md5_hash = str(entry["md5"])
                yield blob


def _write_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "nested").mkdir()
    (root / "nested" / "metrics.json").write_text('{"auc": 0.9}')
    (root / "summary.csv").write_text("id,score\na,1\n")


def _seed_run(bucket: FakeBucket, prefix: str, source_dir: Path) -> None:
    for obj in collect_local_objects(source_dir):
        blob = bucket.blob(f"{prefix}{obj.relative_path}")
        blob.upload_from_filename(str(obj.absolute_path), if_generation_match=0)


def test_generate_run_prefix_format_and_uniqueness() -> None:
    from datetime import datetime, timezone

    first = generate_run_prefix(
        timestamp=datetime(2025, 10, 15, 7, 17, tzinfo=timezone.utc),
        random_suffix="abcd1234",
    )
    second = generate_run_prefix(
        timestamp=datetime(2025, 10, 15, 7, 17, tzinfo=timezone.utc),
        random_suffix="ef567890",
    )
    assert first == "runs/20251015T071700Z-abcd1234/"
    assert second == "runs/20251015T071700Z-ef567890/"
    assert first != second

    auto_a = generate_run_prefix()
    auto_b = generate_run_prefix()
    assert auto_a.startswith("runs/")
    assert auto_a.endswith("/")
    assert auto_a != auto_b


def test_collect_local_objects_uses_relative_paths_only(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    objects = collect_local_objects(tmp_path)
    paths = {obj.relative_path for obj in objects}
    assert paths == {"nested/metrics.json", "summary.csv"}
    for obj in objects:
        assert obj.md5_hash_b64 == md5_hash_b64_for_file(obj.absolute_path)


def test_overwrite_attempt_fails_and_keeps_local_files(tmp_path: Path, capsys) -> None:
    _write_tree(tmp_path)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-deadbeef/"

    upload_model_run_outputs(
        tmp_path,
        bucket_name="test-bucket",
        run_prefix=prefix,
        delete_local=False,
        bucket_handle=bucket,
    )

    assert tmp_path.exists()
    capsys.readouterr()

    with pytest.raises(UploadConflictError):
        upload_model_run_outputs(
            tmp_path,
            bucket_name="test-bucket",
            run_prefix=prefix,
            delete_local=True,
            bucket_handle=bucket,
        )

    assert tmp_path.exists()


def test_failed_verification_does_not_delete_local(tmp_path: Path, monkeypatch) -> None:
    _write_tree(tmp_path)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-cafebabe/"

    def broken_verify(*_args, **_kwargs):
        raise UploadVerificationError("forced verification failure")

    monkeypatch.setattr(
        "sdm.utils.model_run_gcs_upload.verify_upload",
        broken_verify,
    )

    with pytest.raises(UploadVerificationError):
        upload_model_run_outputs(
            tmp_path,
            bucket_name="test-bucket",
            run_prefix=prefix,
            delete_local=True,
            bucket_handle=bucket,
        )

    assert tmp_path.exists()


def test_happy_path_deletes_local_and_prints_links(tmp_path: Path, capsys) -> None:
    _write_tree(tmp_path)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-beefbeef/"

    result = upload_model_run_outputs(
        tmp_path,
        bucket_name="test-bucket",
        run_prefix=prefix,
        delete_local=True,
        bucket_handle=bucket,
    )

    assert not tmp_path.exists()
    assert result.run_prefix == prefix
    assert result.bucket_name == "test-bucket"

    captured = capsys.readouterr().out
    assert f"gs://test-bucket/{prefix}nested/metrics.json" in captured
    assert f"gs://test-bucket/{prefix}summary.csv" in captured
    assert "console.cloud.google.com/storage/browser/test-bucket/" in captured


def test_missing_run_account_fails_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_tree(tmp_path)
    monkeypatch.delenv("MODEL_RUN_IMPERSONATE_SA", raising=False)
    monkeypatch.delenv("MODEL_RUN_SERVICE_ACCOUNT", raising=False)
    monkeypatch.setenv("MODEL_OUTPUT_BUCKET", "test-bucket")
    uploads_started: list[int] = []

    def track_upload(*_args, **_kwargs) -> None:
        uploads_started.append(1)

    monkeypatch.setattr(
        "sdm.utils.model_run_gcs_upload._upload_objects",
        track_upload,
    )

    with pytest.raises(RunCredentialsConfigurationError):
        upload_model_run_outputs(tmp_path, bucket_name="test-bucket")

    assert not uploads_started


def test_credentials_from_matching_service_account_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_sa = "run-sa@test.iam.gserviceaccount.com"
    monkeypatch.setenv(MODEL_RUN_SERVICE_ACCOUNT_ENV, run_sa)
    key_path = tmp_path / "run-key.json"
    key_path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "client_email": run_sa,
                "project_id": "test-project",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))

    sentinel = object()

    def fake_from_file(path: str, scopes: list[str]) -> object:
        assert path == str(key_path)
        assert scopes
        return sentinel

    monkeypatch.setattr(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        fake_from_file,
    )

    creds, project_id = get_gcs_credentials_and_project()
    assert creds is sentinel
    assert project_id == "test-project"


def test_credentials_reject_mismatched_service_account_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODEL_RUN_SERVICE_ACCOUNT_ENV, "expected-sa@test.iam.gserviceaccount.com")
    key_path = tmp_path / "run-key.json"
    key_path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "client_email": "other-sa@test.iam.gserviceaccount.com",
                "project_id": "test-project",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))

    with pytest.raises(RunCredentialsConfigurationError, match="client_email"):
        get_gcs_credentials_and_project()


def test_credentials_impersonation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_sa = "run-sa@test.iam.gserviceaccount.com"
    monkeypatch.setenv(MODEL_RUN_IMPERSONATE_SA_ENV, run_sa)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    source = object()
    impersonated = object()

    monkeypatch.setattr(
        "google.auth.default",
        lambda scopes: (source, "adc-project"),
    )

    class FakeImpersonatedCredentials:
        def __init__(
            self,
            *,
            source_credentials: object,
            target_principal: str,
            target_scopes: list[str],
        ) -> None:
            assert source_credentials is source
            assert target_principal == run_sa
            assert target_scopes

    monkeypatch.setattr(
        "google.auth.impersonated_credentials.Credentials",
        FakeImpersonatedCredentials,
    )

    creds, project_id = get_gcs_credentials_and_project()
    assert isinstance(creds, FakeImpersonatedCredentials)
    assert project_id == "adc-project"


def test_verify_upload_detects_md5_mismatch(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    local_objects = collect_local_objects(tmp_path)
    bucket = FakeBucket()
    prefix = "runs/test/"
    for obj in local_objects:
        blob = bucket.blob(f"{prefix}{obj.relative_path}")
        blob.upload_from_filename(str(obj.absolute_path), if_generation_match=0)

    bucket._objects[f"{prefix}summary.csv"]["md5"] = "wrong"
    with pytest.raises(UploadVerificationError, match="MD5 mismatch"):
        verify_upload(bucket, prefix, local_objects)


def test_list_model_run_folders(tmp_path: Path) -> None:
    bucket = FakeBucket()
    _write_tree(tmp_path)
    _seed_run(bucket, "runs/20251015T071700Z-aaa11111/", tmp_path)
    _seed_run(bucket, "runs/20251015T071800Z-bbb22222/", tmp_path)

    folders = list_model_run_folders(
        bucket_name="test-bucket",
        bucket_handle=bucket,
    )
    assert [folder.run_id for folder in folders] == [
        "20251015T071700Z-aaa11111",
        "20251015T071800Z-bbb22222",
    ]


def test_download_happy_path(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    _write_tree(source)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-dl000001/"
    _seed_run(bucket, prefix, source)

    dest = tmp_path / "dest"
    result = download_model_run_outputs(
        prefix,
        dest,
        bucket_name="test-bucket",
        bucket_handle=bucket,
    )

    assert result.dest_dir == dest.resolve()
    assert (dest / "summary.csv").is_file()
    assert (dest / "nested" / "metrics.json").is_file()
    output = capsys.readouterr().out
    assert str(dest.resolve()) in output


def test_download_refuses_existing_local_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tree(source)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-dl000002/"
    _seed_run(bucket, prefix, source)

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "summary.csv").write_text("existing")

    with pytest.raises(DownloadLocalConflictError):
        download_model_run_outputs(
            prefix,
            dest,
            bucket_name="test-bucket",
            bucket_handle=bucket,
        )


def test_download_detects_md5_mismatch_after_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_tree(source)
    bucket = FakeBucket()
    prefix = "runs/20251015T071700Z-dl000003/"
    _seed_run(bucket, prefix, source)

    original = md5_hash_b64_for_file

    def wrong_md5(path: Path) -> str:
        if path.name == "summary.csv":
            return "invalid-md5-hash-value"
        return original(path)

    monkeypatch.setattr(
        "sdm.utils.model_run_gcs_upload.md5_hash_b64_for_file",
        wrong_md5,
    )

    dest = tmp_path / "dest"
    with pytest.raises(UploadVerificationError, match="MD5 mismatch"):
        download_model_run_outputs(
            prefix,
            dest,
            bucket_name="test-bucket",
            bucket_handle=bucket,
        )
