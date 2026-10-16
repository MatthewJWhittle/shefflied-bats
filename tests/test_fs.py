"""Tests for storage-agnostic ``sdm.utils.fs`` helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import fsspec
import pytest

from sdm.utils.fs import (
    join_uri_path,
    local_materialized_path,
    put_local_tree,
    read_bytes,
    staged_directory_context,
    uri_for_rasterio,
    write_bytes_idempotent,
    write_text_idempotent,
)


def test_join_uri_path_local_and_remote() -> None:
    assert join_uri_path("/data", "inputs", "file.geojson") == "/data/inputs/file.geojson"
    assert (
        join_uri_path("gs://bucket", "runs", "20251015T120000Z-abc")
        == "gs://bucket/runs/20251015T120000Z-abc"
    )


def test_uri_for_rasterio_maps_gs_to_vsigs() -> None:
    assert uri_for_rasterio("gs://bucket/evs/stack.tif") == "/vsigs/bucket/evs/stack.tif"
    assert uri_for_rasterio("/data/local.tif") == "/data/local.tif"


def test_write_bytes_idempotent_local_skips_identical(tmp_path: Path) -> None:
    target = tmp_path / "out" / "artifact.bin"
    write_bytes_idempotent(target, b"payload-v1")
    mtime_first = target.stat().st_mtime_ns
    write_bytes_idempotent(target, b"payload-v1")
    assert target.read_bytes() == b"payload-v1"
    assert target.stat().st_mtime_ns == mtime_first


def test_write_bytes_idempotent_local_overwrites_different(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    write_bytes_idempotent(target, b"before")
    write_bytes_idempotent(target, b"after")
    assert target.read_bytes() == b"after"


def test_write_bytes_idempotent_local_leaves_no_partial_on_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stable")

    real_replace = os.replace

    def fail_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == target:
            raise OSError("simulated failure")
        real_replace(src, dst)

    with mock.patch("sdm.utils.fs.os.replace", side_effect=fail_replace):
        with pytest.raises(OSError, match="simulated"):
            write_bytes_idempotent(target, b"new-bytes")

    assert target.read_bytes() == b"stable"
    partials = list(target.parent.glob(f".{target.name}.*.partial"))
    assert partials == []


def test_write_bytes_idempotent_memory_filesystem() -> None:
    uri = "memory://bucket/idempotent.txt"
    write_text_idempotent(uri, "alpha")
    write_text_idempotent(uri, "alpha")
    assert read_bytes(uri) == b"alpha"
    write_text_idempotent(uri, "beta")
    assert read_bytes(uri) == b"beta"


def test_staged_directory_context_memory_filesystem() -> None:
    root = "memory://staged-root"
    run_uri = join_uri_path(root, "runs", "20251015T120000Z-staged01")

    with staged_directory_context(run_uri) as (write_dir, uri):
        (write_dir / "metrics.json").write_text('{"score": 1.0}')
        (write_dir / "nested").mkdir()
        (write_dir / "nested" / "values.csv").write_text("a\n1\n")
        assert uri == run_uri

    fs = fsspec.filesystem("memory")
    assert fs.exists(f"{root}/runs/20251015T120000Z-staged01/metrics.json")
    assert fs.cat(f"{root}/runs/20251015T120000Z-staged01/nested/values.csv") == b"a\n1\n"


def test_staged_directory_context_local_filesystem(tmp_path: Path) -> None:
    run_uri = str(tmp_path / "outputs" / "runs" / "20251015T120000Z-local01")

    with staged_directory_context(run_uri) as (write_dir, uri):
        (write_dir / "summary.csv").write_text("id\n1\n")

    expected = tmp_path / "outputs" / "runs" / "20251015T120000Z-local01" / "summary.csv"
    assert expected.is_file()
    assert expected.read_text() == "id\n1\n"
    assert uri == run_uri


def test_put_local_tree_to_memory(tmp_path: Path) -> None:
    local = tmp_path / "staging"
    local.mkdir()
    (local / "a.txt").write_text("alpha")
    nested = local / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("beta")

    fs = fsspec.filesystem("memory")
    remote = "memory://dest/run"
    fs.makedirs(remote, exist_ok=True)
    put_local_tree(fs, local, remote)

    assert fs.cat(f"{remote}/a.txt") == b"alpha"
    assert fs.cat(f"{remote}/nested/b.txt") == b"beta"


def test_local_materialized_path_memory_roundtrip() -> None:
    fs = fsspec.filesystem("memory")
    uri = "memory://remote/data.json"
    fs.write_text(uri, '{"ok": true}')

    with local_materialized_path(uri) as local_path:
        assert local_path.is_file()
        assert local_path.read_text() == '{"ok": true}'
