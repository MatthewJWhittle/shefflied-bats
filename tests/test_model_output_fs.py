"""Tests for fsspec model output run directories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import fsspec
import pytest

from sdm.utils.model_output_fs import (
    MODEL_OUTPUT_ROOT_ENV,
    build_run_output_uri,
    generate_run_segment,
    model_run_write_context,
    put_local_tree,
    resolve_output_root,
)


def test_resolve_output_root_prefers_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODEL_OUTPUT_ROOT_ENV, "memory://env-root")
    root = resolve_output_root(
        cli_output_dir=tmp_path / "cli",
        config_models_path=tmp_path / "config",
    )
    assert root == "memory://env-root"


def test_resolve_output_root_cli_then_config(tmp_path: Path) -> None:
    cli = tmp_path / "cli"
    config = tmp_path / "config"
    assert resolve_output_root(cli_output_dir=cli, config_models_path=config) == str(
        cli
    )
    assert resolve_output_root(cli_output_dir=None, config_models_path=config) == str(
        config
    )


def test_build_run_output_uri_local_and_remote() -> None:
    segment = "20251015T120000Z-deadbeef"
    local = build_run_output_uri("/data/models", run_segment=segment)
    assert local == str(Path("/data/models") / "runs" / segment)

    remote = build_run_output_uri("gs://bucket/prefix", run_segment=segment)
    assert remote == "gs://bucket/prefix/runs/20251015T120000Z-deadbeef/"


def test_generate_run_segment_format() -> None:
    segment = generate_run_segment(
        timestamp=datetime(2025, 10, 15, 12, 0, tzinfo=timezone.utc),
        random_suffix="abcd1234",
    )
    assert segment == "20251015T120000Z-abcd1234"


def test_model_run_write_context_memory_filesystem() -> None:
    root = "memory://test-root"
    run_uri = build_run_output_uri(root, run_segment="20251015T120000Z-test0001")

    with model_run_write_context(run_uri) as (write_dir, uri):
        assert write_dir.is_dir()
        (write_dir / "metrics.json").write_text('{"score": 1.0}')
        (write_dir / "nested").mkdir()
        (write_dir / "nested" / "values.csv").write_text("a\n1\n")
        assert uri == run_uri

    fs = fsspec.filesystem("memory")
    assert fs.exists(f"{root}/runs/20251015T120000Z-test0001/metrics.json")
    assert fs.cat(f"{root}/runs/20251015T120000Z-test0001/nested/values.csv") == b"a\n1\n"


def test_model_run_write_context_local_filesystem(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    run_uri = build_run_output_uri(str(root), run_segment="20251015T120000Z-local01")

    with model_run_write_context(run_uri) as (write_dir, uri):
        (write_dir / "summary.csv").write_text("id\n1\n")

    expected = root / "runs" / "20251015T120000Z-local01" / "summary.csv"
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
