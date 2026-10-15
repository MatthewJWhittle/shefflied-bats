"""Tests for fsspec model output run directories."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import fsspec
import pytest

import pandas as pd

from sdm.utils.model_output_fs import (
    INPUT_DATA_ROOT_ENV,
    MODEL_OUTPUT_ROOT_ENV,
    artifact_parent_uri,
    build_run_output_uri,
    generate_run_segment,
    join_run_artifact,
    model_run_write_context,
    put_local_tree,
    resolve_data_path,
    resolve_output_root,
    uri_for_rasterio,
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


def test_resolve_data_path_with_input_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    resolved = resolve_data_path("data/processed/occurrence.geojson")
    assert (
        resolved
        == "gs://habitat-suitability-modelling-data/inputs/data/processed/occurrence.geojson"
    )


def test_uri_for_rasterio_maps_gs_to_vsigs() -> None:
    assert uri_for_rasterio("gs://bucket/evs/stack.tif") == "/vsigs/bucket/evs/stack.tif"
    assert uri_for_rasterio("/data/local.tif") == "/data/local.tif"


def test_published_paths_in_model_results_use_run_uri() -> None:
    root = "memory://results-bucket"
    run_uri = build_run_output_uri(root, run_segment="20251015T120000Z-pubtest")

    with model_run_write_context(run_uri) as (write_dir, written_uri):
        pkg_dir = write_dir / "alpha_pkg"
        pkg_dir.mkdir()
        model_file = pkg_dir / "model.pkl"
        model_file.write_bytes(b"model-bytes")
        published = join_run_artifact(
            written_uri, model_file.relative_to(write_dir)
        )
        parent = artifact_parent_uri(published)
        pd.DataFrame(
            {
                "identifier": ["alpha"],
                "model_path": [published],
                "model_package_dir": [parent],
            }
        ).to_csv(write_dir / "model_results.csv", index=False)
        assert str(write_dir) not in published
        assert published.startswith("memory://results-bucket/runs/")

    fs = fsspec.filesystem("memory")
    csv_bytes = fs.cat(f"{root}/runs/20251015T120000Z-pubtest/model_results.csv")
    assert b"memory://results-bucket/runs/" in csv_bytes
    assert b"alpha_pkg/model.pkl" in csv_bytes
    assert str(write_dir).encode() not in csv_bytes


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
