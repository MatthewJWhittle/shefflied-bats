"""Tests for model run output path helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import fsspec
import pandas as pd
import pytest

from sdm.utils.model_runs import (
    INPUT_DATA_ROOT_ENV,
    MODEL_OUTPUT_ROOT_ENV,
    artifact_parent_uri,
    build_run_output_uri,
    generate_run_segment,
    join_run_artifact,
    model_run_write_context,
    resolve_data_path,
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


def test_resolve_data_path_with_input_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    resolved = resolve_data_path("data/processed/occurrence.geojson")
    assert (
        resolved
        == "gs://habitat-suitability-modelling-data/inputs/data/processed/occurrence.geojson"
    )


def test_resolve_data_path_leaves_absolute_local_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    local_ev = tmp_path / "evs-to-model.tif"
    resolved = resolve_data_path(local_ev)
    assert resolved == str(local_ev)


def test_resolve_data_path_leaves_file_uri_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    uri = "file:///tmp/evs-to-model.tif"
    assert resolve_data_path(uri) == uri


def test_resolve_data_path_expands_tilde_in_local_input_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INPUT_DATA_ROOT_ENV, "~/hsm-inputs")
    resolved = resolve_data_path("data/evs/evs-to-model.tif")
    expected = Path("~/hsm-inputs").expanduser() / "data/evs/evs-to-model.tif"
    assert resolved == str(expected)


def test_local_input_root_with_gcs_output_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_inputs = tmp_path / "hsm-inputs"
    monkeypatch.setenv(INPUT_DATA_ROOT_ENV, str(local_inputs))
    monkeypatch.setenv(MODEL_OUTPUT_ROOT_ENV, "gs://habitat-suitability-modelling-data")

    ev_path = resolve_data_path("data/evs/evs-to-model.tif")
    assert ev_path == str(local_inputs / "data/evs/evs-to-model.tif")

    output_root = resolve_output_root(
        cli_output_dir=None,
        config_models_path=tmp_path / "data/sdm_models",
    )
    assert output_root == "gs://habitat-suitability-modelling-data"


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
