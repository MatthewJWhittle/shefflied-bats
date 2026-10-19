"""Safety guards for EV rebuild output directories."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sdm.cli import app
from sdm.utils.ev_output_guards import (
    ForbiddenEvOutputError,
    normalize_uri_string,
    validate_ev_output_dir,
)
from sdm.utils.model_runs import INPUT_DATA_ROOT_ENV


def test_normalize_repairs_pathlib_collapsed_gs_uri() -> None:
    collapsed = str(Path("gs://bucket/x"))
    assert collapsed == "gs:/bucket/x"
    assert normalize_uri_string(collapsed) == "gs://bucket/x"
    assert normalize_uri_string("gs://bucket/x") == "gs://bucket/x"


def test_allows_caller_chosen_temp_directory(tmp_path: Path) -> None:
    out = tmp_path / "evs-rebuild"
    resolved = validate_ev_output_dir(out, cwd=tmp_path)
    assert resolved == out.resolve()


def test_refuses_data_evs_relative(tmp_path: Path) -> None:
    (tmp_path / "data" / "evs").mkdir(parents=True)
    with pytest.raises(ForbiddenEvOutputError, match="current EV directory"):
        validate_ev_output_dir("data/evs", cwd=tmp_path)


def test_refuses_data_evs_via_parent_traversal(tmp_path: Path) -> None:
    (tmp_path / "data" / "evs").mkdir(parents=True)
    with pytest.raises(ForbiddenEvOutputError, match="current EV directory"):
        validate_ev_output_dir(tmp_path / "data" / "other" / ".." / "evs", cwd=tmp_path)


def test_refuses_subdirectory_of_current_evs(tmp_path: Path) -> None:
    (tmp_path / "data" / "evs").mkdir(parents=True)
    with pytest.raises(ForbiddenEvOutputError, match="current EV directory"):
        validate_ev_output_dir("data/evs/terrain", cwd=tmp_path)


def test_refuses_path_under_inputs_segment(tmp_path: Path) -> None:
    target = tmp_path / "inputs" / "data" / "evs"
    target.mkdir(parents=True)
    with pytest.raises(ForbiddenEvOutputError, match="inputs"):
        validate_ev_output_dir(target, cwd=tmp_path)


def test_refuses_gs_inputs_uri() -> None:
    with pytest.raises(ForbiddenEvOutputError, match="bucket inputs"):
        validate_ev_output_dir("gs://habitat-suitability-modelling-data/inputs/data/evs")


def test_refuses_pathlib_collapsed_gs_inputs_uri() -> None:
    collapsed = str(Path("gs://habitat-suitability-modelling-data/inputs/data/evs"))
    with pytest.raises(ForbiddenEvOutputError, match="bucket inputs|URI|local directory"):
        validate_ev_output_dir(collapsed)


def test_refuses_gs_inputs_root_uri() -> None:
    with pytest.raises(ForbiddenEvOutputError, match="bucket inputs"):
        validate_ev_output_dir("gs://habitat-suitability-modelling-data/inputs")


def test_refuses_non_inputs_cloud_uri() -> None:
    with pytest.raises(ForbiddenEvOutputError, match="local directory"):
        validate_ev_output_dir("gs://habitat-suitability-modelling-data/scratch/rebuild")


def test_refuses_under_input_data_root_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = tmp_path / "hsm-inputs"
    mirror.mkdir()
    monkeypatch.setenv(INPUT_DATA_ROOT_ENV, str(mirror))
    with pytest.raises(ForbiddenEvOutputError, match=INPUT_DATA_ROOT_ENV):
        validate_ev_output_dir(mirror / "data" / "evs-rebuild", cwd=tmp_path)


def test_cli_data_rejects_forbidden_output() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["data", "--output-dir", "data/evs"])
    assert result.exit_code == 2
    assert "Refusing" in result.output


def test_cli_data_rejects_gs_uri_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI takes --output-dir as str; gs:// must exit 2 and write nothing."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["data", "-o", "gs://bucket/x"])
    assert result.exit_code == 2, result.output
    assert "URI" in result.output or "Refusing" in result.output
    assert list(tmp_path.iterdir()) == []


def test_cli_data_rejects_gs_inputs_uri() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data",
            "--output-dir",
            "gs://habitat-suitability-modelling-data/inputs/data/evs",
        ],
    )
    assert result.exit_code == 2
    assert "Refusing" in result.output


def test_cli_data_accepts_allowed_temp_dir(tmp_path: Path) -> None:
    allowed = tmp_path / "safe-rebuild"
    called: dict[str, object] = {}

    def _stub_rebuild(output_dir, **kwargs):  # type: ignore[no-untyped-def]
        resolved = validate_ev_output_dir(output_dir)
        called["output"] = resolved
        called["kwargs"] = kwargs
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    runner = CliRunner()
    with patch(
        "sdm.commands.data_preparation.rebuild_ev_layers.rebuild_environmental_layers",
        _stub_rebuild,
    ):
        result = runner.invoke(
            app,
            ["data", "--output-dir", str(allowed), "--cache-dir", str(tmp_path / "caches"), "--fresh"],
        )

    assert result.exit_code == 0, result.output
    assert called["output"] == allowed.resolve()
    assert called["kwargs"]["fresh"] is True
    assert Path(str(called["kwargs"]["cache_dir"])) == tmp_path / "caches"


def test_rebuild_cache_layout_defaults_under_output(tmp_path: Path) -> None:
    from sdm.commands.data_preparation.rebuild_ev_layers import rebuild_cache_layout

    layout = rebuild_cache_layout(tmp_path / "cache")
    assert layout["worldclim"].is_dir()
    assert layout["os_raw"].is_dir()
    assert layout["os_parquets"].is_dir()
    assert layout["boundary_line"].is_dir()
