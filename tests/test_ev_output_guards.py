"""Safety guards for EV rebuild output directories."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sdm.cli import app
from sdm.utils.ev_output_guards import (
    ForbiddenEvOutputError,
    validate_ev_output_dir,
)
from sdm.utils.model_runs import INPUT_DATA_ROOT_ENV


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
        result = runner.invoke(app, ["data", "--output-dir", str(allowed)])

    assert result.exit_code == 0, result.output
    assert called["output"] == allowed.resolve()
