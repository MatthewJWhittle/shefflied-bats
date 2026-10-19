"""Tests for registry-driven ``--skip`` / ``--only`` on ``sdm data``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sdm.cli import app, run_data_rebuild
from sdm.commands.data_preparation.rebuild_steps import (
    REBUILD_STEP_NAMES,
    UnknownRebuildStepError,
    parse_step_name_list,
    resolve_active_steps,
    valid_step_names_message,
)


def test_parse_step_name_list_accepts_comma_separated() -> None:
    assert parse_step_name_list("ceh,coastal") == frozenset({"ceh", "coastal"})
    assert parse_step_name_list(" ceh , coastal ") == frozenset({"ceh", "coastal"})
    assert parse_step_name_list(None) == frozenset()
    assert parse_step_name_list("") == frozenset()


def test_parse_step_name_list_rejects_unknown() -> None:
    with pytest.raises(UnknownRebuildStepError, match="Valid names"):
        parse_step_name_list("ceh,not-a-step")


def test_resolve_active_steps_skip_and_only() -> None:
    active = resolve_active_steps(skip=["ceh", "coastal"])
    assert "ceh" not in active
    assert "coastal" not in active
    assert "merge" in active
    assert resolve_active_steps(only=["merge"]) == frozenset({"merge"})
    with pytest.raises(ValueError, match="only one"):
        resolve_active_steps(skip=["ceh"], only=["merge"])


def test_cli_skip_ceh_coastal_reaches_rebuild(tmp_path: Path) -> None:
    out = tmp_path / "safe"
    captured: dict = {}

    def _stub(output_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir)

    runner = CliRunner()
    with patch(
        "sdm.commands.data_preparation.rebuild_ev_layers.rebuild_environmental_layers",
        _stub,
    ):
        result = runner.invoke(
            app,
            ["data", "--output-dir", str(out), "--skip", "ceh,coastal"],
        )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["skip"] == frozenset({"ceh", "coastal"})
    assert captured["kwargs"]["only"] == frozenset()


def test_cli_unknown_skip_exits_2_and_lists_valid_names(tmp_path: Path) -> None:
    out = tmp_path / "safe"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["data", "--output-dir", str(out), "--skip", "ceh,nope"],
    )
    assert result.exit_code == 2, result.output
    assert "Unknown rebuild step" in result.output
    assert "nope" in result.output
    for name in ("ceh", "coastal", "merge"):
        assert name in result.output
    assert valid_step_names_message().split(", ")[0] in result.output
    assert not out.exists()


def test_cli_skip_and_only_mutually_exclusive(tmp_path: Path) -> None:
    out = tmp_path / "safe"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data",
            "--output-dir",
            str(out),
            "--skip",
            "ceh",
            "--only",
            "merge",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "only one" in result.output.lower()


def test_run_data_rebuild_parses_skip_string(tmp_path: Path) -> None:
    out = tmp_path / "safe"
    captured: dict = {}

    def _stub(output_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return Path(output_dir)

    with patch(
        "sdm.commands.data_preparation.rebuild_ev_layers.rebuild_environmental_layers",
        _stub,
    ):
        run_data_rebuild(str(out), skip="ceh,coastal")

    assert captured["skip"] == frozenset({"ceh", "coastal"})
    assert REBUILD_STEP_NAMES >= captured["skip"]
