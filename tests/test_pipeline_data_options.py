"""Regression: pipeline must not treat Typer OptionInfo defaults as true."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from sdm.cli import app, data, run_data_rebuild


def test_typer_option_default_is_truthy_optioninfo() -> None:
    """Documents the root cause: OptionInfo objects are truthy in boolean context."""
    option_default = typer.Option(None, "--skip")
    assert bool(option_default) is True
    assert not isinstance(option_default, str)


def test_pipeline_runs_all_rebuild_steps(tmp_path: Path) -> None:
    """``sdm pipeline`` must call the rebuild core with no skip/only filters."""
    out = tmp_path / "evs-rebuild"
    captured: dict = {}

    def _fake_rebuild(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return Path(output_dir)

    runner = CliRunner()
    with patch("sdm.cli.run_data_rebuild", side_effect=_fake_rebuild), patch(
        "sdm.cli.train", MagicMock()
    ), patch("sdm.cli.predict", MagicMock()), patch(
        "sdm.cli.visualize", MagicMock()
    ):
        result = runner.invoke(app, ["pipeline", "--output-dir", str(out)])

    assert result.exit_code == 0, result.output
    assert captured.get("skip") is None
    assert captured.get("only") is None
    assert "plan" not in captured


def test_calling_data_command_as_function_would_skip_without_core() -> None:
    """Calling the Typer ``data`` callable without kwargs leaves OptionInfo defaults.

    ``run_data_rebuild`` is the safe entry point for in-process callers.
    """
    import inspect

    sig = inspect.signature(data)
    skip_default = sig.parameters["skip"].default
    only_default = sig.parameters["only"].default
    assert bool(skip_default) is True
    assert bool(only_default) is True

    core_sig = inspect.signature(run_data_rebuild)
    assert core_sig.parameters["skip"].default is None
    assert core_sig.parameters["only"].default is None
