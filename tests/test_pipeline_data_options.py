"""Regression: pipeline must not treat Typer OptionInfo defaults as true."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from sdm.cli import app, data, run_data_rebuild


def test_typer_option_default_is_truthy_optioninfo() -> None:
    """Documents the root cause: OptionInfo objects are truthy in boolean context."""
    option_default = typer.Option(False, "--skip-ceh")
    assert bool(option_default) is True
    assert not isinstance(option_default, bool)


def test_pipeline_runs_ceh_and_coastal_steps(tmp_path: Path) -> None:
    """``sdm pipeline`` must call the rebuild core with skip flags false."""
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
    assert captured.get("skip_ceh") is False
    assert captured.get("skip_coastal") is False
    assert "plan" not in captured


def test_calling_data_command_as_function_would_skip_without_core() -> None:
    """Calling the Typer ``data`` callable without kwargs leaves OptionInfo defaults.

    ``run_data_rebuild`` is the safe entry point for in-process callers.
    """
    # Inspect the wrapped Typer callback defaults indirectly: omitted kwargs on
    # the command function signature use OptionInfo, which is why pipeline must
    # not call ``data()`` directly.
    import inspect

    sig = inspect.signature(data)
    skip_ceh_default = sig.parameters["skip_ceh"].default
    skip_coastal_default = sig.parameters["skip_coastal"].default
    assert bool(skip_ceh_default) is True
    assert bool(skip_coastal_default) is True

    core_sig = inspect.signature(run_data_rebuild)
    assert core_sig.parameters["skip_ceh"].default is False
    assert core_sig.parameters["skip_coastal"].default is False
