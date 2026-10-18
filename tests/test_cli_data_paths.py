"""CLI path options must preserve URI schemes (pathlib.Path normalizes them away)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from sdm.cli import app


def test_train_cli_preserves_file_uri_for_ev_file() -> None:
    captured: dict[str, str] = {}

    def _stub_train(**kwargs: object) -> None:
        captured["ev_file"] = str(kwargs["ev_file"])

    runner = CliRunner()
    with patch(
        "sdm.commands.modelling.train_sdm_models.train_sdm_models",
        _stub_train,
    ):
        result = runner.invoke(
            app,
            ["train", "--ev-file", "file:///tmp/evs-to-model.tif"],
        )

    assert result.exit_code == 0, result.output
    assert captured["ev_file"] == "file:///tmp/evs-to-model.tif"


def test_train_cli_preserves_absolute_local_ev_file(
    tmp_path_factory,
) -> None:
    captured: dict[str, str] = {}
    ev = tmp_path_factory.mktemp("ev") / "evs-to-model.tif"
    ev.write_bytes(b"TIFF")

    def _stub_train(**kwargs: object) -> None:
        captured["ev_file"] = str(kwargs["ev_file"])

    runner = CliRunner()
    with patch(
        "sdm.commands.modelling.train_sdm_models.train_sdm_models",
        _stub_train,
    ):
        result = runner.invoke(app, ["train", "--ev-file", str(ev)])

    assert result.exit_code == 0, result.output
    assert captured["ev_file"] == str(ev)
