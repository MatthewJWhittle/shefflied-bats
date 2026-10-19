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


def test_rebuild_only_merge_uses_existing_ceh_and_coastal(tmp_path, monkeypatch):
    """``--only merge`` must still merge CEH/coastal files already in the folder.

    Root cause: skip_ceh/skip_coastal tracked ``name not in active``, so
    ``--only merge`` omitted those layers, overwrote all-evs.tif without them,
    then build_model_ev_stack raised KeyError on selected variables.
    """
    from sdm.commands.data_preparation import rebuild_ev_layers as mod

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    out = tmp_path / "evs-rebuild"
    captured_inputs: dict = {}

    def _ok(*_args, **_kwargs):
        return out

    def _capture_inputs(*args, **kwargs):
        captured_inputs["args"] = args
        captured_inputs["kwargs"] = kwargs
        return ["terrain=x.tif"]

    monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
    for name in (
        "generate_terrain_data",
        "generate_terrain_stats",
        "generate_climate_data",
        "generate_ceh_lc_data",
        "generate_vom_data",
        "generate_coastal_distance",
        "process_os_data",
        "merge_ev_layers",
    ):
        monkeypatch.setattr(mod, name, _ok)
    monkeypatch.setattr(mod, "build_ev_dataset_inputs", _capture_inputs)
    stub_settings = {"layer_clipping": {"enabled": True, "layers": {}, "applied": {}}}
    monkeypatch.setattr(
        mod,
        "build_model_ev_stack",
        lambda *_a, **_k: (out / "evs-to-model.tif", stub_settings),
    )
    monkeypatch.setattr(mod, "build_settings_from_config", lambda *_a, **_k: stub_settings)
    monkeypatch.setattr(mod, "load_config", lambda: {})
    monkeypatch.setattr(mod, "collect_rebuild_provenance", lambda **_k: {})

    mod.rebuild_environmental_layers(
        out,
        boundary_path=boundary,
        resolution=100,
        only=["merge"],
    )

    kwargs = captured_inputs.get("kwargs", {})
    assert kwargs.get("skip_ceh") is False, (
        "--only merge must include existing CEH outputs, not skip them"
    )
    assert kwargs.get("skip_coastal") is False, (
        "--only merge must include existing coastal outputs, not skip them"
    )


def test_rebuild_explicit_skip_still_omits_ceh_from_merge(tmp_path, monkeypatch):
    """``--skip ceh,coastal`` may omit those layers from merge inputs."""
    from sdm.commands.data_preparation import rebuild_ev_layers as mod

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    out = tmp_path / "evs-rebuild"
    captured_inputs: dict = {}

    def _ok(*_args, **_kwargs):
        return out

    def _capture_inputs(*args, **kwargs):
        captured_inputs["kwargs"] = kwargs
        return ["terrain=x.tif"]

    monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
    for name in (
        "generate_terrain_data",
        "generate_terrain_stats",
        "generate_climate_data",
        "generate_ceh_lc_data",
        "generate_vom_data",
        "generate_coastal_distance",
        "process_os_data",
        "merge_ev_layers",
    ):
        monkeypatch.setattr(mod, name, _ok)
    monkeypatch.setattr(mod, "build_ev_dataset_inputs", _capture_inputs)
    stub_settings = {"layer_clipping": {"enabled": True, "layers": {}, "applied": {}}}
    monkeypatch.setattr(
        mod,
        "build_model_ev_stack",
        lambda *_a, **_k: (out / "evs-to-model.tif", stub_settings),
    )
    monkeypatch.setattr(mod, "build_settings_from_config", lambda *_a, **_k: stub_settings)
    monkeypatch.setattr(mod, "load_config", lambda: {})
    monkeypatch.setattr(mod, "collect_rebuild_provenance", lambda **_k: {})

    mod.rebuild_environmental_layers(
        out,
        boundary_path=boundary,
        resolution=100,
        skip=["ceh", "coastal"],
    )
    kwargs = captured_inputs.get("kwargs", {})
    assert kwargs.get("skip_ceh") is True
    assert kwargs.get("skip_coastal") is True


def test_rebuild_only_step_skips_model_stack(tmp_path, monkeypatch):
    """--only <step> must not build the model stack (exit cleanly after the step)."""
    from sdm.commands.data_preparation import rebuild_ev_layers as mod

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    out = tmp_path / "evs-rebuild"
    stack_calls: list = []

    def _ok(*_args, **_kwargs):
        return out

    monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
    monkeypatch.setattr(mod, "generate_climate_data", _ok)
    monkeypatch.setattr(mod, "generate_terrain_data", _ok)
    monkeypatch.setattr(mod, "generate_terrain_stats", _ok)
    monkeypatch.setattr(mod, "generate_ceh_lc_data", _ok)
    monkeypatch.setattr(mod, "generate_vom_data", _ok)
    monkeypatch.setattr(mod, "generate_coastal_distance", _ok)
    monkeypatch.setattr(mod, "process_os_data", _ok)
    monkeypatch.setattr(mod, "merge_ev_layers", _ok)
    monkeypatch.setattr(mod, "build_ev_dataset_inputs", lambda *a, **k: [])
    stub_settings = {"layer_clipping": {"enabled": True, "layers": {}, "applied": {}}}

    def _stack(*_a, **_k):
        stack_calls.append(True)
        return out / "evs-to-model.tif", stub_settings

    monkeypatch.setattr(mod, "build_model_ev_stack", _stack)
    monkeypatch.setattr(mod, "build_settings_from_config", lambda *_a, **_k: stub_settings)
    monkeypatch.setattr(mod, "load_config", lambda: {})
    monkeypatch.setattr(mod, "collect_rebuild_provenance", lambda **_k: {})

    mod.rebuild_environmental_layers(
        out,
        boundary_path=boundary,
        resolution=100,
        only=["climate"],
    )
    assert stack_calls == [], "model stack must not run when merge was not selected"


def test_cli_passes_coastline_path_to_rebuild(tmp_path: Path) -> None:
    """Explicit GeoCoast / coastline file must reach the rebuild orchestrator."""
    out = tmp_path / "safe"
    coast = tmp_path / "geocoast.shp"
    coast.write_text("placeholder", encoding="utf-8")
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
            [
                "data",
                "--output-dir",
                str(out),
                "--coastline-path",
                str(coast),
                "--only",
                "coastal",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["coastline_path"] == coast


def test_rebuild_passes_coastline_path_to_coastal_step(tmp_path, monkeypatch):
    """Orchestrator must forward coastline_path into generate_coastal_distance."""
    from sdm.commands.data_preparation import rebuild_ev_layers as mod

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    coast = tmp_path / "BGS_GeoCoast.shp"
    coast.write_text("x", encoding="utf-8")
    out = tmp_path / "evs-rebuild"
    captured: dict = {}

    def _coastal(*_args, **kwargs):
        captured.update(kwargs)
        return out / "coastal_distance.tif"

    def _ok(*_args, **_kwargs):
        return out

    monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
    monkeypatch.setattr(mod, "generate_coastal_distance", _coastal)
    for name in (
        "generate_terrain_data",
        "generate_terrain_stats",
        "generate_climate_data",
        "generate_ceh_lc_data",
        "generate_vom_data",
        "process_os_data",
        "merge_ev_layers",
    ):
        monkeypatch.setattr(mod, name, _ok)
    monkeypatch.setattr(mod, "build_ev_dataset_inputs", lambda *a, **k: [])
    stub_settings = {"layer_clipping": {"enabled": True, "layers": {}, "applied": {}}}
    monkeypatch.setattr(
        mod,
        "build_model_ev_stack",
        lambda *_a, **_k: (out / "evs-to-model.tif", stub_settings),
    )
    monkeypatch.setattr(mod, "build_settings_from_config", lambda *_a, **_k: stub_settings)
    monkeypatch.setattr(mod, "load_config", lambda: {})
    monkeypatch.setattr(mod, "collect_rebuild_provenance", lambda **_k: {})

    mod.rebuild_environmental_layers(
        out,
        boundary_path=boundary,
        resolution=100,
        only=["coastal"],
        coastline_path=coast,
    )
    assert captured.get("coastline_path") == coast
