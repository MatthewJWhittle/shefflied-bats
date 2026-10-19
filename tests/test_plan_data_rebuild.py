"""Tests for ``sdm data --plan`` (no downloads / no writes)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from typer.testing import CliRunner

from sdm.cli import app
from sdm.commands.data_preparation.plan_data_rebuild import (
    DEFAULT_CEH_PATH,
    build_data_plan,
    run_data_plan,
)
from sdm.data.coastline_download import LEGACY_BGS_GEOCOAST


def _write_boundary(path: Path) -> None:
    # Skipton-ish AOI in OSGB — intersects tile SE.
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(397_100, 450_300, 400_100, 453_300)],
        crs="EPSG:27700",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def test_plan_lists_steps_tiles_and_optional_bgs(tmp_path: Path) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    ceh = tmp_path / "gblcm2023_10m.tif"
    ceh.write_bytes(b"fake")

    steps, tiles, notes = build_data_plan(
        boundary_path=boundary,
        ceh_data_path=ceh,
        buffer_m=7000,
    )
    names = [step.name for step in steps]
    assert names[0] == "generate_terrain_data"
    assert "merge_ev_layers" in names
    assert "build_model_ev_stack" in names
    assert "SE" in tiles
    assert any("GeoCoast" in note for note in notes)
    assert str(LEGACY_BGS_GEOCOAST) in " ".join(notes)

    ceh_step = next(step for step in steps if step.name == "generate_ceh_lc_data")
    assert ceh_step.required is True
    assert ceh_step.present is True


def test_plan_exits_nonzero_when_ceh_missing(tmp_path: Path, capsys) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    missing_ceh = tmp_path / "missing-ceh.tif"

    code = run_data_plan(
        boundary_path=boundary,
        ceh_data_path=missing_ceh,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "MISSING" in captured.out or "missing" in captured.out.lower()
    assert "generate_ceh_lc_data" in captured.out


def test_plan_ok_when_ceh_present(tmp_path: Path) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    ceh = tmp_path / "gblcm2023_10m.tif"
    ceh.write_bytes(b"fake")

    code = run_data_plan(boundary_path=boundary, ceh_data_path=ceh)
    assert code == 0


def test_cli_plan_flag_no_writes(tmp_path: Path, monkeypatch) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    ceh = tmp_path / "ceh.tif"
    ceh.write_bytes(b"fake")

    # Point project boundary at our temp file; --skip ceh so missing default CEH
    # does not fail, then separately assert default CEH path is documented.
    runner = CliRunner()
    monkeypatch.setattr(
        "sdm.cli.PROJECT_CONFIG.paths.boundary",
        str(boundary),
    )
    result = runner.invoke(app, ["data", "--plan", "--skip", "ceh"])
    assert result.exit_code == 0, result.output
    assert "build_model_ev_stack" in result.output
    assert "OS 100 km tiles" in result.output
    assert "GeoCoast" in result.output
    assert "Skipped via --skip / --only (ceh)" in result.output
    # No output-dir created by plan
    assert not (tmp_path / "evs-rebuild").exists()
    assert DEFAULT_CEH_PATH.as_posix().endswith("gblcm2023_10m.tif")


def test_cli_plan_skip_ceh_coastal(tmp_path: Path, monkeypatch) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    runner = CliRunner()
    monkeypatch.setattr(
        "sdm.cli.PROJECT_CONFIG.paths.boundary",
        str(boundary),
    )
    result = runner.invoke(
        app,
        ["data", "-o", str(tmp_path / "x"), "--skip", "ceh,coastal", "--plan"],
    )
    assert result.exit_code == 0, result.output
    assert "Skipped via --skip / --only (ceh)" in result.output
    assert "Skipped via --skip / --only (coastal)" in result.output


def test_cli_plan_unknown_skip_exits_2(tmp_path: Path, monkeypatch) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    runner = CliRunner()
    monkeypatch.setattr(
        "sdm.cli.PROJECT_CONFIG.paths.boundary",
        str(boundary),
    )
    result = runner.invoke(app, ["data", "--plan", "--skip", "ceh,nope"])
    assert result.exit_code == 2, result.output
    assert "Unknown rebuild step" in result.output
    assert "nope" in result.output


def test_cli_plan_respects_ceh_path(tmp_path: Path, monkeypatch) -> None:
    boundary = tmp_path / "boundary.geojson"
    _write_boundary(boundary)
    ceh = tmp_path / "custom-ceh.tif"
    ceh.write_bytes(b"fake")

    runner = CliRunner()
    monkeypatch.setattr(
        "sdm.cli.PROJECT_CONFIG.paths.boundary",
        str(boundary),
    )
    result = runner.invoke(app, ["data", "--plan", "--ceh-path", str(ceh)])
    assert result.exit_code == 0, result.output
    assert str(ceh) in result.output


def test_plan_prints_all_tiles_for_3x3_bbox(tmp_path: Path, monkeypatch) -> None:
    """``--plan`` must list every intersecting 100 km tile, not only corners."""
    boundary = tmp_path / "boundary.geojson"
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(200_000.0, 300_000.0, 499_999.0, 599_999.0)],
        crs="EPSG:27700",
    )
    gdf.to_file(boundary, driver="GeoJSON")
    ceh = tmp_path / "ceh.tif"
    ceh.write_bytes(b"fake")

    expected = ["NX", "NY", "NZ", "SC", "SD", "SE", "SH", "SJ", "SK"]
    _steps, tiles, _notes = build_data_plan(
        boundary_path=boundary,
        ceh_data_path=ceh,
        buffer_m=0,
    )
    assert tiles == expected

    runner = CliRunner()
    monkeypatch.setattr("sdm.cli.PROJECT_CONFIG.paths.boundary", str(boundary))
    monkeypatch.setattr("sdm.cli.PROJECT_CONFIG.spatial.study_area_buffer", 0)
    result = runner.invoke(app, ["data", "--plan", "--ceh-path", str(ceh)])
    assert result.exit_code == 0, result.output
    for tile in expected:
        assert tile in result.output
    assert "OS 100 km tiles" in result.output
