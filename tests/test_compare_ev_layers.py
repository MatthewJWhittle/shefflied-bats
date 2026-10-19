"""Synthetic-raster tests for environmental layer comparison (#38)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from sdm.commands.data_preparation.processing.compare_ev_layers import (
    compare_ev_layers,
    load_priority_layer_names,
)
from sdm.raster.layer_compare import (
    LayerValueKind,
    LayerVerdict,
    classify_value_kind,
    compare_layer_pair,
    discover_layers,
    ordered_layer_names,
)


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float = -9999.0,
    description: str | None = None,
    crs: str = "EPSG:27700",
    origin: tuple[float, float] = (400_000.0, 500_000.0),
    cell_size: float = 100.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    transform = from_origin(origin[0], origin[1], cell_size, cell_size)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if description:
            dst.descriptions = [description]
    return path


@pytest.fixture
def variables_config(tmp_path: Path) -> Path:
    path = tmp_path / "variables_config.yml"
    path.write_text(
        yaml.dump(
            {
                "variables": [
                    "os_distance_distance_to_minor_roads",
                    "ceh_landcover_arable_1000m",
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_classify_value_kind():
    assert classify_value_kind("ceh_landcover_arable_1000m") is LayerValueKind.EXACT
    assert classify_value_kind("os_cover_buildings") is LayerValueKind.EXACT
    assert (
        classify_value_kind("os_distance_distance_to_minor_roads")
        is LayerValueKind.CONTINUOUS
    )
    assert classify_value_kind("climate_bioclim_bio_2") is LayerValueKind.CONTINUOUS
    assert classify_value_kind("vom_vegetation_height_mean") is LayerValueKind.CONTINUOUS


def test_ordered_layer_names_priority_then_rest():
    ordered = ordered_layer_names(
        ["b_first", "a_second"],
        ["z_extra", "b_first", "m_mid"],
    )
    assert ordered == ["b_first", "a_second", "m_mid", "z_extra"]


def test_discover_layers_single_and_multiband(tmp_path: Path):
    _write_raster(
        tmp_path / "os_distance_distance_to_minor_roads.tif",
        np.zeros((4, 4), dtype=np.float32),
        description="os_distance_distance_to_minor_roads",
    )
    multi = tmp_path / "stack.tif"
    data = np.stack(
        [
            np.ones((4, 4), dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
        ]
    )
    transform = from_origin(400_000.0, 500_000.0, 100.0, 100.0)
    with rasterio.open(
        multi,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data)
        dst.descriptions = ["terrain_stats_slope", "climate_bioclim_bio_2"]

    found = discover_layers(tmp_path)
    assert "os_distance_distance_to_minor_roads" in found
    assert "terrain_stats_slope" in found
    assert "climate_bioclim_bio_2" in found


def test_identical_layers_pass(tmp_path: Path, variables_config: Path):
    current = tmp_path / "current"
    rebuilt = tmp_path / "rebuilt"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    name = "os_distance_distance_to_minor_roads"
    _write_raster(current / f"{name}.tif", data, description=name)
    _write_raster(rebuilt / f"{name}.tif", data.copy(), description=name)

    out_csv = tmp_path / "out.csv"
    summary = compare_ev_layers(
        rebuilt,
        current,
        out_csv,
        variables_config_path=variables_config,
        difference_maps_dir=tmp_path / "diffs",
    )
    by_name = {r.layer_name: r for r in summary.layers}
    assert by_name[name].verdict is LayerVerdict.PASS
    assert by_name[name].share_within_tolerance == 1.0
    assert by_name[name].grid_ok is True
    assert Path(by_name[name].difference_map_path).exists()
    # Priority names missing from folders still appear as missing rows.
    assert by_name["ceh_landcover_arable_1000m"].verdict is LayerVerdict.MISSING
    df = pd.read_csv(out_csv)
    assert list(df["layer_name"])[0] == "os_distance_distance_to_minor_roads"


def test_continuous_within_one_percent_passes(tmp_path: Path):
    name = "climate_stats_temp_ann_avg"
    # Range 0–100 → 1% tolerance = 1.0
    current = np.linspace(0, 100, 16, dtype=np.float32).reshape(4, 4)
    rebuilt = current + 0.5  # within 1% of range
    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    assert result.verdict is LayerVerdict.PASS
    assert result.share_within_tolerance == 1.0
    assert result.max_difference == pytest.approx(0.5)


def test_continuous_local_patch_still_passes_when_share_ok(tmp_path: Path):
    name = "os_distance_distance_to_major_roads"
    current = np.zeros((10, 10), dtype=np.float32)
    current[0, :] = np.linspace(0, 100, 10)
    rebuilt = current.copy()
    # Compact 2×2 patch far outside 1% of range (range≈100 → tol=1).
    rebuilt[4:6, 4:6] = 50.0

    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    # 4 of 100 cells differ → 96% within tolerance → still ≥95%, and local → pass
    assert result.share_within_tolerance == pytest.approx(0.96)
    assert result.differences_local is True
    assert result.verdict is LayerVerdict.PASS


def test_exact_category_mismatch_share(tmp_path: Path):
    name = "ceh_landcover_arable_1000m"
    current = np.ones((5, 5), dtype=np.float32)
    rebuilt = current.copy()
    rebuilt[0, 0] = 2.0  # 1/25 = 4% mismatch → 96% within → pass if local

    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    assert result.value_kind is LayerValueKind.EXACT
    assert result.share_within_tolerance == pytest.approx(0.96)
    assert result.verdict is LayerVerdict.PASS


def test_grid_mismatch_fails(tmp_path: Path):
    name = "terrain_stats_slope"
    data = np.ones((4, 4), dtype=np.float32)
    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", data, description=name, cell_size=100.0)
    _write_raster(reb_dir / f"{name}.tif", data, description=name, cell_size=50.0)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    assert result.grid_ok is False
    assert result.verdict is LayerVerdict.FAIL
    assert "cell_size" in result.grid_check


def test_source_change_flags_not_fails(tmp_path: Path):
    name = "os_cover_buildings"
    current = np.zeros((8, 8), dtype=np.float32)
    rebuilt = np.ones((8, 8), dtype=np.float32)  # 0% within tolerance
    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
        source_version_current="OS 2023",
        source_version_rebuilt="OS 2024",
    )
    assert result.verdict is LayerVerdict.FLAGGED_SOURCE_CHANGE


def test_widespread_mismatch_fails(tmp_path: Path):
    name = "vom_vegetation_height_mean"
    rng = np.random.default_rng(0)
    current = rng.uniform(0, 100, size=(20, 20)).astype(np.float32)
    # Scatter large differences across many cells (~20%).
    rebuilt = current.copy()
    mask = rng.random(current.shape) < 0.2
    rebuilt[mask] = current[mask] + 50.0

    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    assert result.share_within_tolerance is not None
    assert result.share_within_tolerance < 0.95
    assert result.verdict is LayerVerdict.FAIL


def test_nodata_cells_must_match(tmp_path: Path):
    name = "climate_bioclim_bio_2"
    current = np.ones((4, 4), dtype=np.float32)
    rebuilt = current.copy()
    current[0, 0] = -9999.0
    # rebuilt keeps a valid value in that cell → nodata cell mismatch

    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", current, description=name)
    _write_raster(reb_dir / f"{name}.tif", rebuilt, description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
    )
    assert result.grid_ok is False
    assert "nodata_cells" in result.grid_check
    assert result.verdict is LayerVerdict.FAIL


def test_cli_compare_layers(tmp_path: Path, variables_config: Path):
    from typer.testing import CliRunner

    from sdm.cli import app

    name = "os_distance_distance_to_minor_roads"
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    current = tmp_path / "current"
    rebuilt = tmp_path / "rebuilt"
    _write_raster(current / f"{name}.tif", data, description=name)
    _write_raster(rebuilt / f"{name}.tif", data.copy(), description=name)
    out_csv = tmp_path / "cmp.csv"
    hooks = tmp_path / "hooks.csv"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compare-layers",
            str(rebuilt),
            str(current),
            "-o",
            str(out_csv),
            "--variables-config",
            str(variables_config),
            "--difference-maps-dir",
            str(tmp_path / "diffs"),
            "--model-hooks-csv",
            str(hooks),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    assert hooks.exists()
    hooks_df = pd.read_csv(hooks)
    assert hooks_df.loc[0, "verdict"] == "not_run"


def test_load_priority_from_repo_config():
    path = Path("data/sdm_config/Plecotus_auritus_Roost/variables_config.yml")
    names = load_priority_layer_names(path)
    assert "os_distance_distance_to_minor_roads" in names
    assert names[0] == "os_distance_distance_to_minor_roads"
