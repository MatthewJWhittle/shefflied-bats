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
    default_priority_config_paths,
    load_priority_layer_names,
    validate_comparison_write_path,
)
from sdm.raster.layer_compare import (
    LayerValueKind,
    LayerVerdict,
    analyze_mismatch_patches,
    classify_value_kind,
    compare_layer_pair,
    discover_layers,
    is_exact_match_layer,
    ordered_layer_names,
)
from sdm.utils.ev_output_guards import ForbiddenEvOutputError


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
def variables_configs(tmp_path: Path) -> list[Path]:
    a = tmp_path / "model_a" / "variables_config.yml"
    b = tmp_path / "model_b" / "variables_config.yml"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_text(
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
    b.write_text(
        yaml.dump(
            {
                "variables": [
                    "ceh_landcover_arable_1000m",
                    "climate_bioclim_bio_2",
                ]
            }
        ),
        encoding="utf-8",
    )
    return [a, b]


def test_classify_value_kind_and_exact_rule_centralized():
    assert classify_value_kind("ceh_landcover_arable_1000m") is LayerValueKind.EXACT
    assert classify_value_kind("ceh_landcover_urban_1500m") is LayerValueKind.EXACT
    assert is_exact_match_layer("ceh_landcover_broadleaved_woodland_500m")
    assert classify_value_kind("os_cover_buildings") is LayerValueKind.EXACT
    assert (
        classify_value_kind("os_distance_distance_to_minor_roads")
        is LayerValueKind.CONTINUOUS
    )
    assert classify_value_kind("climate_bioclim_bio_2") is LayerValueKind.CONTINUOUS


def test_priority_union_preserves_first_seen_order(variables_configs: list[Path]):
    names = load_priority_layer_names(variables_configs)
    assert names == [
        "os_distance_distance_to_minor_roads",
        "ceh_landcover_arable_1000m",
        "climate_bioclim_bio_2",
    ]


def test_default_priority_configs_cover_release_models():
    paths = default_priority_config_paths()
    assert paths
    assert any(p.parent.name == "Plecotus_auritus_Roost" for p in paths)
    names = load_priority_layer_names(paths)
    assert "os_distance_distance_to_minor_roads" in names


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


def test_stack_only_ignores_intermediate_files(tmp_path: Path):
    _write_raster(
        tmp_path / "intermediate.tif",
        np.ones((3, 3), dtype=np.float32),
        description="terrain_stats_slope",
    )
    stack = tmp_path / "evs-to-model.tif"
    data = np.stack(
        [
            np.full((3, 3), 1.0, dtype=np.float32),
            np.full((3, 3), 2.0, dtype=np.float32),
        ]
    )
    transform = from_origin(400_000.0, 500_000.0, 100.0, 100.0)
    with rasterio.open(
        stack,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=2,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data)
        dst.descriptions = ["os_cover_buildings", "climate_bioclim_bio_2"]

    all_layers = discover_layers(tmp_path)
    assert "terrain_stats_slope" in all_layers
    stack_layers = discover_layers(tmp_path, stack_only=True)
    assert set(stack_layers) == {"os_cover_buildings", "climate_bioclim_bio_2"}
    glob_layers = discover_layers(tmp_path, file_glob="evs-to-model.tif")
    assert set(glob_layers) == set(stack_layers)


def test_identical_layers_pass(tmp_path: Path, variables_configs: list[Path]):
    current = tmp_path / "current"
    rebuilt = tmp_path / "rebuilt"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    name = "os_distance_distance_to_minor_roads"
    _write_raster(current / f"{name}.tif", data, description=name)
    _write_raster(rebuilt / f"{name}.tif", data.copy(), description=name)

    out_csv = tmp_path / "out" / "cmp.csv"
    summary = compare_ev_layers(
        rebuilt,
        current,
        out_csv,
        priority_config_paths=variables_configs,
        difference_maps_dir=tmp_path / "out" / "diffs",
    )
    by_name = {r.layer_name: r for r in summary.layers}
    assert by_name[name].verdict is LayerVerdict.PASS
    assert by_name[name].share_within_tolerance == 1.0
    assert by_name[name].grid_ok is True
    assert Path(by_name[name].difference_map_path).exists()
    assert Path(by_name[name].patch_table_path).exists()
    assert by_name["ceh_landcover_arable_1000m"].verdict is LayerVerdict.MISSING
    assert by_name["climate_bioclim_bio_2"].verdict is LayerVerdict.MISSING
    df = pd.read_csv(out_csv)
    assert list(df["layer_name"])[0] == "os_distance_distance_to_minor_roads"


def test_continuous_within_one_percent_passes(tmp_path: Path):
    name = "climate_stats_temp_ann_avg"
    current = np.linspace(0, 100, 16, dtype=np.float32).reshape(4, 4)
    rebuilt = current + 0.5
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


def test_continuous_local_compact_patch_passes(tmp_path: Path):
    name = "os_distance_distance_to_major_roads"
    current = np.zeros((10, 10), dtype=np.float32)
    current[0, :] = np.linspace(0, 100, 10)
    rebuilt = current.copy()
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
    assert result.share_within_tolerance == pytest.approx(0.96)
    assert result.differences_local is True
    assert result.verdict is LayerVerdict.PASS
    patches = pd.read_csv(result.patch_table_path)
    assert len(patches) == 1
    assert patches.loc[0, "cell_count"] == 4
    assert patches.loc[0, "bbox_fill_ratio"] == pytest.approx(1.0)


def test_sparse_road_shaped_patch_fails_local_check():
    # Plus-sign spanning a large bbox → one component, low fill ratio.
    mismatch = np.zeros((21, 21), dtype=bool)
    mismatch[10, :] = True
    mismatch[:, 10] = True
    jointly = np.ones((21, 21), dtype=bool)
    local, patches, n = analyze_mismatch_patches(mismatch, jointly)
    assert n == 1
    assert patches[0].bbox_fill_ratio < 0.25
    assert local is False


def test_exact_category_mismatch_share(tmp_path: Path):
    name = "ceh_landcover_arable_1000m"
    current = np.ones((5, 5), dtype=np.float32)
    rebuilt = current.copy()
    rebuilt[0, 0] = 2.0

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


def test_source_change_flags_even_when_values_pass(tmp_path: Path):
    name = "os_cover_buildings"
    data = np.ones((4, 4), dtype=np.float32)
    cur_dir = tmp_path / "current"
    reb_dir = tmp_path / "rebuilt"
    _write_raster(cur_dir / f"{name}.tif", data, description=name)
    _write_raster(reb_dir / f"{name}.tif", data.copy(), description=name)

    result = compare_layer_pair(
        name,
        discover_layers(cur_dir)[name],
        discover_layers(reb_dir)[name],
        diff_dir=tmp_path / "diffs",
        source_version_current="OS 2023",
        source_version_rebuilt="OS 2024",
    )
    assert result.share_within_tolerance == 1.0
    assert result.verdict is LayerVerdict.FLAGGED_SOURCE_CHANGE
    assert "source versions differ" in result.notes


def test_source_change_flags_when_values_fail(tmp_path: Path):
    name = "os_cover_buildings"
    current = np.zeros((8, 8), dtype=np.float32)
    rebuilt = np.ones((8, 8), dtype=np.float32)
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


def test_validate_comparison_write_path_refuses_data_evs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "evs").mkdir(parents=True)
    with pytest.raises(ForbiddenEvOutputError):
        validate_comparison_write_path(
            Path("data/evs/layer_comparison.csv"), label="comparison CSV"
        )


def test_cli_compare_layers(tmp_path: Path, variables_configs: list[Path]):
    from typer.testing import CliRunner

    from sdm.cli import app

    name = "os_distance_distance_to_minor_roads"
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    current = tmp_path / "current"
    rebuilt = tmp_path / "rebuilt"
    _write_raster(current / f"{name}.tif", data, description=name)
    _write_raster(rebuilt / f"{name}.tif", data.copy(), description=name)
    out_csv = tmp_path / "cmp" / "cmp.csv"
    hooks = tmp_path / "cmp" / "hooks.csv"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compare-layers",
            str(rebuilt),
            str(current),
            "-o",
            str(out_csv),
            "--priority-config",
            str(variables_configs[0]),
            "--priority-config",
            str(variables_configs[1]),
            "--difference-maps-dir",
            str(tmp_path / "cmp" / "diffs"),
            "--model-hooks-csv",
            str(hooks),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_csv.exists()
    assert hooks.exists()
    hooks_df = pd.read_csv(hooks)
    assert hooks_df.loc[0, "verdict"] == "not_run"
