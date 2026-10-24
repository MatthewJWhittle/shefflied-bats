"""Pins for #51 orphan API cleanup (#47 items 3, 4, 7).

Deletion absence checks stay as contract pins. Remaining helpers are exercised
on synthetic values / a tiny EV GeoTIFF so the suite can fail on wrong behaviour.
"""

import importlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from sdm.data.landcover import (
    create_binary_raster_from_category,
    define_broad_habitat_categories,
    get_ceh_land_cover_codes_v2023,
)
from sdm.data.processing import annotate_points
import xarray as xr


@pytest.mark.parametrize(
    "module_name,attr",
    [
        ("sdm.data.processing.core", "merge_environmental_layers"),
        ("sdm.data.processing.core", "process_occurrence_data"),
        ("sdm.data.processing.core", "process_background_data"),
        ("sdm.data.processing.core", "extract_environmental_data"),
        ("sdm.data", "merge_environmental_layers"),
        ("sdm.raster.utils", "aggregate_raster"),
        ("sdm.raster.utils", "tile_bounding_box"),
        ("sdm.raster.utils", "points_to_grid_squares"),
        ("sdm.raster.utils", "align_to_grid"),
        ("sdm.raster.utils", "calculate_offset"),
    ],
)
def test_deleted_orphan_symbols_absent(module_name, attr):
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, attr), f"{module_name}.{attr} should be removed"


def test_ceh_processing_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sdm.data.ceh_processing")


def test_landcover_ceh_codes_and_habitat_map_values():
    codes = get_ceh_land_cover_codes_v2023()
    assert codes["1"] == "Broadleaved woodland"
    assert codes["20"] == "Urban"
    assert codes["21"] == "Suburban"
    assert len(codes) == 21

    cats = define_broad_habitat_categories()
    assert cats["Grassland"] == [
        "Neutral grassland",
        "Calcareous grassland",
        "Acid grassland",
    ]
    assert "Bog" in cats["Wetland"]
    assert "Saltmarsh" in cats["Marine, Littoral"]


def test_create_binary_raster_from_category_matches_mask():
    source = xr.DataArray(
        np.array([[1, 2], [2, 1]], dtype=np.int16),
        dims=("y", "x"),
    )
    out = create_binary_raster_from_category(source, category_value=2, output_var_name="c2")
    np.testing.assert_array_equal(
        out["c2"].values,
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_annotate_points_samples_known_raster_values(tmp_path: Path):
    """Remaining annotate_points path returns EV values at point locations."""
    raster_path = tmp_path / "ev.tif"
    # 2×2, 10 m cells, origin top-left (0,20); centres at (5,15), (15,15), (5,5), (15,5)
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    transform = from_origin(0.0, 20.0, 10.0, 10.0)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    bats = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=[Point(5.0, 15.0)],
        crs="EPSG:27700",
    )
    background = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(15.0, 5.0)],
        crs="EPSG:27700",
    )
    bats_out, bg_out = annotate_points(
        bats, background, raster_path, ev_columns=["elev"]
    )
    assert float(bats_out["elev"].iloc[0]) == pytest.approx(10.0)
    assert float(bg_out["elev"].iloc[0]) == pytest.approx(40.0)
