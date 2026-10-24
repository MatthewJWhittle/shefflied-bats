"""Behaviour pins for region density weighting (rasterize path, #47 / #51 item 5).

Expected rasters use known cell-to-region membership on a synthetic 4×4 grid
and the documented ``normalise_to_distribution`` weighting — not a legacy clip
reimplementation or a speed assertion.
"""

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import box

from sdm.occurrence.sampling import (
    normalise_to_distribution,
    weight_density_array_by_regions,
)


def _grid_with_transform(data: np.ndarray) -> xr.DataArray:
    """Integer cell-centre coords with matching Affine (pixel edges at ±0.5)."""
    n_y, n_x = data.shape
    da = xr.DataArray(
        data.astype(float),
        coords={"y": np.arange(n_y, dtype=float), "x": np.arange(n_x, dtype=float)},
        dims=["y", "x"],
    )
    da = da.rio.write_crs("EPSG:27700")
    da.rio.write_transform(Affine(1.0, 0.0, -0.5, 0.0, 1.0, -0.5), inplace=True)
    return da


def _left_right_regions() -> gpd.GeoDataFrame:
    """Non-overlapping left/right halves of a 4×4 cell-centre grid."""
    return gpd.GeoDataFrame(
        geometry=[
            box(-0.1, -0.1, 1.9, 3.9),
            box(1.9, -0.1, 3.9, 3.9),
        ],
        crs="EPSG:27700",
    )


@pytest.fixture
def density_4x4() -> xr.DataArray:
    # [[1..4], [5..8], [9..12], [13..16]] — left cols sum to 60, right to 76
    return _grid_with_transform(np.arange(1, 17, dtype=float).reshape(4, 4))


def test_weighted_density_matches_known_left_right_split(density_4x4):
    """Left/right region sums → normalised weights × density (known raster)."""
    regions = _left_right_regions()
    region_sums = normalise_to_distribution(
        np.array([60.0, 76.0, 0.0]), mean=1.0, std=1.0
    )
    expected = np.asarray(density_4x4.values, dtype=float).copy()
    expected[:, 0:2] *= region_sums[0]
    expected[:, 2:4] *= region_sums[1]

    got = weight_density_array_by_regions(density_4x4, regions, weight=1.0)
    np.testing.assert_allclose(got.values, expected, rtol=0, atol=1e-9)


def test_reverse_weights_and_std_match_known_raster(density_4x4):
    regions = _left_right_regions()
    region_sums = normalise_to_distribution(
        np.array([60.0, 76.0, 0.0]), mean=1.0, std=1.5
    )
    region_sums = 1.0 - region_sums + 1
    expected = np.asarray(density_4x4.values, dtype=float).copy()
    expected[:, 0:2] *= region_sums[0]
    expected[:, 2:4] *= region_sums[1]

    got = weight_density_array_by_regions(
        density_4x4, regions, weight=1.5, reverse_weights=True
    )
    np.testing.assert_allclose(got.values, expected, rtol=0, atol=1e-9)


def test_background_region_receives_its_own_weight(density_4x4):
    """Cells outside all polygons use the background weight slot."""
    regions = gpd.GeoDataFrame(
        geometry=[box(-0.1, -0.1, 1.9, 3.9)],
        crs="EPSG:27700",
    )
    region_sums = normalise_to_distribution(np.array([60.0, 76.0]), mean=1.0, std=1.0)
    expected = np.asarray(density_4x4.values, dtype=float).copy()
    expected[:, 0:2] *= region_sums[0]
    expected[:, 2:4] *= region_sums[1]

    got = weight_density_array_by_regions(density_4x4, regions, weight=1.0)
    np.testing.assert_allclose(got.values, expected, rtol=0, atol=1e-9)


def test_overlapping_regions_last_geometry_wins(density_4x4):
    """When polygons overlap, the later region in ``regions`` owns the cell."""
    regions = gpd.GeoDataFrame(
        geometry=[
            box(-0.1, -0.1, 3.9, 3.9),  # whole grid first
            box(1.9, -0.1, 3.9, 3.9),  # right half last → wins there
        ],
        crs="EPSG:27700",
    )
    region_sums = normalise_to_distribution(
        np.array([60.0, 76.0, 0.0]), mean=1.0, std=1.0
    )
    expected = np.asarray(density_4x4.values, dtype=float).copy()
    expected[:, 0:2] *= region_sums[0]
    expected[:, 2:4] *= region_sums[1]

    got = weight_density_array_by_regions(density_4x4, regions, weight=1.0)
    np.testing.assert_allclose(got.values, expected, rtol=0, atol=1e-9)


def test_empty_regions_raises(density_4x4):
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
    with pytest.raises(ValueError, match="at least one"):
        weight_density_array_by_regions(density_4x4, empty)


def test_reserved_region_id_column_raises(density_4x4):
    regions = _left_right_regions()
    regions = regions.copy()
    regions["_region_id"] = ["a", "b"]
    with pytest.raises(ValueError, match="_region_id"):
        weight_density_array_by_regions(density_4x4, regions)
