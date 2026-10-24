"""Identity + timing proof for region density weighting via rasterize (#51 item 5)."""

import time

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


def _legacy_clip_weight(
    density_array: xr.DataArray,
    regions: gpd.GeoDataFrame,
    weight: float = 1.0,
    reverse_weights: bool = False,
) -> xr.DataArray:
    """Pre-swap per-region ``rio.clip`` implementation (reference only)."""
    regions = regions.copy()
    regions["_region_id"] = [f"region_{i}" for i in range(len(regions))]

    ones = xr.ones_like(density_array).squeeze()
    ones.rio.write_nodata(0, inplace=True)
    region_masks = []
    region_ids = []
    for _, row in regions.iterrows():
        mask = ones.rio.clip([row.geometry], density_array.rio.crs, drop=False)
        region_masks.append(mask)
        region_ids.append(row["_region_id"])

    region_masks = xr.concat(region_masks, dim="region_id")
    background_region = region_masks.sum(dim="region_id") == 0
    region_masks = xr.concat([region_masks, background_region], dim="region_id")
    region_masks.coords["region_id"] = region_ids + ["background"]
    region_masks.rio.write_nodata(0, inplace=True)

    region_sums = (density_array * region_masks).sum(dim=["x", "y"])
    region_sums = normalise_to_distribution(region_sums, mean=1.0, std=weight)
    if reverse_weights:
        region_sums = 1.0 - region_sums + 1

    weight_array = xr.zeros_like(density_array)
    for region_id, region_weight in zip(region_masks.coords["region_id"], region_sums):
        weight_array = weight_array + (
            region_masks.sel(region_id=region_id) * region_weight
        )
    return density_array * weight_array


def _synthetic_grid(n: int = 40, seed: int = 0) -> xr.DataArray:
    """Grid with cell-centre coords and matching south-up Affine (like sampling fixtures)."""
    rng = np.random.default_rng(seed)
    data = rng.random((n, n))
    coords = np.arange(n, dtype=float)
    da = xr.DataArray(data, coords={"y": coords, "x": coords}, dims=["y", "x"])
    da = da.rio.write_crs("EPSG:27700")
    # Pixel edges at -0.5 .. n-0.5 so centres sit on integer coords.
    da.rio.write_transform(Affine(1.0, 0.0, -0.5, 0.0, 1.0, -0.5), inplace=True)
    return da


def _nonoverlapping_regions(n_regions: int, grid_n: int) -> gpd.GeoDataFrame:
    # Tile the grid into non-overlapping boxes (matches production region use).
    side = int(np.ceil(np.sqrt(n_regions)))
    cell = grid_n / side
    geoms = []
    for i in range(n_regions):
        r, c = divmod(i, side)
        geoms.append(box(c * cell, r * cell, (c + 1) * cell, (r + 1) * cell))
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:27700")


def test_rasterize_matches_legacy_clip_nonoverlapping():
    density = _synthetic_grid(48, seed=1)
    regions = _nonoverlapping_regions(9, 48)
    legacy = _legacy_clip_weight(density, regions, weight=1.0)
    modern = weight_density_array_by_regions(density, regions, weight=1.0)
    np.testing.assert_allclose(modern.values, legacy.values, rtol=1e-6, atol=1e-6)


def test_rasterize_matches_legacy_with_reverse_and_weight():
    density = _synthetic_grid(32, seed=2)
    regions = _nonoverlapping_regions(4, 32)
    legacy = _legacy_clip_weight(
        density, regions, weight=1.5, reverse_weights=True
    )
    modern = weight_density_array_by_regions(
        density, regions, weight=1.5, reverse_weights=True
    )
    np.testing.assert_allclose(modern.values, legacy.values, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_rasterize_faster_than_clip_loop():
    density = _synthetic_grid(200, seed=3)
    regions = _nonoverlapping_regions(20, 200)

    t0 = time.perf_counter()
    legacy = _legacy_clip_weight(density, regions)
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    modern = weight_density_array_by_regions(density, regions)
    modern_s = time.perf_counter() - t0

    np.testing.assert_allclose(modern.values, legacy.values, rtol=1e-6, atol=1e-6)
    assert modern_s < legacy_s / 5, (
        f"rasterize not clearly faster: modern={modern_s:.4f}s legacy={legacy_s:.4f}s"
    )
