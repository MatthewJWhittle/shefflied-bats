"""Identity + timing for in-memory feature cover rasterise (#51 item 6)."""

import time
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Polygon, box

from sdm.raster.processing import calculate_feature_cover
from sdm.raster.utils import rasterise_gdf, rasterise_gdf_to_dataarray


def _legacy_tempfile_cover(gdf, resolution, bbox):
    with NamedTemporaryFile(suffix=".tif") as f:
        rasterise_gdf(gdf, resolution=resolution, output_file=f.name, bbox=bbox)
        return rxr.open_rasterio(f.name).squeeze(drop=True)


def test_inmemory_rasterise_matches_tempfile_geotiff():
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(10, 10), (40, 10), (40, 40), (10, 40)]),
                Polygon([(60, 60), (90, 60), (90, 90), (60, 90)]),
            ]
        },
        crs="EPSG:27700",
    )
    bbox = (0.0, 0.0, 100.0, 100.0)
    resolution = 10.0
    legacy = _legacy_tempfile_cover(buildings, resolution, bbox)
    modern = rasterise_gdf_to_dataarray(buildings, resolution, bbox=bbox)

    assert modern.rio.transform() == legacy.rio.transform()
    np.testing.assert_array_equal(modern.values, legacy.values)


def test_calculate_feature_cover_matches_legacy_tempfile_path():
    """End-to-end cover equals tempfile→open→coarsen on a small synthetic AOI."""
    boundary = gpd.GeoDataFrame(
        geometry=[box(0, 0, 200, 200)],
        crs="EPSG:27700",
    )
    features = {
        "buildings": gpd.GeoDataFrame(
            {"geometry": [Polygon([(20, 20), (80, 20), (80, 80), (20, 80)])]},
            crs="EPSG:27700",
        )
    }
    modern = calculate_feature_cover(
        features, boundary, target_resolution=20, base_resolution=10
    )

    # Legacy path: tempfile + open + coarsen (pre-swap calculate_cover_for_gdf core)
    gdf = features["buildings"]
    bbox = tuple(boundary.unary_union.bounds)
    with NamedTemporaryFile(suffix=".tif") as f:
        rasterise_gdf(gdf, resolution=10, output_file=f.name, bbox=bbox)
        cover_raster = rxr.open_rasterio(f.name).squeeze(drop=True)
        legacy_area = cover_raster.coarsen(x=2, y=2, boundary="trim").sum()
    # Clip like production
    legacy = legacy_area.rename("buildings").to_dataset()
    legacy = legacy.rio.write_crs(boundary.crs)
    legacy = legacy.rio.clip(boundary.geometry.values, boundary.crs, drop=False)

    modern_vals = np.asarray(modern["buildings"].values, dtype=float)
    legacy_vals = np.asarray(legacy["buildings"].values, dtype=float)
    # Align NaN mask after clip
    np.testing.assert_allclose(
        np.nan_to_num(modern_vals, nan=-999),
        np.nan_to_num(legacy_vals, nan=-999),
        rtol=0,
        atol=0,
    )


@pytest.mark.slow
def test_inmemory_rasterise_faster_than_tempfile_roundtrip():
    rng = np.random.default_rng(0)
    polys = []
    for _ in range(200):
        x, y = rng.uniform(0, 450, size=2)
        polys.append(box(x, y, x + 5, y + 5))
    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:27700")
    bbox = (0.0, 0.0, 500.0, 500.0)
    resolution = 10.0

    t0 = time.perf_counter()
    legacy = _legacy_tempfile_cover(gdf, resolution, bbox)
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    modern = rasterise_gdf_to_dataarray(gdf, resolution, bbox=bbox)
    modern_s = time.perf_counter() - t0

    np.testing.assert_array_equal(modern.values, legacy.values)
    assert modern_s < legacy_s, (
        f"in-memory not faster: modern={modern_s:.4f}s legacy={legacy_s:.4f}s"
    )
