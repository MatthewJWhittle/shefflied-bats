"""Synthetic tests for clipping study boundaries to land (MHW).

Fixtures are invented geometry only — no licensed rasters or real sites.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from sdm.data.coastline_download import MHW_CODE
from sdm.data.land_clip import (
    approximate_sea_from_mhw,
    clip_boundary_to_land,
    load_boundary_land_clipped,
)
from sdm.utils.io import load_boundary_and_transform


def _land_gdf() -> gpd.GeoDataFrame:
    """Rectangular land block; east edge is the 'coast'."""
    return gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:27700",
    )


def _mhw_gdf() -> gpd.GeoDataFrame:
    """Mean high water along x=1000 (east edge of land)."""
    return gpd.GeoDataFrame(
        {"CODE": [MHW_CODE]},
        geometry=[LineString([(1000, -500), (1000, 1500)])],
        crs="EPSG:27700",
    )


class TestApproximateSeaFromMhw:
    def test_sea_lies_east_of_coast_not_on_land(self) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf()
        sea = approximate_sea_from_mhw(land, mhw, sea_buffer_m=500)

        assert sea.contains(Point(1200, 500))
        assert not sea.contains(Point(500, 500))
        # Landward side outside the land polygon must not be treated as sea
        # (otherwise inland study buffers would be clipped away).
        assert not sea.contains(Point(-150, 500))
        assert land.union_all().intersection(sea).area < 1.0

    def test_empty_mhw_returns_empty_sea(self) -> None:
        """Inland AOIs with no MHW in range: empty sea, boundary unchanged."""
        land = _land_gdf()
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        sea = approximate_sea_from_mhw(land, empty, sea_buffer_m=100)
        assert sea.is_empty


class TestClipBoundaryToLand:
    def test_buffered_boundary_loses_seaward_lobe(self) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf()
        buffered = land.copy()
        buffered["geometry"] = buffered.buffer(200)

        # Before clip, buffer reaches into 'sea' (x > 1000).
        assert buffered.union_all().bounds[2] > 1000

        clipped = clip_boundary_to_land(
            buffered,
            mhw,
            land_gdf=land,
            sea_buffer_m=400,
        )
        union = clipped.union_all()
        assert union.contains(Point(500, 500))
        # Inland buffer (west) retained.
        assert union.contains(Point(-150, 500))
        # Seaward buffer removed.
        assert not union.contains(Point(1150, 500))
        assert union.bounds[2] <= 1000 + 1e-6

    def test_unbuffered_land_unchanged_inland(self) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf()
        clipped = clip_boundary_to_land(land, mhw, land_gdf=land, sea_buffer_m=400)
        # Inland area preserved (within dissolve/buffer(0) tolerance).
        assert abs(clipped.union_all().area - land.union_all().area) / land.union_all().area < 0.02

    def test_crs_mismatch_reprojects_mhw(self) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf().to_crs("EPSG:4326")
        buffered = land.copy()
        buffered["geometry"] = buffered.buffer(200)
        clipped = clip_boundary_to_land(
            buffered,
            mhw,
            land_gdf=land,
            sea_buffer_m=400,
        )
        assert clipped.crs == land.crs
        assert not clipped.union_all().contains(Point(1150, 500))


class TestLoadBoundaryLandClipped:
    def test_file_roundtrip_with_synthetic_coast(self, tmp_path: Path) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf()
        boundary_path = tmp_path / "boundary.geojson"
        coast_path = tmp_path / "mhw.shp"
        land.to_file(boundary_path, driver="GeoJSON")
        mhw.to_file(coast_path)

        result = load_boundary_land_clipped(
            boundary_path,
            coastline_path=coast_path,
            buffer_distance=200,
            target_crs="EPSG:27700",
        )
        union = result.union_all()
        assert union.contains(Point(500, 500))
        assert union.contains(Point(-150, 500))
        assert not union.contains(Point(1150, 500))


class TestLoadBoundaryAndTransformLandClip:
    def test_clip_enabled_removes_sea_from_grid_bounds(self, tmp_path: Path) -> None:
        land = _land_gdf()
        mhw = _mhw_gdf()
        boundary_path = tmp_path / "boundary.geojson"
        coast_path = tmp_path / "mhw.shp"
        land.to_file(boundary_path, driver="GeoJSON")
        mhw.to_file(coast_path)

        spatial = {"resolution": 100, "crs": "EPSG:27700"}
        with patch("sdm.utils.io.load_spatial_config", return_value=spatial):
            boundary, _transform, bounds, _cfg = load_boundary_and_transform(
                boundary_path,
                buffer_distance=200,
                clip_to_land=True,
                coastline_path=coast_path,
            )

        # Grid east edge should not extend past the coast.
        _minx, _miny, maxx, _maxy = bounds
        assert maxx <= 1100  # one cell of alignment slack at 100 m
        assert not boundary.union_all().contains(Point(1150, 500))

    def test_clip_disabled_keeps_seaward_buffer(self, tmp_path: Path) -> None:
        land = _land_gdf()
        boundary_path = tmp_path / "boundary.geojson"
        land.to_file(boundary_path, driver="GeoJSON")

        spatial = {"resolution": 100, "crs": "EPSG:27700"}
        with patch("sdm.utils.io.load_spatial_config", return_value=spatial):
            boundary, _transform, _bounds, _cfg = load_boundary_and_transform(
                boundary_path,
                buffer_distance=200,
                clip_to_land=False,
            )

        assert boundary.union_all().contains(Point(1150, 500))
