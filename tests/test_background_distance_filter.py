"""Behaviour pins for background distance filtering (library-reuse #47 / #51 item 2).

Asserts which background IDs survive known presence geometry and d_min/d_max
rings. Does not time cKDTree vs shapely or re-implement the nearest-distance path.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from sdm.commands.modelling.train_sdm_models import filter_background_by_distance


def _points_gdf(xy, crs="EPSG:27700", **cols):
    geom = [Point(x, y) for x, y in xy]
    data = {"geometry": geom}
    data.update(cols)
    return gpd.GeoDataFrame(data, crs=crs)


class TestFilterBackgroundByDistance:
    """Behaviour pins for min/max distance filtering of background points."""

    @pytest.fixture
    def presence(self):
        return _points_gdf([(0.0, 0.0), (1000.0, 0.0)], id=["p0", "p1"])

    @pytest.fixture
    def background(self):
        # Distances to nearest presence (hand-derived):
        # b0 (0,0)->0, b1 (200,0)->200, b2 (600,0)->400 (to 1000),
        # b3 (2000,0)->1000, b4 (5000,0)->4000
        return _points_gdf(
            [
                (0.0, 0.0),
                (200.0, 0.0),
                (600.0, 0.0),
                (2000.0, 0.0),
                (5000.0, 0.0),
            ],
            bg_id=["b0", "b1", "b2", "b3", "b4"],
        )

    def test_empty_presence_returns_all_background(self, background):
        empty = _points_gdf([])
        out = filter_background_by_distance(background, empty, d_min=500)
        assert list(out["bg_id"]) == ["b0", "b1", "b2", "b3", "b4"]

    def test_empty_background_returns_empty(self, presence):
        empty = _points_gdf([], bg_id=[])
        out = filter_background_by_distance(empty, presence, d_min=0.0)
        assert len(out) == 0

    def test_d_min_excludes_near_presence(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=500.0, d_max=np.inf
        )
        assert list(out["bg_id"]) == ["b3", "b4"]

    def test_d_max_excludes_far_presence(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=0.0, d_max=1500.0
        )
        assert list(out["bg_id"]) == ["b0", "b1", "b2", "b3"]

    def test_ring_filter(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=300.0, d_max=1500.0
        )
        assert list(out["bg_id"]) == ["b2", "b3"]

    def test_inclusive_bounds_keep_exact_d_min_and_d_max(self, presence, background):
        # b3 is exactly 1000 from nearest presence; b2 is 400
        out = filter_background_by_distance(
            background, presence, d_min=400.0, d_max=1000.0
        )
        assert list(out["bg_id"]) == ["b2", "b3"]
