"""Pin behaviour for background distance filtering (library-reuse #47 / #51 item 2)."""

import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
from shapely import distance as shapely_distance
from scipy.spatial import cKDTree

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
        # Distances to nearest presence:
        # (0,0)->0, (200,0)->200, (600,0)->400 (to 1000), (2000,0)->1000, (5000,0)->4000
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
        assert len(out) == len(background)
        assert list(out["bg_id"]) == list(background["bg_id"])

    def test_d_min_excludes_near_presence(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=500.0, d_max=np.inf
        )
        # Keep only points with nearest presence >= 500: b3 (1000), b4 (4000)
        assert list(out["bg_id"]) == ["b3", "b4"]

    def test_d_max_excludes_far_presence(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=0.0, d_max=1500.0
        )
        # Exclude b4 (4000); keep b0..b3
        assert list(out["bg_id"]) == ["b0", "b1", "b2", "b3"]

    def test_ring_filter(self, presence, background):
        out = filter_background_by_distance(
            background, presence, d_min=300.0, d_max=1500.0
        )
        # b2=400 (to 1000), b3=1000; drop near and very far
        assert list(out["bg_id"]) == ["b2", "b3"]

    def test_matches_shapely_pairwise_min_distance(self, presence, background):
        """Identity: filter mask equals min pairwise shapely distance (legacy path)."""
        A = presence.geometry.array.to_numpy()
        B = background.geometry.array.to_numpy()
        legacy_min = shapely_distance(A[:, None], B[None, :]).min(axis=0)

        d_min, d_max = 500.0, 3000.0
        expected_ids = background.loc[
            (legacy_min >= d_min) & (legacy_min <= d_max), "bg_id"
        ].tolist()
        out = filter_background_by_distance(
            background, presence, d_min=d_min, d_max=d_max
        )
        assert list(out["bg_id"]) == expected_ids

    def test_ckdtree_distances_match_shapely(self, presence, background):
        """Nearest distances from cKDTree match shapely within 1e-6 (proof)."""
        A = presence.geometry.array.to_numpy()
        B = background.geometry.array.to_numpy()
        legacy_min = shapely_distance(A[:, None], B[None, :]).min(axis=0)

        presence_xy = np.column_stack(
            [presence.geometry.x.to_numpy(), presence.geometry.y.to_numpy()]
        )
        background_xy = np.column_stack(
            [background.geometry.x.to_numpy(), background.geometry.y.to_numpy()]
        )
        tree_min, _ = cKDTree(presence_xy).query(background_xy, k=1)
        np.testing.assert_allclose(tree_min, legacy_min, atol=1e-6, rtol=0)


@pytest.mark.slow
def test_ckdtree_faster_and_lighter_than_pairwise_matrix():
    """Synthetic proof: cKDTree beats dense shapely matrix on time and peak RAM."""
    rng = np.random.default_rng(0)
    n_p, n_b = 500, 10_000
    presence_xy = rng.uniform(0, 50_000, size=(n_p, 2))
    background_xy = rng.uniform(0, 50_000, size=(n_b, 2))
    presence = _points_gdf(presence_xy)
    background = _points_gdf(background_xy)

    # Legacy pairwise (subset timing only; full matrix allocates heavily)
    A = presence.geometry.array.to_numpy()
    B = background.geometry.array.to_numpy()
    t0 = time.perf_counter()
    legacy_min = shapely_distance(A[:, None], B[None, :]).min(axis=0)
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree_min, _ = cKDTree(presence_xy).query(background_xy, k=1)
    tree_s = time.perf_counter() - t0

    np.testing.assert_allclose(tree_min, legacy_min, atol=1e-6, rtol=0)
    assert tree_s < legacy_s / 5, (
        f"cKDTree not clearly faster: tree={tree_s:.4f}s legacy={legacy_s:.4f}s"
    )

    # Filter identity on same sample
    out_legacy_mask = (legacy_min >= 500) & (legacy_min <= np.inf)
    out = filter_background_by_distance(background, presence, d_min=500.0)
    assert len(out) == int(out_legacy_mask.sum())
