"""Behaviour pins for in-memory feature cover rasterise (#47 / #51 item 6).

Expected masks use known polygon↔cell-centre coverage on a snapped grid.
Does not re-open a tempfile GeoTIFF as the oracle or assert timing.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

from sdm.raster.processing import calculate_feature_cover
from sdm.raster.utils import rasterise_gdf_to_dataarray


def test_inmemory_rasterise_matches_known_cell_mask():
    """Single small polygon covers exactly the centre cell of a 3×3 @ 10 m grid."""
    features = gpd.GeoDataFrame(
        {"geometry": [Polygon([(12, 12), (18, 12), (18, 18), (12, 18)])]},
        crs="EPSG:27700",
    )
    cover = rasterise_gdf_to_dataarray(
        features, resolution=10.0, bbox=(0.0, 0.0, 30.0, 30.0)
    )

    np.testing.assert_array_equal(cover.x.values, [5.0, 15.0, 25.0])
    np.testing.assert_array_equal(cover.y.values, [25.0, 15.0, 5.0])
    expected = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(cover.values, expected)


def test_inmemory_rasterise_two_buildings_known_mask():
    buildings = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(10, 10), (40, 10), (40, 40), (10, 40)]),
                Polygon([(60, 60), (90, 60), (90, 90), (60, 90)]),
            ]
        },
        crs="EPSG:27700",
    )
    cover = rasterise_gdf_to_dataarray(
        buildings, resolution=10.0, bbox=(0.0, 0.0, 100.0, 100.0)
    )
    # Hand-checked against cell centres at 5..95; 1s only under the two footprints
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(cover.values, expected)
    assert int(cover.values.sum()) == 18


def test_calculate_feature_cover_known_coarsened_counts():
    """10 m cover coarsened to 20 m: SW 20×20 block has 4 covered fine cells."""
    boundary = gpd.GeoDataFrame(
        geometry=[box(0, 0, 40, 40)],
        crs="EPSG:27700",
    )
    features = {
        "buildings": gpd.GeoDataFrame(
            {"geometry": [Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])]},
            crs="EPSG:27700",
        )
    }
    cover = calculate_feature_cover(
        features, boundary, target_resolution=20, base_resolution=10
    )

    np.testing.assert_array_equal(cover.x.values, [10.0, 30.0])
    np.testing.assert_array_equal(cover.y.values, [30.0, 10.0])
    expected = np.array([[0.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(
        np.asarray(cover["buildings"].values, dtype=float),
        expected,
        rtol=0,
        atol=0,
    )
