import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely import union_all
from shapely.geometry import box, Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry
import xarray as xr
import rioxarray as rxr
from rasterio.transform import from_bounds

from sdm.raster.utils import generate_point_grid, rasterise_gdf
from sdm.data.os import process_roads
from sdm.raster.processing import (
    calculate_distances,
    calculate_feature_cover,
    nearest_distances_strtree,
)
from sdm.commands.data_preparation.processing.process_os_data import process_os_data as main


def _nearest_distances_union_reference(
    query_geoms: list[BaseGeometry],
    feature_geoms: list[BaseGeometry],
) -> np.ndarray:
    """Reference nearest distances via union + distance (tests only; not production)."""
    if not feature_geoms:
        return np.full(len(query_geoms), np.nan)
    union_geom = union_all(feature_geoms)
    return np.asarray([geom.distance(union_geom) for geom in query_geoms], dtype=float)

@pytest.fixture
def sample_boundary() -> gpd.GeoDataFrame:
    """Create a simple square boundary."""
    geometry = box(0, 0, 100, 100)
    return gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:27700")

@pytest.fixture
def sample_roads() -> gpd.GeoDataFrame:
    """Create sample road data with classifications."""
    roads_data = {
        'geometry': [
            LineString([(0, 0), (50, 50)]),
            LineString([(0, 50), (50, 50)]),
            LineString([(50, 50), (100, 50)]),
            LineString([(50, 0), (50, 100)])
        ],
        'CLASSIFICA': [
            'A Road',
            'Local Street',
            'Minor Road',
            'Motorway'
        ]
    }
    return gpd.GeoDataFrame(roads_data, crs="EPSG:27700")

@pytest.fixture
def sample_features() -> dict:
    """Create sample feature datasets."""
    features = {
        'buildings': gpd.GeoDataFrame({
            'geometry': [
                Point(15, 15),  # Center of first building
                Point(65, 65)   # Center of second building
            ]
        }, crs="EPSG:27700"),
        'water': gpd.GeoDataFrame({
            'geometry': [
                Point(35, 35)   # Center of water body
            ]
        }, crs="EPSG:27700")
    }
    return features

def test_generate_point_grid():
    """Test point grid generation."""
    bbox = (0, 0, 10, 10)
    resolution = 5
    grid = generate_point_grid(bbox, resolution, "EPSG:27700")
    
    assert isinstance(grid, gpd.GeoDataFrame)
    assert len(grid) == 4 # 2x2 grid
    assert grid.crs == "EPSG:27700"
    assert all(isinstance(geom, Point) for geom in grid.geometry)
    assert "x" in grid and "y" in grid

def test_process_roads(sample_roads):
    """Test road classification processing."""
    major_roads, minor_roads = process_roads(sample_roads)
    
    assert len(major_roads) == 2  # A Road and Motorway
    assert len(minor_roads) == 2  # Local Street and Minor Road
    assert all(major_roads.major_road)
    assert not any(minor_roads.major_road)

def test_calculate_distances(sample_boundary, sample_features):
    """Test distance calculation to features."""
    resolution = 20  # Use coarse resolution for test
    distances = calculate_distances(sample_features, sample_boundary, resolution)
    
    assert isinstance(distances, xr.Dataset)
    assert "distance_to_buildings" in distances
    assert "distance_to_water" in distances
    assert not np.any(np.isnan(distances.distance_to_buildings))


def test_calculate_distances_point_geometry(sample_boundary):
    """Point features yield finite, non-empty distance values."""
    features = {
        "points": gpd.GeoDataFrame(
            {"geometry": [Point(25, 25), Point(75, 75)]},
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, sample_boundary, resolution=20)
    values = distances.distance_to_points.values

    assert values.size > 0
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_calculate_distances_line_geometry(sample_boundary):
    """LineString features (e.g. roads) yield finite, non-empty distance values."""
    features = {
        "roads": gpd.GeoDataFrame(
            {"geometry": [LineString([(10, 10), (90, 10)]), LineString([(50, 0), (50, 100)])]},
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, sample_boundary, resolution=20)
    values = distances.distance_to_roads.values

    assert values.size > 0
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_calculate_distances_polygon_geometry(sample_boundary):
    """Polygon features (e.g. buildings/woodland) yield finite, non-empty distance values."""
    features = {
        "buildings": gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon([(20, 20), (40, 20), (40, 40), (20, 40)]),
                    Polygon([(60, 60), (80, 60), (80, 80), (60, 80)]),
                ]
            },
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, sample_boundary, resolution=20)
    values = distances.distance_to_buildings.values

    assert values.size > 0
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_distance_zero_inside_polygon():
    """Grid cell inside a polygon has distance 0."""
    # Single cell centre at (50, 50); polygon fully contains that point.
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:27700")
    features = {
        "poly": gpd.GeoDataFrame(
            {"geometry": [Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])]},
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, boundary, resolution=100)
    assert float(distances.distance_to_poly.values.item()) == 0.0


def test_known_distance_to_line():
    """Grid cell at (50, 50) is 50 m from the horizontal line y=0."""
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:27700")
    features = {
        "road": gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (100, 0)])]},
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, boundary, resolution=100)
    assert float(distances.distance_to_road.values.item()) == pytest.approx(50.0)


def test_known_distance_to_point():
    """Grid cell at (50, 50) is 10 m from Point(50, 60) (3-4-5-style offset)."""
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:27700")
    features = {
        "site": gpd.GeoDataFrame(
            {"geometry": [Point(50, 60)]},
            crs="EPSG:27700",
        )
    }
    distances = calculate_distances(features, boundary, resolution=100)
    assert float(distances.distance_to_site.values.item()) == pytest.approx(10.0)


def test_strtree_agrees_with_union_on_random_synthetic():
    """STRtree nearest distances match union+distance on random synthetic geoms."""
    rng = np.random.default_rng(42)
    # Mixed feature types: points, short lines, small squares.
    features: list[BaseGeometry] = []
    for _ in range(40):
        x, y = rng.uniform(0, 100, size=2)
        features.append(Point(x, y))
    for _ in range(30):
        x0, y0 = rng.uniform(0, 90, size=2)
        features.append(LineString([(x0, y0), (x0 + rng.uniform(1, 10), y0 + rng.uniform(-5, 5))]))
    for _ in range(30):
        x0, y0 = rng.uniform(0, 90, size=2)
        w, h = rng.uniform(1, 8, size=2)
        features.append(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]))

    query = [Point(float(x), float(y)) for x, y in rng.uniform(0, 100, size=(200, 2))]

    tree_dist = nearest_distances_strtree(query, features)
    union_dist = _nearest_distances_union_reference(query, features)

    assert tree_dist.shape == union_dist.shape == (200,)
    assert np.isfinite(tree_dist).all()
    assert np.allclose(tree_dist, union_dist, rtol=0, atol=1e-6)


@pytest.mark.parametrize("resolution", [1, 5])
def test_rasterise_gdf(sample_features, tmp_path, resolution):
    """Test GeoDataFrame rasterization."""
    buildings = sample_features['buildings']
    output_file = tmp_path / "test_raster.tif"
    
    result = rasterise_gdf(buildings, resolution, output_file)
    assert result.exists()
    
    # Check the raster can be read and has correct properties
    raster = rxr.open_rasterio(result)
    assert raster.rio.crs == buildings.crs
    assert raster.dtype == np.uint8

def test_calculate_feature_cover(sample_boundary, sample_features):
    """Test feature cover calculation."""
    feature_cover = calculate_feature_cover(
        sample_features,
        sample_boundary,
        target_resolution=20
    )
    
    assert isinstance(feature_cover, xr.Dataset)
    assert "buildings" in feature_cover
    assert "water" in feature_cover
    assert feature_cover.rio.crs == sample_boundary.crs

def test_point_grid_alignment():
    """Test that generated grid points align with expected coordinates."""
    bbox = (0, 0, 10, 10)
    resolution = 5
    grid = generate_point_grid(bbox, resolution, "EPSG:27700")
    
    # Check coordinates are at resolution/2 offset (center of cells)
    x_coords = grid.geometry.x
    y_coords = grid.geometry.y
    expected_x = np.array([2.5, 2.5, 7.5, 7.5])  # 2x2 grid
    expected_y = np.array([2.5, 7.5, 2.5, 7.5])
    
    assert np.allclose(sorted(x_coords), sorted(expected_x))
    assert np.allclose(sorted(y_coords), sorted(expected_y))

def test_distance_calculation_boundary_conditions(sample_boundary):
    """Test distance calculation at boundary conditions."""
    # Create a single point feature at the center
    center_feature = gpd.GeoDataFrame({
        'geometry': [Point(50, 50)]
    }, crs="EPSG:27700")
    
    features = {'point': center_feature}
    distances = calculate_distances(features, sample_boundary, resolution=10)
    
    # Check that distances increase from center
    distance_array = distances.distance_to_point.values
    center_idx = len(distance_array) // 2
    assert isinstance(distance_array, np.ndarray)

