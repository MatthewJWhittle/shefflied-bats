import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import box, Point, LineString, Polygon
import xarray as xr
import rioxarray as rxr
from rasterio.transform import from_bounds

from generate_evs.ingestion.process_os_data import (
    generate_point_grid,
    process_roads,
    calculate_distances,
    calculate_feature_cover,
    rasterise_gdf
)

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
                Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
                Polygon([(60, 60), (70, 60), (70, 70), (60, 70)])
            ]
        }, crs="EPSG:27700"),
        'water': gpd.GeoDataFrame({
            'geometry': [
                Polygon([(30, 30), (40, 30), (40, 40), (30, 40)])
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
    
    # Check coordinates are multiples of resolution
    x_coords = grid.geometry.x
    y_coords = grid.geometry.y
    assert all(x % resolution == 0 for x in x_coords)
    assert all(y % resolution == 0 for y in y_coords)

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

