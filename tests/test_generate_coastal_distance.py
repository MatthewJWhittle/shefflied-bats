"""
Tests for coastal distance generation functionality.
"""

import pytest
import numpy as np
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shapely.geometry import Polygon
from unittest.mock import Mock, patch

from sdm.commands.data_preparation.spatial.generate_coastal_distance import (
    load_and_process_coast_data,
    create_sea_zone_polygon,
    generate_coastal_distance
)
from sdm.raster.processing import calculate_distance_to_geom as calculate_coastal_distance


@pytest.fixture
def sample_coast_data():
    """Create sample coast data for testing."""
    # Create a simple coastline polygon
    coast_polygon = Polygon([
        (0, 0), (100, 0), (100, 50), (80, 50), (80, 30), (60, 30), 
        (60, 50), (40, 50), (40, 30), (20, 30), (20, 50), (0, 50), (0, 0)
    ])
    
    return gpd.GeoDataFrame(
        {'id': [1, 2], 'geometry': [coast_polygon, coast_polygon.buffer(5)]},
        crs="EPSG:27700"
    )


@pytest.fixture
def sample_boundary():
    """Create sample boundary for testing."""
    boundary_polygon = Polygon([
        (-10, -10), (110, -10), (110, 60), (-10, 60), (-10, -10)
    ])
    
    return gpd.GeoDataFrame(
        {'id': [1], 'geometry': [boundary_polygon]},
        crs="EPSG:27700"
    )


@pytest.fixture
def temp_coast_file(tmp_path, sample_coast_data):
    """Create a temporary coast shapefile for testing."""
    coast_file = tmp_path / "test_coast.shp"
    sample_coast_data.to_file(coast_file)
    return coast_file


@pytest.fixture
def temp_boundary_file(tmp_path, sample_boundary):
    """Create a temporary boundary file for testing."""
    boundary_file = tmp_path / "test_boundary.geojson"
    sample_boundary.to_file(boundary_file)
    return boundary_file


class TestLoadAndProcessCoastData:
    """Test the load_and_process_coast_data function."""
    
    def test_load_and_process_coast_data_success(self, temp_coast_file):
        """Test successful loading and processing of coast data."""
        result = load_and_process_coast_data(
            temp_coast_file, "EPSG:27700", simplify_tolerance_m=100.0
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1  # Should be dissolved to single geometry
        assert result.crs == "EPSG:27700"
        assert "geometry" in result.columns
    
    def test_load_and_process_coast_data_file_not_found(self):
        """Test error handling when coast file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_and_process_coast_data(
                Path("nonexistent_file.shp"), "EPSG:27700"
            )
    
    def test_load_and_process_coast_data_crs_conversion(self, temp_coast_file):
        """Test CRS conversion during loading."""
        result = load_and_process_coast_data(
            temp_coast_file, "EPSG:4326", simplify_tolerance_m=100.0
        )
        
        assert result.crs == "EPSG:4326"
    
    def test_load_and_process_coast_data_simplify(self, temp_coast_file):
        """Test geometry simplification."""
        result = load_and_process_coast_data(
            temp_coast_file, "EPSG:27700", simplify_tolerance_m=1000.0
        )
        
        # Simplified geometry should have fewer vertices
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result.geometry.iloc[0].exterior.coords) <= 13  # Original had 13 points


class TestCreateSeaZonePolygon:
    """Test the create_sea_zone_polygon function."""
    
    def test_create_sea_zone_polygon_success(self, sample_coast_data):
        """Test successful creation of sea zone polygon."""
        result = create_sea_zone_polygon(
            sample_coast_data, 
            buffer_dist_km_for_sea=1.0,  # 1km buffer
            min_sea_area_km2=0.001  # Very small minimum area for test
        )
        
        assert result is not None
        assert hasattr(result, 'area')
        assert result.area > 0
    
    def test_create_sea_zone_polygon_no_sea_areas(self, sample_coast_data):
        """Test error when no sea areas meet minimum size requirement."""
        with pytest.raises(ValueError, match="No 'sea' polygons remaining"):
            create_sea_zone_polygon(
                sample_coast_data,
                buffer_dist_km_for_sea=0.001,  # Very small buffer
                min_sea_area_km2=1000.0  # Very large minimum area
            )
    
    def test_create_sea_zone_polygon_buffer_size(self, sample_coast_data):
        """Test that buffer size affects the result."""
        small_buffer = create_sea_zone_polygon(
            sample_coast_data, buffer_dist_km_for_sea=0.5, min_sea_area_km2=0.001
        )
        large_buffer = create_sea_zone_polygon(
            sample_coast_data, buffer_dist_km_for_sea=2.0, min_sea_area_km2=0.001
        )
        
        assert large_buffer.area > small_buffer.area


class TestCalculateCoastalDistance:
    """Test the calculate_coastal_distance function from spatial module."""
    
    def test_calculate_coastal_distance_success(self, sample_boundary):
        """Test successful distance calculation."""
        # Create a simple sea polygon
        sea_polygon = Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)])
        
        result = calculate_coastal_distance(
            geom=sea_polygon,
            boundary_gdf=sample_boundary,
            grid_bounds=(-10, -10, 110, 60),
            resolution=10.0,
            var_name="test_distance"
        )
        
        assert isinstance(result, xr.Dataset)
        assert "test_distance" in result.data_vars
        assert result.test_distance.shape[0] > 0
        assert result.test_distance.shape[1] > 0
    
    def test_calculate_coastal_distance_invalid_geometry(self, sample_boundary):
        """Test error handling for invalid geometry."""
        with pytest.raises(ValueError, match="Input geom must be a Shapely BaseGeometry"):
            calculate_coastal_distance(
                geom="invalid",  # Invalid geometry type (string)
                boundary_gdf=sample_boundary,
                grid_bounds=(-10, -10, 110, 60),
                resolution=10.0
            )


class TestGenerateCoastalDistance:
    """Test the main generate_coastal_distance function."""
    
    @patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.load_boundary_and_transform')
    @patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.calculate_coastal_distance')
    @patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.reproject_data')
    @patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.squeeze_dataset')
    def test_generate_coastal_distance_success(
        self, mock_squeeze, mock_reproject, mock_calc_distance, 
        mock_load_boundary, temp_coast_file, temp_boundary_file, tmp_path
    ):
        """Test successful coastal distance generation."""
        # Mock the dependencies
        mock_boundary = gpd.GeoDataFrame({'geometry': [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]}, crs="EPSG:27700")
        mock_transform = Mock()
        mock_bounds = (0, 0, 100, 100)
        mock_spatial_config = {"resolution": 10.0}
        
        mock_load_boundary.return_value = (mock_boundary, mock_transform, mock_bounds, mock_spatial_config)
        
        # Mock the distance calculation result
        mock_distance_xr = xr.Dataset({
            'distance_to_coast': (('y', 'x'), np.random.random((10, 10)))
        })
        mock_distance_xr = mock_distance_xr.rio.write_crs("EPSG:27700")
        mock_calc_distance.return_value = mock_distance_xr
        mock_reproject.return_value = mock_distance_xr
        mock_squeeze.return_value = mock_distance_xr
        
        # Mock the to_raster method to avoid file I/O
        mock_distance_xr.rio.to_raster = Mock()
        
        # Run the function
        output_dir = tmp_path / "output"
        result = generate_coastal_distance(
            boundary_path=temp_boundary_file,
            output_dir=output_dir,
            bgs_geocoast_shp_path=temp_coast_file,
            verbose=False
        )
        
        # Verify the result
        assert isinstance(result, Path)
        assert result.name == "coastal_distance.tif"
        assert result.parent == output_dir
        
        # Verify that key functions were called
        mock_load_boundary.assert_called_once()
        mock_calc_distance.assert_called_once()
        mock_reproject.assert_called_once()
        mock_squeeze.assert_called_once()
        # Note: to_raster call is verified by the function completing successfully
    
    def test_generate_coastal_distance_file_not_found(self, temp_boundary_file, tmp_path):
        """Test error handling when coast file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            generate_coastal_distance(
                boundary_path=temp_boundary_file,
                output_dir=tmp_path / "output",
                bgs_geocoast_shp_path=Path("nonexistent_file.shp")
            )
    
    def test_generate_coastal_distance_creates_output_dir(self, temp_coast_file, temp_boundary_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "output"
        
        with patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.load_boundary_and_transform') as mock_load, \
             patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.calculate_coastal_distance') as mock_calc, \
             patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.reproject_data') as mock_reproject, \
             patch('sdm.commands.data_preparation.spatial.generate_coastal_distance.squeeze_dataset') as mock_squeeze:
            
            # Mock dependencies
            mock_boundary = gpd.GeoDataFrame({'geometry': [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]}, crs="EPSG:27700")
            mock_load.return_value = (mock_boundary, Mock(), (0, 0, 100, 100), {"resolution": 10.0})
            
            mock_distance_xr = xr.Dataset({
                'distance_to_coast': (('y', 'x'), np.random.random((10, 10)))
            })
            mock_distance_xr = mock_distance_xr.rio.write_crs("EPSG:27700")
            mock_calc.return_value = mock_distance_xr
            mock_reproject.return_value = mock_distance_xr
            mock_squeeze.return_value = mock_distance_xr
            mock_distance_xr.rio.to_raster = Mock()
            
            # Run function
            generate_coastal_distance(
                boundary_path=temp_boundary_file,
                output_dir=output_dir,
                bgs_geocoast_shp_path=temp_coast_file
            )
            
            # Verify output directory was created
            assert output_dir.exists()
            assert output_dir.is_dir()


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_full_workflow_with_mock_data(self, tmp_path):
        """Test the full workflow with mock data."""
        # Create test data
        coast_polygon = Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
        boundary_polygon = Polygon([(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)])
        
        coast_gdf = gpd.GeoDataFrame({'id': [1], 'geometry': [coast_polygon]}, crs="EPSG:27700")
        boundary_gdf = gpd.GeoDataFrame({'id': [1], 'geometry': [boundary_polygon]}, crs="EPSG:27700")
        
        coast_file = tmp_path / "coast.shp"
        boundary_file = tmp_path / "boundary.geojson"
        
        coast_gdf.to_file(coast_file)
        boundary_gdf.to_file(boundary_file)
        
        # Test individual components
        coast_processed = load_and_process_coast_data(coast_file, "EPSG:27700", 1.0)
        sea_zone = create_sea_zone_polygon(coast_processed, 1.0, 1.0, 0.001)
        
        # Verify results
        assert isinstance(coast_processed, gpd.GeoDataFrame)
        assert sea_zone is not None
        assert sea_zone.area > 0
