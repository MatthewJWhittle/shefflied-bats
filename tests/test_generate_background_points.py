"""
Tests for background points generation functionality.
"""

import pytest
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shapely.geometry import Point, Polygon

from sdm.commands.data_preparation.spatial.generate_background_points import (
    generate_background_points_wrapper
)
from sdm.occurrence.sampling import TransformMethod


@pytest.fixture
def sample_boundary():
    """Create sample boundary for testing."""
    # Small test area in Sheffield region
    boundary_polygon = Polygon([
        (420000, 380000), (421000, 380000), (421000, 381000), (420000, 381000), (420000, 380000)
    ])
    
    return gpd.GeoDataFrame(
        {'id': [1], 'geometry': [boundary_polygon]},
        crs="EPSG:27700"
    )


@pytest.fixture
def temp_boundary_file(tmp_path, sample_boundary):
    """Create a temporary boundary file for testing."""
    boundary_file = tmp_path / "test_boundary.geojson"
    sample_boundary.to_file(boundary_file)
    return boundary_file


@pytest.fixture
def sample_occurrence_data():
    """Create sample occurrence data for testing."""
    # Create some test occurrence points
    occurrence_points = [
        Point(420200, 380200),
        Point(420300, 380300),
        Point(420400, 380400),
        Point(420500, 380500),
        Point(420600, 380600),
        Point(420700, 380700),
        Point(420800, 380800),
        Point(420900, 380900),
    ]
    
    return gpd.GeoDataFrame(
        {
            'species': ['Myotis daubentonii'] * len(occurrence_points),
            'geometry': occurrence_points
        },
        crs="EPSG:27700"
    )


@pytest.fixture
def temp_occurrence_file(tmp_path, sample_occurrence_data):
    """Create a temporary occurrence data file for testing."""
    occurrence_file = tmp_path / "test_occurrences.geojson"
    sample_occurrence_data.to_file(occurrence_file)
    return occurrence_file


class TestBackgroundPointsWorkflow:
    """Test the background points workflow with real functionality."""
    
    def test_generate_background_points_creates_output_dir(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "output"
        
        # Run the function
        bg_points_path, density_raster_path = generate_background_points_wrapper(
            occurrence_data_path=temp_occurrence_file,
            boundary_path=temp_boundary_file,
            output_dir=output_dir,
            n_background_points=100,  # Small number for testing
            grid_resolution=100,  # Provide grid resolution
            verbose=False
        )
        
        # Verify output directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Verify the output files were created
        assert bg_points_path.exists()
        assert density_raster_path.exists()
        
        # Verify the files have expected extensions
        assert bg_points_path.suffix in ['.geojson', '.parquet', '.gpkg']
        assert density_raster_path.suffix in ['.tif', '.tiff']
    
    def test_generate_background_points_with_custom_parameters(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test background points generation with custom parameters."""
        output_dir = tmp_path / "output"
        
        # Run with custom parameters
        bg_points_path, density_raster_path = generate_background_points_wrapper(
            occurrence_data_path=temp_occurrence_file,
            boundary_path=temp_boundary_file,
            output_dir=output_dir,
            n_background_points=50,
            background_method="percentile",
            background_value=0.5,
            grid_resolution=50,
            transform_method=TransformMethod.SQRT,
            cap_percentile=85.0,
            sigma=2.0,
            verbose=True
        )
        
        # Verify the result files exist
        assert bg_points_path.exists()
        assert density_raster_path.exists()
        
        # Verify the background points file contains data
        bg_points = gpd.read_file(bg_points_path)
        assert len(bg_points) > 0
        assert 'geometry' in bg_points.columns
        
        # Verify the density raster file is valid
        import rioxarray as rxr
        with rxr.open_rasterio(density_raster_path) as density_raster:
            assert density_raster.ndim >= 2
            assert density_raster.sizes['x'] > 0 and density_raster.sizes['y'] > 0
    
    def test_generate_background_points_file_not_found(self, temp_boundary_file, tmp_path):
        """Test error handling when occurrence file doesn't exist."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or DataSourceError
            generate_background_points_wrapper(
                occurrence_data_path=Path("nonexistent_file.geojson"),
                boundary_path=temp_boundary_file,
                output_dir=tmp_path / "output",
                grid_resolution=100
            )
    
    def test_background_points_processing_pipeline(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test the complete background points processing pipeline."""
        output_dir = tmp_path / "output"
        
        # Run the processing pipeline with different methods
        for background_method in ["contrast", "percentile", "scale", "fixed", "binary"]:
            bg_points_path, density_raster_path = generate_background_points_wrapper(
                occurrence_data_path=temp_occurrence_file,
                boundary_path=temp_boundary_file,
                output_dir=output_dir / background_method,
                n_background_points=25,  # Small number for testing
                background_method=background_method,
                background_value=0.3 if background_method in ["contrast", "scale"] else 0.5,
                grid_resolution=100,
                verbose=False
            )
            
            # Verify the output files were created
            assert bg_points_path.exists()
            assert density_raster_path.exists()
            
            # Verify background points contain valid data
            bg_points = gpd.read_file(bg_points_path)
            assert len(bg_points) > 0
            assert bg_points.crs == "EPSG:27700"
            
            # Verify points are within the boundary (approximately)
            boundary = gpd.read_file(temp_boundary_file)
            boundary_bounds = boundary.total_bounds
            for _, point in bg_points.iterrows():
                geom = point.geometry
                assert boundary_bounds[0] <= geom.x <= boundary_bounds[2]
                assert boundary_bounds[1] <= geom.y <= boundary_bounds[3]
    
    def test_different_transform_methods(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test different transform methods for occurrence density."""
        output_dir = tmp_path / "output"
        
        transform_methods = [TransformMethod.LOG, TransformMethod.SQRT, TransformMethod.PRESENCE, TransformMethod.CAP]
        
        for transform_method in transform_methods:
            bg_points_path, density_raster_path = generate_background_points_wrapper(
                occurrence_data_path=temp_occurrence_file,
                boundary_path=temp_boundary_file,
                output_dir=output_dir / transform_method.value,
                n_background_points=30,
                transform_method=transform_method,
                cap_percentile=90.0 if transform_method == TransformMethod.CAP else 90.0,
                grid_resolution=100,
                verbose=False
            )
            
            # Verify the output files were created
            assert bg_points_path.exists()
            assert density_raster_path.exists()
            
            # Verify background points contain valid data
            bg_points = gpd.read_file(bg_points_path)
            assert len(bg_points) > 0
    
    def test_background_points_number_control(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test that the number of background points is controlled correctly."""
        output_dir = tmp_path / "output"
        
        test_numbers = [10, 25, 50, 100]
        
        for n_points in test_numbers:
            bg_points_path, density_raster_path = generate_background_points_wrapper(
                occurrence_data_path=temp_occurrence_file,
                boundary_path=temp_boundary_file,
                output_dir=output_dir / f"n_{n_points}",
                n_background_points=n_points,
                grid_resolution=100,
                verbose=False
            )
            
            # Verify the output files were created
            assert bg_points_path.exists()
            assert density_raster_path.exists()
            
            # Verify background points count (should be close to requested number)
            bg_points = gpd.read_file(bg_points_path)
            assert len(bg_points) <= n_points  # Should not exceed requested number
            assert len(bg_points) > 0  # Should have some points


class TestBackgroundPointsValidation:
    """Test validation and edge cases for background points generation."""
    
    def test_empty_occurrence_data(self, temp_boundary_file, tmp_path):
        """Test handling of empty occurrence data."""
        # Create empty occurrence data
        empty_occurrence = gpd.GeoDataFrame(
            {'species': [], 'geometry': []},
            crs="EPSG:27700"
        )
        empty_occurrence_file = tmp_path / "empty_occurrences.geojson"
        empty_occurrence.to_file(empty_occurrence_file)
        
        # This should either raise an error or handle gracefully
        with pytest.raises(Exception):
            generate_background_points_wrapper(
                occurrence_data_path=empty_occurrence_file,
                boundary_path=temp_boundary_file,
                output_dir=tmp_path / "output",
                n_background_points=10,
                grid_resolution=100,
                verbose=False
            )
    
    def test_single_occurrence_point(self, temp_boundary_file, tmp_path):
        """Test handling of single occurrence point."""
        # Create occurrence data with single point
        single_point = gpd.GeoDataFrame(
            {
                'species': ['Myotis daubentonii'],
                'geometry': [Point(420500, 380500)]
            },
            crs="EPSG:27700"
        )
        single_point_file = tmp_path / "single_occurrence.geojson"
        single_point.to_file(single_point_file)
        
        bg_points_path, density_raster_path = generate_background_points_wrapper(
            occurrence_data_path=single_point_file,
            boundary_path=temp_boundary_file,
            output_dir=tmp_path / "output",
            n_background_points=20,
            grid_resolution=100,
            verbose=False
        )
        
        # Should still work with single point
        assert bg_points_path.exists()
        assert density_raster_path.exists()
        
        bg_points = gpd.read_file(bg_points_path)
        assert len(bg_points) > 0
    
    def test_different_sigma_values(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test different sigma values for Gaussian smoothing."""
        output_dir = tmp_path / "output"
        
        sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for sigma in sigma_values:
            bg_points_path, density_raster_path = generate_background_points_wrapper(
                occurrence_data_path=temp_occurrence_file,
                boundary_path=temp_boundary_file,
                output_dir=output_dir / f"sigma_{sigma}",
                n_background_points=30,
                sigma=sigma,
                grid_resolution=100,
                verbose=False
            )
            
            # Verify the output files were created
            assert bg_points_path.exists()
            assert density_raster_path.exists()
            
            # Verify background points contain valid data
            bg_points = gpd.read_file(bg_points_path)
            assert len(bg_points) > 0
    
    def test_output_file_formats(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test that output files are in expected formats."""
        output_dir = tmp_path / "output"
        
        bg_points_path, density_raster_path = generate_background_points_wrapper(
            occurrence_data_path=temp_occurrence_file,
            boundary_path=temp_boundary_file,
            output_dir=output_dir,
            n_background_points=25,
            grid_resolution=100,
            verbose=False
        )
        
        # Verify file formats
        assert bg_points_path.suffix in ['.geojson', '.parquet', '.gpkg']
        assert density_raster_path.suffix in ['.tif', '.tiff']
        
        # Verify files can be read
        bg_points = gpd.read_file(bg_points_path)
        assert isinstance(bg_points, gpd.GeoDataFrame)
        
        import rioxarray as rxr
        with rxr.open_rasterio(density_raster_path) as density_raster:
            assert isinstance(density_raster, xr.DataArray)
    
    def test_crs_consistency(self, temp_boundary_file, temp_occurrence_file, tmp_path):
        """Test that CRS is consistent throughout the process."""
        output_dir = tmp_path / "output"
        
        bg_points_path, density_raster_path = generate_background_points_wrapper(
            occurrence_data_path=temp_occurrence_file,
            boundary_path=temp_boundary_file,
            output_dir=output_dir,
            n_background_points=30,
            grid_resolution=100,
            verbose=False
        )
        
        # Verify CRS consistency
        bg_points = gpd.read_file(bg_points_path)
        boundary = gpd.read_file(temp_boundary_file)
        occurrence_data = gpd.read_file(temp_occurrence_file)
        
        # All should have the same CRS
        assert bg_points.crs == boundary.crs
        assert bg_points.crs == occurrence_data.crs
        assert bg_points.crs == "EPSG:27700"
        
        # Verify density raster has spatial reference
        import rioxarray as rxr
        with rxr.open_rasterio(density_raster_path) as density_raster:
            assert density_raster.rio.crs == "EPSG:27700"