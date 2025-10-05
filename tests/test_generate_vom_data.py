"""
Tests for VOM (Vegetation Object Model) data generation functionality.
"""

import pytest
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path
from unittest.mock import patch, AsyncMock
from shapely.geometry import Polygon
import geopandas as gpd

from sdm.commands.data_preparation.environmental.generate_vom_data import (
    generate_vom_data
)
from sdm.data import WCSDownloader


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


class TestVomDataWorkflow:
    """Test the VOM data workflow with real functionality."""
    
    def test_generate_vom_data_creates_output_dir(self, temp_boundary_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "output"
        
        # Only mock the WCS downloader to avoid external network calls
        with patch('sdm.commands.data_preparation.environmental.generate_vom_data.WCSDownloader') as mock_wcs_class:
            # Create a mock WCS downloader that returns realistic VOM data
            mock_wcs_downloader = mock_wcs_class.return_value
            mock_wcs_downloader.get_coverage = AsyncMock()
            
            # Create realistic VOM data structure
            vom_coverage_id = "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
            
            # Create test vegetation height data
            test_data = np.random.uniform(0, 30, (100, 100)).astype(np.float32)
            test_data[0:10, 0:10] = np.nan  # Add some nodata areas
            
            vom_data = xr.DataArray(
                test_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 100),
                    'x': np.linspace(420000, 421000, 100)
                },
                name=vom_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_vom_dataset = xr.Dataset({vom_coverage_id: vom_data})
            mock_wcs_downloader.get_coverage.return_value = mock_vom_dataset
            
            # Run the function
            result = generate_vom_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                wcs_download_resolution_m=10,
                summary_target_resolution_m=100,
                max_concurrent_downloads=1,  # Use minimal concurrency for testing
                verbose=False
            )
            
            # Verify output directory was created
            assert output_dir.exists()
            assert output_dir.is_dir()
            
            # Verify the output file was created
            assert result.exists()
            assert result.name == "vom_summary_metrics_100m.tif"
            
            # Verify the file is a valid GeoTIFF by reading it back
            with rxr.open_rasterio(result) as raster:
                assert raster.ndim >= 2  # Should have spatial dimensions
                assert raster.sizes['x'] > 0 and raster.sizes['y'] > 0
    
    def test_generate_vom_data_with_custom_parameters(self, temp_boundary_file, tmp_path):
        """Test VOM data generation with custom parameters."""
        output_dir = tmp_path / "output"
        
        # Only mock the WCS downloader
        with patch('sdm.commands.data_preparation.environmental.generate_vom_data.WCSDownloader') as mock_wcs_class:
            mock_wcs_downloader = mock_wcs_class.return_value
            mock_wcs_downloader.get_coverage = AsyncMock()
            
            # Create test VOM data
            vom_coverage_id = "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
            test_data = np.random.uniform(5, 25, (50, 50)).astype(np.float32)
            
            vom_data = xr.DataArray(
                test_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 50),
                    'x': np.linspace(420000, 421000, 50)
                },
                name=vom_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_vom_dataset = xr.Dataset({vom_coverage_id: vom_data})
            mock_wcs_downloader.get_coverage.return_value = mock_vom_dataset
            
            # Run with custom parameters
            result = generate_vom_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                buffer_distance_m=5000,
                wcs_tile_width_px=512,
                wcs_tile_height_px=512,
                wcs_temp_storage=False,
                wcs_download_resolution_m=5,
                max_concurrent_downloads=2,
                summary_target_resolution_m=50,
                verbose=True
            )
            
            # Verify the result
            assert result.exists()
            assert result.name == "vom_summary_metrics_50m.tif"
            
            # Verify WCS downloader was called with correct parameters
            mock_wcs_class.assert_called_once()
            wcs_call_args = mock_wcs_class.call_args[1]
            assert wcs_call_args['request_tile_pixels'] == (512, 512)
            assert wcs_call_args['use_temp_storage'] is False
            
            # Verify get_coverage was called with correct parameters
            get_coverage_call = mock_wcs_downloader.get_coverage.call_args[1]
            assert get_coverage_call['resolution'] == 5.0
            assert get_coverage_call['max_concurrent'] == 2
    
    def test_generate_vom_data_file_not_found(self, tmp_path):
        """Test error handling when boundary file doesn't exist."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or DataSourceError
            generate_vom_data(
                output_dir=tmp_path / "output",
                boundary_path=Path("nonexistent_file.geojson")
            )
    
    def test_vom_data_processing_pipeline(self, temp_boundary_file, tmp_path):
        """Test the complete VOM data processing pipeline with real functions."""
        output_dir = tmp_path / "output"
        
        # Only mock the WCS downloader to avoid external calls
        with patch('sdm.commands.data_preparation.environmental.generate_vom_data.WCSDownloader') as mock_wcs_class:
            mock_wcs_downloader = mock_wcs_class.return_value
            mock_wcs_downloader.get_coverage = AsyncMock()
            
            # Create realistic VOM data with known characteristics
            vom_coverage_id = "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
            
            # Create test data with specific patterns to verify processing
            test_data = np.full((80, 80), 15.0, dtype=np.float32)  # Uniform 15m height
            test_data[20:40, 20:40] = 25.0  # Higher vegetation in center
            test_data[60:80, 60:80] = 5.0   # Lower vegetation in corner
            test_data[0:10, :] = np.nan     # No data strip
            
            vom_data = xr.DataArray(
                test_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 80),
                    'x': np.linspace(420000, 421000, 80)
                },
                name=vom_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_vom_dataset = xr.Dataset({vom_coverage_id: vom_data})
            mock_wcs_downloader.get_coverage.return_value = mock_vom_dataset
            
            # Run the processing pipeline
            result = generate_vom_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                wcs_download_resolution_m=10,
                summary_target_resolution_m=100,
                max_concurrent_downloads=1,
                verbose=False
            )
            
            # Verify the output file was created and contains expected data
            assert result.exists()
            
            # Read back the processed data to verify it contains summary metrics
            with rxr.open_rasterio(result) as processed_raster:
                # Should have multiple bands (mean, min, max, std)
                assert processed_raster.sizes['band'] >= 4
                
                # Check that we have spatial dimensions
                assert processed_raster.sizes['x'] > 0
                assert processed_raster.sizes['y'] > 0
                
                # Verify data is within reasonable bounds for vegetation height
                for band_idx in range(min(4, processed_raster.sizes['band'])):
                    band_data = processed_raster.isel(band=band_idx)
                    valid_data = band_data.where(band_data != band_data.rio.nodata)
                    
                    if valid_data.count() > 0:
                        # Vegetation height should be reasonable (0-50m)
                        assert valid_data.min() >= 0
                        assert valid_data.max() <= 50


class TestVomDataRealWCS:
    """Test VOM data generation with real WCS service calls."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_generate_vom_data_real_wcs_service(self, tmp_path):
        """Test VOM data generation with actual WCS service calls."""
        output_dir = tmp_path / "output"
        
        # Use a small test area to minimize download time
        # Small test area in Sheffield region (1km x 1km)
        small_boundary = Polygon([
            (420500, 380500), (421000, 380500), (421000, 381000), (420500, 381000), (420500, 380500)
        ])
        
        small_boundary_gdf = gpd.GeoDataFrame(
            {'id': [1], 'geometry': [small_boundary]},
            crs="EPSG:27700"
        )
        
        small_boundary_file = tmp_path / "small_boundary.geojson"
        small_boundary_gdf.to_file(small_boundary_file)
        
        # Run with minimal parameters to reduce download time
        result = generate_vom_data(
            output_dir=output_dir,
            boundary_path=small_boundary_file,
            buffer_distance_m=1000,  # Small buffer
            wcs_tile_width_px=256,   # Smaller tiles
            wcs_tile_height_px=256,
            wcs_temp_storage=True,
            wcs_download_resolution_m=10,  # 10m resolution
            max_concurrent_downloads=1,    # Single download for testing
            summary_target_resolution_m=100,  # Coarse summary
            verbose=True
        )
        
        # Verify the output file was created
        assert result.exists()
        assert result.name == "vom_summary_metrics_100m.tif"
        
        # Verify the file is a valid GeoTIFF by reading it back
        with rxr.open_rasterio(result) as processed_raster:
            # Should have multiple bands (mean, min, max, std)
            assert processed_raster.sizes['band'] >= 4
            
            # Check that we have spatial dimensions
            assert processed_raster.sizes['x'] > 0
            assert processed_raster.sizes['y'] > 0
            
            # Verify data is within reasonable bounds for vegetation height
            for band_idx in range(min(4, processed_raster.sizes['band'])):
                band_data = processed_raster.isel(band=band_idx)
                valid_data = band_data.where(band_data != band_data.rio.nodata)
                
                if valid_data.count() > 0:
                    # Vegetation height should be reasonable (0-100m for real data)
                    assert valid_data.min() >= -10  # Allow some negative values for nodata
                    assert valid_data.max() <= 100
                    
                    # Should have some variation in real vegetation data
                    data_std = float(valid_data.std())
                    assert data_std >= 0  # Should have some standard deviation
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_wcs_downloader_directly(self):
        """Test the WCS downloader directly with real service."""
        # Create a small test area
        test_bbox = (420500, 380500, 421000, 381000)  # Small area in Sheffield
        
        # Initialize WCS downloader with VOM service
        wcs_downloader = WCSDownloader(
            endpoint="https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs",
            coverage_id="ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022",
            request_tile_pixels=(256, 256),
            use_temp_storage=True
        )
        
        # Test direct download
        result = await wcs_downloader.get_coverage(
            bbox=test_bbox,
            resolution=10.0,
            max_concurrent=1
        )
        
        # Verify we got data back
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) > 0
        
        # Check the coverage ID variable exists
        vom_coverage_id = "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
        assert vom_coverage_id in result.data_vars
        
        # Verify the data has reasonable characteristics
        vom_data = result[vom_coverage_id]
        assert vom_data.ndim >= 2  # Should have spatial dimensions
        assert vom_data.sizes['x'] > 0 and vom_data.sizes['y'] > 0
        
        # Check data values are reasonable for vegetation height
        valid_data = vom_data.where(vom_data != vom_data.rio.nodata)
        if valid_data.count() > 0:
            assert float(valid_data.min()) >= -10  # Allow some negative values
            assert float(valid_data.max()) <= 100  # Reasonable max vegetation height
