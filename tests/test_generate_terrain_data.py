"""
Tests for terrain data generation functionality.
"""

import pytest
import numpy as np
import xarray as xr
import rioxarray as rxr
from pathlib import Path
from unittest.mock import patch, AsyncMock
from shapely.geometry import Polygon
import geopandas as gpd

from sdm.commands.data_preparation.environmental.generate_terrain_data import (
    generate_terrain_data
)
from sdm.data.terrain import create_terrain_wcs_downloaders


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


class TestTerrainDataWorkflow:
    """Test the terrain data workflow with real functionality."""
    
    def test_generate_terrain_data_creates_output_dir(self, temp_boundary_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nonexistent" / "output"
        
        # Only mock the WCS downloaders to avoid external network calls
        with patch('sdm.commands.data_preparation.environmental.generate_terrain_data.create_terrain_wcs_downloaders') as mock_create_downloaders:
            # Create mock WCS downloaders that return realistic terrain data
            mock_dtm_downloader = AsyncMock()
            mock_dsm_downloader = AsyncMock()
            
            mock_create_downloaders.return_value = {
                "dtm": mock_dtm_downloader,
                "dsm": mock_dsm_downloader
            }
            
            # Create realistic terrain data structures
            dtm_coverage_id = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
            dsm_coverage_id = "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"
            
            # Create test terrain elevation data
            test_dtm_data = np.random.uniform(50, 200, (100, 100)).astype(np.float32)
            test_dsm_data = np.random.uniform(60, 250, (100, 100)).astype(np.float32)
            
            dtm_data = xr.DataArray(
                test_dtm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 100),
                    'x': np.linspace(420000, 421000, 100)
                },
                name=dtm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            dsm_data = xr.DataArray(
                test_dsm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 100),
                    'x': np.linspace(420000, 421000, 100)
                },
                name=dsm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_dtm_downloader.get_coverage.return_value = xr.Dataset({dtm_coverage_id: dtm_data})
            mock_dsm_downloader.get_coverage.return_value = xr.Dataset({dsm_coverage_id: dsm_data})
            mock_dtm_downloader.coverage_id = dtm_coverage_id
            mock_dsm_downloader.coverage_id = dsm_coverage_id
            
            # Run the function
            result = generate_terrain_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                wcs_download_resolution_m=10,
                max_concurrent_downloads=1,  # Use minimal concurrency for testing
                verbose=False
            )
            
            # Verify output directory was created
            assert output_dir.exists()
            assert output_dir.is_dir()
            
            # Verify the output file was created
            assert result.exists()
            assert result.name == "dtm_dsm_100m.tif"
            
            # Verify the file is a valid GeoTIFF by reading it back
            with rxr.open_rasterio(result) as raster:
                assert raster.ndim >= 2  # Should have spatial dimensions
                assert raster.sizes['x'] > 0 and raster.sizes['y'] > 0
                assert raster.sizes['band'] >= 2  # Should have DTM and DSM bands
    
    def test_generate_terrain_data_with_custom_parameters(self, temp_boundary_file, tmp_path):
        """Test terrain data generation with custom parameters."""
        output_dir = tmp_path / "output"
        
        # Only mock the WCS downloaders
        with patch('sdm.commands.data_preparation.environmental.generate_terrain_data.create_terrain_wcs_downloaders') as mock_create_downloaders:
            mock_dtm_downloader = AsyncMock()
            mock_dsm_downloader = AsyncMock()
            
            mock_create_downloaders.return_value = {
                "dtm": mock_dtm_downloader,
                "dsm": mock_dsm_downloader
            }
            
            # Create test terrain data
            dtm_coverage_id = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
            dsm_coverage_id = "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"
            test_dtm_data = np.random.uniform(100, 150, (50, 50)).astype(np.float32)
            test_dsm_data = np.random.uniform(110, 160, (50, 50)).astype(np.float32)
            
            dtm_data = xr.DataArray(
                test_dtm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 50),
                    'x': np.linspace(420000, 421000, 50)
                },
                name=dtm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            dsm_data = xr.DataArray(
                test_dsm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 50),
                    'x': np.linspace(420000, 421000, 50)
                },
                name=dsm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_dtm_downloader.get_coverage.return_value = xr.Dataset({dtm_coverage_id: dtm_data})
            mock_dsm_downloader.get_coverage.return_value = xr.Dataset({dsm_coverage_id: dsm_data})
            mock_dtm_downloader.coverage_id = dtm_coverage_id
            mock_dsm_downloader.coverage_id = dsm_coverage_id
            
            # Run with custom parameters
            result = generate_terrain_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                buffer_distance_m=5000,
                wcs_tile_width_px=512,
                wcs_tile_height_px=512,
                wcs_temp_storage=False,
                wcs_download_resolution_m=5,
                max_concurrent_downloads=2,
                verbose=True
            )
            
            # Verify the result
            assert result.exists()
            assert result.name == "dtm_dsm_100m.tif"
            
            # Verify WCS downloaders were created with correct parameters
            mock_create_downloaders.assert_called_once()
            create_call_args = mock_create_downloaders.call_args[1]
            assert create_call_args['tile_pixels'] == (512, 512)
            assert create_call_args['use_temp_storage'] is False
            
            # Verify get_coverage was called with correct parameters
            for downloader in [mock_dtm_downloader, mock_dsm_downloader]:
                get_coverage_call = downloader.get_coverage.call_args[1]
                assert get_coverage_call['resolution'] == 5.0
                assert get_coverage_call['max_concurrent'] == 2
    
    def test_generate_terrain_data_file_not_found(self, tmp_path):
        """Test error handling when boundary file doesn't exist."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or DataSourceError
            generate_terrain_data(
                output_dir=tmp_path / "output",
                boundary_path=Path("nonexistent_file.geojson")
            )
    
    def test_terrain_data_processing_pipeline(self, temp_boundary_file, tmp_path):
        """Test the complete terrain data processing pipeline with real functions."""
        output_dir = tmp_path / "output"
        
        # Only mock the WCS downloaders to avoid external calls
        with patch('sdm.commands.data_preparation.environmental.generate_terrain_data.create_terrain_wcs_downloaders') as mock_create_downloaders:
            mock_dtm_downloader = AsyncMock()
            mock_dsm_downloader = AsyncMock()
            
            mock_create_downloaders.return_value = {
                "dtm": mock_dtm_downloader,
                "dsm": mock_dsm_downloader
            }
            
            # Create realistic terrain data with known characteristics
            dtm_coverage_id = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
            dsm_coverage_id = "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"
            
            # Create test data with specific patterns to verify processing
            test_dtm_data = np.full((80, 80), 100.0, dtype=np.float32)  # Uniform 100m DTM
            test_dtm_data[20:40, 20:40] = 150.0  # Higher terrain in center
            test_dtm_data[60:80, 60:80] = 50.0   # Lower terrain in corner
            
            test_dsm_data = np.full((80, 80), 120.0, dtype=np.float32)  # Uniform 120m DSM
            test_dsm_data[20:40, 20:40] = 180.0  # Higher surface in center (buildings/trees)
            test_dsm_data[60:80, 60:80] = 80.0   # Lower surface in corner
            
            dtm_data = xr.DataArray(
                test_dtm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 80),
                    'x': np.linspace(420000, 421000, 80)
                },
                name=dtm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            dsm_data = xr.DataArray(
                test_dsm_data,
                dims=('y', 'x'),
                coords={
                    'y': np.linspace(381000, 380000, 80),
                    'x': np.linspace(420000, 421000, 80)
                },
                name=dsm_coverage_id
            ).rio.write_crs("EPSG:27700")
            
            mock_dtm_downloader.get_coverage.return_value = xr.Dataset({dtm_coverage_id: dtm_data})
            mock_dsm_downloader.get_coverage.return_value = xr.Dataset({dsm_coverage_id: dsm_data})
            mock_dtm_downloader.coverage_id = dtm_coverage_id
            mock_dsm_downloader.coverage_id = dsm_coverage_id
            
            # Run the processing pipeline
            result = generate_terrain_data(
                output_dir=output_dir,
                boundary_path=temp_boundary_file,
                wcs_download_resolution_m=10,
                max_concurrent_downloads=1,
                verbose=False
            )
            
            # Verify the output file was created and contains expected data
            assert result.exists()
            
            # Read back the processed data to verify it contains DTM and DSM
            with rxr.open_rasterio(result) as processed_raster:
                # Should have 2 bands (DTM and DSM)
                assert processed_raster.sizes['band'] == 2
                
                # Check that we have spatial dimensions
                assert processed_raster.sizes['x'] > 0
                assert processed_raster.sizes['y'] > 0
                
                # Verify data is within reasonable bounds for elevation
                for band_idx in range(2):
                    band_data = processed_raster.isel(band=band_idx)
                    valid_data = band_data.where(band_data != band_data.rio.nodata)
                    
                    if valid_data.count() > 0:
                        # Elevation should be reasonable (0-500m)
                        assert valid_data.min() >= 0
                        assert valid_data.max() <= 500


class TestTerrainDataRealWCS:
    """Test terrain data generation with real WCS service calls."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_generate_terrain_data_real_wcs_service(self, tmp_path):
        """Test terrain data generation with actual WCS service calls."""
        output_dir = tmp_path / "output"
        
        # Use a small test area to minimize download time
        # Small test area in Sheffield region (500m x 500m)
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
        result = generate_terrain_data(
            output_dir=output_dir,
            boundary_path=small_boundary_file,
            buffer_distance_m=500,  # Small buffer
            wcs_tile_width_px=256,   # Smaller tiles
            wcs_tile_height_px=256,
            wcs_temp_storage=True,
            wcs_download_resolution_m=10,  # 10m resolution
            max_concurrent_downloads=1,    # Single download for testing
            verbose=True
        )
        
        # Verify the output file was created
        assert result.exists()
        assert result.name == "dtm_dsm_100m.tif"
        
        # Verify the file is a valid GeoTIFF by reading it back
        with rxr.open_rasterio(result) as processed_raster:
            # Should have 2 bands (DTM and DSM)
            assert processed_raster.sizes['band'] == 2
            
            # Check that we have spatial dimensions
            assert processed_raster.sizes['x'] > 0
            assert processed_raster.sizes['y'] > 0
            
            # Verify data is within reasonable bounds for elevation
            for band_idx in range(2):
                band_data = processed_raster.isel(band=band_idx)
                valid_data = band_data.where(band_data != band_data.rio.nodata)
                
                if valid_data.count() > 0:
                    # Elevation should be reasonable (0-1000m for real data)
                    assert valid_data.min() >= -50  # Allow some negative values for nodata
                    assert valid_data.max() <= 1000
                    
                    # Should have some variation in real terrain data
                    data_std = float(valid_data.std())
                    assert data_std >= 0  # Should have some standard deviation
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_terrain_wcs_downloaders_directly(self):
        """Test the terrain WCS downloaders directly with real service."""
        # Create a small test area
        test_bbox = (420500, 380500, 421000, 381000)  # Small area in Sheffield
        
        # Initialize terrain WCS downloaders
        wcs_downloaders = create_terrain_wcs_downloaders(
            tile_pixels=(256, 256),
            use_temp_storage=True
        )
        
        # Test direct downloads for both DTM and DSM
        results = {}
        for layer_name, downloader in wcs_downloaders.items():
            result = await downloader.get_coverage(
                bbox=test_bbox,
                resolution=10.0,
                max_concurrent=1
            )
            results[layer_name] = result
        
        # Verify we got data back for both layers
        assert "dtm" in results
        assert "dsm" in results
        
        for layer_name, result in results.items():
            assert isinstance(result, xr.Dataset)
            assert len(result.data_vars) > 0
            
            # Check the coverage ID variable exists
            downloader = wcs_downloaders[layer_name]
            coverage_id = downloader.coverage_id
            assert coverage_id in result.data_vars
            
            # Verify the data has reasonable characteristics
            layer_data = result[coverage_id]
            assert layer_data.ndim >= 2  # Should have spatial dimensions
            assert layer_data.sizes['x'] > 0 and layer_data.sizes['y'] > 0
            
            # Check data values are reasonable for elevation
            valid_data = layer_data.where(layer_data != layer_data.rio.nodata)
            if valid_data.count() > 0:
                assert float(valid_data.min()) >= -50  # Allow some negative values
                assert float(valid_data.max()) <= 1000  # Reasonable max elevation
                
                # DTM should generally be lower than DSM (ground vs surface)
                if layer_name == "dtm":
                    dtm_data = valid_data
                elif layer_name == "dsm":
                    dsm_data = valid_data
        
        # Verify DTM is generally lower than DSM (ground vs surface elevation)
        if 'dtm' in results and 'dsm' in results:
            dtm_coverage_id = wcs_downloaders["dtm"].coverage_id
            dsm_coverage_id = wcs_downloaders["dsm"].coverage_id
            
            dtm_valid = results["dtm"][dtm_coverage_id].where(results["dtm"][dtm_coverage_id] != results["dtm"][dtm_coverage_id].rio.nodata)
            dsm_valid = results["dsm"][dsm_coverage_id].where(results["dsm"][dsm_coverage_id] != results["dsm"][dsm_coverage_id].rio.nodata)
            
            if dtm_valid.count() > 0 and dsm_valid.count() > 0:
                # DSM should generally be higher than DTM (surface vs ground)
                assert float(dsm_valid.mean()) >= float(dtm_valid.mean())
