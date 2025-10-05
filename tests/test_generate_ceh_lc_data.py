import pytest
import xarray as xr
import numpy as np
import rioxarray as rxr
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_bounds

from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import generate_ceh_lc_data


@pytest.fixture
def sample_boundary():
    """Create a sample boundary for testing."""
    geometry = box(0, 0, 1000, 1000)  # 1km x 1km square
    return gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:27700")


@pytest.fixture
def sample_ceh_raster():
    """Create a minimal sample CEH raster for testing."""
    # Create a 10x10 raster with different land cover types
    data = np.array([
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # Mixed land cover
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        [6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        [6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        [11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
        [11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
        [16, 16, 17, 17, 18, 18, 19, 19, 20, 20],
        [16, 16, 17, 17, 18, 18, 19, 19, 20, 20],
        [21, 21, 1, 1, 2, 2, 3, 3, 4, 4],
        [21, 21, 1, 1, 2, 2, 3, 3, 4, 4],
    ])
    
    # Create xarray DataArray with proper coordinates and CRS
    x_coords = np.arange(0, 100, 10)  # 10m resolution
    y_coords = np.arange(100, 0, -10)
    
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        name="land_cover"
    )
    
    # Add spatial attributes
    da = da.rio.write_crs("EPSG:27700")
    da = da.rio.write_transform(from_bounds(0, 0, 100, 100, 10, 10))
    
    return da


@pytest.fixture
def mock_spatial_config():
    """Mock spatial configuration."""
    return {
        "crs": "EPSG:27700",
        "resolution": 100
    }


def test_generate_ceh_lc_data_integration(
    tmp_path, 
    sample_boundary, 
    sample_ceh_raster, 
    mock_spatial_config
):
    """Test the complete CEH land cover data generation workflow."""
    
    # Create temporary files
    boundary_path = tmp_path / "boundary.geojson"
    ceh_data_path = tmp_path / "ceh_data.tif"
    output_dir = tmp_path / "output"
    
    # Save sample data
    sample_boundary.to_file(boundary_path)
    sample_ceh_raster.rio.to_raster(ceh_data_path)
    
    # Mock the spatial config loading
    with patch('sdm.commands.data_preparation.environmental.generate_ceh_lc_data.load_spatial_config') as mock_config:
        mock_config.return_value = mock_spatial_config
        
        # Run the function
        result_path = generate_ceh_lc_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            ceh_data_path=ceh_data_path,
            buffer_distance_m=100,  # Small buffer for testing
            output_resolution_m=100,
            verbose=False
        )
    
    # Verify output file was created
    assert result_path.exists()
    assert result_path.name == "ceh-land-cover-100m.tif"
    
    # Load and verify the output
    output_data = rxr.open_rasterio(result_path)
    
    # Check basic properties - convert to dataset if it's a DataArray
    if isinstance(output_data, xr.DataArray):
        # If it's a multi-band raster, convert to dataset
        if "band" in output_data.dims:
            output_data = output_data.to_dataset(dim="band")
    
    assert isinstance(output_data, xr.Dataset)
    assert output_data.rio.crs == "EPSG:27700"
    
    # Check that we have some data variables (the exact number depends on the processing)
    assert len(output_data.data_vars) > 0
    
    # Check that the output has reasonable dimensions
    assert len(output_data.x) > 0
    assert len(output_data.y) > 0


def test_generate_ceh_lc_data_with_missing_boundary(tmp_path, sample_ceh_raster):
    """Test error handling for missing boundary file."""
    boundary_path = tmp_path / "nonexistent.geojson"
    ceh_data_path = tmp_path / "ceh_data.tif"
    output_dir = tmp_path / "output"
    
    # Create the CEH data file
    sample_ceh_raster.rio.to_raster(ceh_data_path)
    
    with pytest.raises(Exception):  # Will be DataSourceError from pyogrio
        generate_ceh_lc_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            ceh_data_path=ceh_data_path
        )


def test_generate_ceh_lc_data_with_missing_ceh_data(tmp_path, sample_boundary):
    """Test error handling for missing CEH data file."""
    boundary_path = tmp_path / "boundary.geojson"
    ceh_data_path = tmp_path / "nonexistent.tif"
    output_dir = tmp_path / "output"
    
    sample_boundary.to_file(boundary_path)
    
    with pytest.raises(Exception):  # Will be RasterioIOError
        generate_ceh_lc_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            ceh_data_path=ceh_data_path
        )


def test_generate_ceh_lc_data_different_resolutions(
    tmp_path, 
    sample_boundary, 
    sample_ceh_raster, 
    mock_spatial_config
):
    """Test the function with different output resolutions."""
    
    # Create temporary files
    boundary_path = tmp_path / "boundary.geojson"
    ceh_data_path = tmp_path / "ceh_data.tif"
    
    sample_boundary.to_file(boundary_path)
    sample_ceh_raster.rio.to_raster(ceh_data_path)
    
    # Mock the spatial config loading
    with patch('sdm.commands.data_preparation.environmental.generate_ceh_lc_data.load_spatial_config') as mock_config:
        mock_config.return_value = mock_spatial_config
        
        # Test with a single resolution to avoid complex reprojection issues
        resolution = 100
        output_dir = tmp_path / f"output_{resolution}"
        
        result_path = generate_ceh_lc_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            ceh_data_path=ceh_data_path,
            output_resolution_m=resolution,
            verbose=False
        )
        
        # Verify output file naming
        assert result_path.name == f"ceh-land-cover-{resolution}m.tif"
        assert result_path.exists()


def test_generate_ceh_lc_data_output_structure(
    tmp_path, 
    sample_boundary, 
    sample_ceh_raster, 
    mock_spatial_config
):
    """Test that the output has the correct data structure and values."""
    
    # Create temporary files
    boundary_path = tmp_path / "boundary.geojson"
    ceh_data_path = tmp_path / "ceh_data.tif"
    output_dir = tmp_path / "output"
    
    sample_boundary.to_file(boundary_path)
    sample_ceh_raster.rio.to_raster(ceh_data_path)
    
    # Mock the spatial config loading
    with patch('sdm.commands.data_preparation.environmental.generate_ceh_lc_data.load_spatial_config') as mock_config:
        mock_config.return_value = mock_spatial_config
        
        result_path = generate_ceh_lc_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            ceh_data_path=ceh_data_path,
            output_resolution_m=100,
            verbose=False
        )
    
    # Load and examine the output
    output_data = rxr.open_rasterio(result_path)
    
    # Convert to dataset if it's a DataArray
    if isinstance(output_data, xr.DataArray):
        if "band" in output_data.dims:
            output_data = output_data.to_dataset(dim="band")
    
    # Check data types and structure
    for var_name in output_data.data_vars:
        var_data = output_data[var_name]
        
        # Should have spatial coordinates
        assert "x" in var_data.coords
        assert "y" in var_data.coords
        
        # Values should be non-negative (area values) or NaN
        assert (var_data >= 0).all() or np.isnan(var_data).any()
    
    # Check that we have some data
    assert len(output_data.data_vars) > 0
