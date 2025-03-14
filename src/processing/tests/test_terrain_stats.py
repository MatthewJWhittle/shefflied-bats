import os
import pytest
import numpy as np
import xarray as xr
import richdem as rd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.processing.terrain_stats import (
    calculate_slope_aspect,
    calculate_aspect_components,
    calculate_twi,
    calculate_curvature,
    calculate_roughness,
    calculate_tpi,
    calculate_weighted_aspect,
    process_terrain_stats,
    save_terrain_stats,
    main
)


@pytest.fixture
def sample_dem_array():
    """Create a sample DEM array for testing."""
    # Create a simple 5x5 DEM with a hill in the middle
    dem = np.array([
        [10, 10, 10, 10, 10],
        [10, 12, 14, 12, 10],
        [10, 14, 16, 14, 10],
        [10, 12, 14, 12, 10],
        [10, 10, 10, 10, 10]
    ], dtype=np.float32)
    return dem


@pytest.fixture
def sample_dem_dataset(sample_dem_array):
    """Create a sample DEM xarray dataset for testing."""
    dem_da = xr.DataArray(
        sample_dem_array,
        dims=["y", "x"],
        coords={"y": range(5), "x": range(5)},
        attrs={"_FillValue": -9999}
    )
    return dem_da.to_dataset(name="dem")


@pytest.fixture
def sample_dem_rd(sample_dem_array):
    """Create a sample RichDEM array for testing."""
    dem_rd = rd.rdarray(sample_dem_array, no_data=-9999)
    return dem_rd


def test_calculate_slope_aspect(sample_dem_rd):
    """Test slope and aspect calculation."""
    slope, aspect = calculate_slope_aspect(sample_dem_rd)
    
    # Check that slope and aspect are rdarray objects
    assert isinstance(slope, rd.rdarray)
    assert isinstance(aspect, rd.rdarray)
    
    # Check dimensions
    assert slope.shape == sample_dem_rd.shape
    assert aspect.shape == sample_dem_rd.shape
    
    slope = np.array(slope, copy=False)
    aspect = np.array(aspect, copy=False)

    # Check that slopes are higher in the middle (steeper)
    assert np.mean(slope[1:4, 1:4]) > np.mean(slope[0, :])


def test_calculate_aspect_components(sample_dem_rd):
    """Test aspect components calculation."""
    _, aspect = calculate_slope_aspect(sample_dem_rd)
    eastness, northness = calculate_aspect_components(aspect)
    
    # Check types and shapes
    assert isinstance(eastness, np.ndarray)
    assert isinstance(northness, np.ndarray)
    assert eastness.shape == aspect.shape
    assert northness.shape == aspect.shape
    
    # Check that values are within expected range [-1, 1]
    assert np.all(eastness >= -1) and np.all(eastness <= 1)
    assert np.all(northness >= -1) and np.all(northness <= 1)


def test_calculate_twi(sample_dem_rd):
    """Test TWI calculation."""
    slope, _ = calculate_slope_aspect(sample_dem_rd)
    slope_array = np.array(slope, copy=False)
    twi = calculate_twi(sample_dem_rd, slope_array)
    
    # Check type and shape
    assert isinstance(twi, np.ndarray)
    assert twi.shape == sample_dem_rd.shape


def test_calculate_curvature(sample_dem_rd):
    """Test curvature calculation."""
    curvature = calculate_curvature(sample_dem_rd)
    
    # Check type and shape
    assert isinstance(curvature, rd.rdarray)
    assert curvature.shape == sample_dem_rd.shape


def test_calculate_roughness(sample_dem_dataset):
    """Test roughness calculation."""
    # Create a slope array similar to what would be calculated
    sample_dem_dataset["slope"] = (("y", "x"), np.random.random((5, 5)))
    
    roughness = calculate_roughness(sample_dem_dataset.slope)
    
    # Check type and attributes
    assert isinstance(roughness, xr.DataArray)
    assert roughness.dims == sample_dem_dataset.slope.dims
    
    # Center cells should have values, edges should be NaN due to rolling window
    assert ~np.isnan(roughness.values[2, 2])  # Center should have value
    assert np.isnan(roughness.values[0, 0])   # Corners should be NaN


def test_calculate_tpi(sample_dem_dataset):
    """Test TPI calculation."""
    tpi = calculate_tpi(sample_dem_dataset.dem)
    
    # Check type and attributes
    assert isinstance(tpi, xr.DataArray)
    assert tpi.dims == sample_dem_dataset.dem.dims
    
    # The highest elevation point should have positive TPI
    max_index = np.unravel_index(np.argmax(sample_dem_dataset.dem.values), sample_dem_dataset.dem.shape)
    assert tpi.values[max_index] > 0


def test_calculate_weighted_aspect(sample_dem_dataset):
    """Test weighted aspect calculation."""
    # Add sample slope and aspect components to dataset
    sample_dem_dataset["slope"] = (("y", "x"), np.random.random((5, 5)))
    sample_dem_dataset["aspect_eastness"] = (("y", "x"), np.random.uniform(-1, 1, (5, 5)))
    sample_dem_dataset["aspect_northness"] = (("y", "x"), np.random.uniform(-1, 1, (5, 5)))
    
    weighted_eastness, weighted_northness = calculate_weighted_aspect(
        sample_dem_dataset.slope, 
        sample_dem_dataset.aspect_eastness,
        sample_dem_dataset.aspect_northness
    )
    
    # Check type and attributes
    assert isinstance(weighted_eastness, xr.DataArray)
    assert isinstance(weighted_northness, xr.DataArray)
    assert weighted_eastness.dims == sample_dem_dataset.slope.dims
    assert weighted_northness.dims == sample_dem_dataset.slope.dims


@patch("richdem.LoadGDAL")
@patch("rioxarray.open_rasterio")
def test_process_terrain_stats(mock_open_rasterio, mock_load_gdal, sample_dem_rd, sample_dem_dataset):
    """Test terrain stats processing."""
    # Mock the file loading functions
    mock_load_gdal.return_value = sample_dem_rd
    mock_open_rasterio.return_value = sample_dem_dataset.dem
    
    # Process terrain stats
    terrain = process_terrain_stats("dummy_path.tif", visualize=False)
    
    # Check that the expected variables are in the result
    expected_vars = [
        "dem", "slope", "aspect_eastness", "aspect_northness", 
        "twi", "curvature", "roughness", "tpi",
        "aspect_eastness_slope", "aspect_northness_slope"
    ]
    
    for var in expected_vars:
        assert var in terrain.data_vars


@patch("xarray.DataArray.rio")
def test_save_terrain_stats(mock_rio, sample_dem_dataset, tmp_path):
    """Test saving terrain stats."""
    # Set up mock
    mock_rio.to_raster = MagicMock()

    # add another variable to the dataset
    sample_dem_dataset["slope"] = (("y", "x"), np.random.random((5, 5)))
    
    # Execute function
    output_path = tmp_path / "terrain-stats.tif"
    result_path = save_terrain_stats(sample_dem_dataset, output_path)
    
    # Check result
    assert result_path == output_path
    mock_rio.to_raster.assert_called_once()


@patch("src.processing.terrain_stats.process_terrain_stats")
@patch("src.processing.terrain_stats.save_terrain_stats")
def test_main(mock_save, mock_process, sample_dem_dataset, tmp_path):
    """Test the main function."""
    # Set up mocks
    mock_process.return_value = sample_dem_dataset
    output_path = tmp_path / "terrain-stats.tif"
    mock_save.return_value = output_path
    
    # Execute main function
    result = main("dummy_path.tif", output_path=output_path)
    
    # Check result
    assert result == output_path
    mock_process.assert_called_once()
    mock_save.assert_called_once()
