import pytest
import xarray as xr
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import inspect
import rioxarray as rxr

from sdm.commands.data_preparation.environmental.generate_climate_data import generate_climate_data
from sdm.data.climate import calculate_climate_statistics, assign_climate_variable_names, write_climate_data


@pytest.fixture
def sample_boundary():
    """Create a sample boundary for testing."""
    geometry = box(0, 0, 1000, 1000)  # 1km x 1km square
    return gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:27700")


def test_generate_climate_data_with_missing_boundary(tmp_path):
    """Test error handling for missing boundary file."""
    boundary_path = tmp_path / "nonexistent.geojson"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    
    with pytest.raises(Exception):  # Will be FileNotFoundError or similar
        generate_climate_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            worldclim_cache_dir=cache_dir
        )


def test_calculate_climate_statistics(tmp_path):
    """Test climate statistics calculation function."""
    # Create multi-band climate data (12 months)
    temp_data = xr.DataArray(
        np.random.rand(12, 5, 5),
        dims=['band', 'y', 'x'],
        coords={'band': range(12), 'y': range(5), 'x': range(5)},
        name='tavg'
    )
    temp_data = temp_data.rio.write_crs('EPSG:27700')
    
    prec_data = xr.DataArray(
        np.random.rand(12, 5, 5),
        dims=['band', 'y', 'x'],
        coords={'band': range(12), 'y': range(5), 'x': range(5)},
        name='prec'
    )
    prec_data = prec_data.rio.write_crs('EPSG:27700')
    
    wind_data = xr.DataArray(
        np.random.rand(12, 5, 5),
        dims=['band', 'y', 'x'],
        coords={'band': range(12), 'y': range(5), 'x': range(5)},
        name='wind'
    )
    wind_data = wind_data.rio.write_crs('EPSG:27700')
    
    # Test the function
    climate_stats = calculate_climate_statistics(temp_data, prec_data, wind_data)
    
    # Verify output is an xarray Dataset
    assert isinstance(climate_stats, xr.Dataset)
    
    # Check the dataset structure
    expected_vars = [
        'temp_ann_var', 'temp_ann_avg', 'temp_mat_avg',
        'prec_ann_var', 'prec_ann_avg', 'wind_ann_var', 'wind_ann_avg'
    ]
    for var in expected_vars:
        assert var in climate_stats.data_vars
    
    # Test serialization separately
    result_path = tmp_path / "climate_stats.tif"
    climate_stats.rio.to_raster(result_path)
    assert result_path.exists()
    
    # Check the output structure
    stats = rxr.open_rasterio(result_path)
    assert stats.shape[0] == 7  # 7 bands


def test_climate_data_processing_functions(tmp_path):
    """Test other climate data processing functions."""
    
    # Create test data
    data = xr.DataArray(
        np.random.rand(5, 5),
        dims=['y', 'x'],
        coords={'y': range(5), 'x': range(5)},
        name='test_var'
    )
    datasets = {'test_var': data}
    
    # Test assign_climate_variable_names
    named_datasets = assign_climate_variable_names(datasets)
    assert 'test_var' in named_datasets
    
    # Test write_climate_data
    output_paths = write_climate_data(datasets, tmp_path)
    assert 'test_var' in output_paths
    assert Path(output_paths['test_var']).exists()


def test_generate_climate_data_hardcoded_variables():
    """Test that variables are hardcoded correctly."""
    # Test that the function uses hardcoded variables
    sig = inspect.signature(generate_climate_data)
    
    # Check that variables parameter no longer exists
    assert 'variables' not in sig.parameters
    
    # The function now hardcodes variables = ["bio", "tavg", "prec", "wind"]
