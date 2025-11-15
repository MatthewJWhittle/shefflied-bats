import xarray as xr
import numpy as np

from sdm.data.landcover import (
    get_ceh_land_cover_codes_v2023, 
    define_broad_habitat_categories
)
from sdm.raster.processing import (
    create_binary_raster_from_category, 
    aggregate_categorical_rasters
)


def test_ceh_lc_types():
    """Test that the function returns a dictionary with expected keys and values."""
    lc_types = get_ceh_land_cover_codes_v2023()
    
    # Check type
    assert isinstance(lc_types, dict)
    
    # Check some expected keys and values
    assert "1" in lc_types
    assert lc_types["1"] == "Broadleaved woodland"
    assert "3" in lc_types
    assert lc_types["3"] == "Arable"
    assert "20" in lc_types
    assert lc_types["20"] == "Urban"


def test_create_binary_raster_from_category():
    """Test converting land cover array to binary category."""
    # Create a sample array
    data = np.array([[1, 2, 3], [3, 2, 1], [np.nan, 1, 2]])
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": [0, 1, 2], "x": [0, 1, 2]}
    )
    
    # Add CRS to the data array
    da = da.rio.write_crs("EPSG:27700")
    
    # Test conversion for category 1
    result = create_binary_raster_from_category(da, 1, "Test Category", nodata_val=np.nan)
    
    # Check the result
    assert "Test Category" in result.data_vars
    expected = np.array([[1, 0, 0], [0, 0, 1], [np.nan, 1, 0]])
    np.testing.assert_array_equal(result["Test Category"].values, expected)


def test_define_broad_habitat_categories():
    """Test that broad habitat categories are defined correctly."""
    categories = define_broad_habitat_categories()
    
    # Check type
    assert isinstance(categories, dict)
    
    # Check some expected categories
    assert "Grassland" in categories
    assert "Neutral grassland" in categories["Grassland"]
    assert "Wetland" in categories
    assert "Bog" in categories["Wetland"]


def test_aggregate_categorical_rasters():
    """Test feature engineering on a sample dataset."""
    # Create a sample dataset with habitat types
    data = {
        "Neutral grassland": xr.DataArray(np.ones((3, 3))),
        "Calcareous grassland": xr.DataArray(np.ones((3, 3)) * 2),
        "Bog": xr.DataArray(np.ones((3, 3)) * 3),
        "Fen, Marsh and Swamp": xr.DataArray(np.ones((3, 3)) * 4),
        "Inland rock": xr.DataArray(np.ones((3, 3)) * 5),
        "Urban": xr.DataArray(np.ones((3, 3)) * 6),
    }
    ds = xr.Dataset(data)
    
    # Define aggregation map
    aggregation_map = define_broad_habitat_categories()
    categories_to_drop = ["Inland rock"]
    
    # Apply feature engineering
    result = aggregate_categorical_rasters(ds, aggregation_map, categories_to_drop)
    
    # Check that broad habitats are created and original categories removed
    assert "Grassland" in result.data_vars
    assert "Wetland" in result.data_vars
    assert "Neutral grassland" not in result.data_vars
    assert "Calcareous grassland" not in result.data_vars
    assert "Bog" not in result.data_vars
    assert "Fen, Marsh and Swamp" not in result.data_vars
    
    # Check that specified categories are removed
    assert "Inland rock" not in result.data_vars
    
    # Check that unprocessed categories remain
    assert "Urban" in result.data_vars
    
    # Check values for aggregated categories
    expected_grassland = np.ones((3, 3)) + np.ones((3, 3)) * 2  # Sum of 1 and 2
    expected_wetland = np.ones((3, 3)) * 3 + np.ones((3, 3)) * 4  # Sum of 3 and 4
    np.testing.assert_array_equal(result["Grassland"].values, expected_grassland)
    np.testing.assert_array_equal(result["Wetland"].values, expected_wetland)
