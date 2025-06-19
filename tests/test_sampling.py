import pytest
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box, Polygon
from rasterio.transform import Affine

from sdm.occurrence.sampling import (
    generate_background_points,
    generate_background_points_from_data,
    weight_density_array_by_regions,
    TransformMethod,
    BackgroundMethod
)

@pytest.fixture
def config() -> dict:
    """Create a sample grid."""
    return {
        "grid_resolution": 100,
        "n_points": 100,
        "n_regions": 4,
        "n_background_points": 10,
    }



@pytest.fixture
def sample_boundary(config: dict) -> gpd.GeoDataFrame:
    """Create a simple square boundary."""
    geometry = box(0, 0, config["grid_resolution"] * 10, config["grid_resolution"] * 10)
    return gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:27700")



@pytest.fixture
def sample_occurrences(
    sample_boundary: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    """Create a sample occurrence dataset."""
    n_points = config["n_points"]
    presence_gdf : gpd.GeoDataFrame = sample_boundary.sample_points(size=n_points).to_frame() # type: ignore
    presence_gdf = presence_gdf.explode() # type: ignore
    
    return presence_gdf

@pytest.fixture
def sample_regions(
    sample_boundary: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    """Create a sample grid with 4 quadrants."""
    
    boundary_bounds = sample_boundary.total_bounds
    xmin, ymin, xmax, ymax = boundary_bounds

    # Calculate midpoints to create quadrants
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    # Create 4 quadrants
    regions_polygons = [
        box(xmin, ymid, xmid, ymax),    # Top left
        box(xmid, ymid, xmax, ymax),    # Top right
        box(xmin, ymin, xmid, ymid),    # Bottom left
        box(xmid, ymin, xmax, ymid)     # Bottom right
    ]

    # Create GeoDataFrame with quadrants
    regions = gpd.GeoDataFrame(
        data={
            "region_id": [f"region_{i}" for i in range(len(regions_polygons))]
        },
        geometry=regions_polygons,
        crs="EPSG:27700"
    )
    
    return regions


@pytest.fixture
def sample_density_array(
    config: dict,
    sample_boundary: gpd.GeoDataFrame,
) -> xr.DataArray:
    """
    Create a sample density array.
    """
    # create a sample density array
    boundary_bounds = sample_boundary.total_bounds
    xmin, ymin, xmax, ymax = boundary_bounds
    width = xmax - xmin
    height = ymax - ymin
    x_n = int(width / config["grid_resolution"]) + 1
    y_n = int(height / config["grid_resolution"]) + 1
    x = np.linspace(xmin, xmax, x_n)
    y = np.linspace(ymin, ymax, y_n)


    # create a random density array
    density_array = np.random.rand(y_n, x_n)
    # create a dataarray
    density_array = xr.DataArray(
        data=density_array,
        coords={"x": x, "y": y},
        dims=["x", "y"],
    )
    density_array.rio.write_crs(sample_boundary.crs, inplace=True)

    return density_array


def test_weight_density_array_by_regions(
    sample_density_array: xr.DataArray,
    sample_regions: gpd.GeoDataFrame,
):
    """Test the weight_density_array_by_regions function."""
    weighted_density_array = weight_density_array_by_regions(
        density_array=sample_density_array,
        regions=sample_regions,
    )
    assert isinstance(weighted_density_array, xr.DataArray)
    assert weighted_density_array.shape == sample_density_array.shape
    assert weighted_density_array.rio.crs == sample_density_array.rio.crs
    assert weighted_density_array.rio.bounds() == sample_density_array.rio.bounds() # type: ignore


def test_generate_background_points_from_data(sample_occurrences, sample_boundary):
    """Test the main in-memory background points generation function."""
    n_points = 10
    bg_points, density_raster = generate_background_points_from_data(
        occurrence_data=sample_occurrences,
        boundary=sample_boundary,
        n_background_points=n_points,
        background_method=BackgroundMethod.CONTRAST,
        background_value=0.3,
        sigma=1.0,
        transform_method=TransformMethod.PRESENCE
    )
    
    assert isinstance(bg_points, gpd.GeoDataFrame)
    assert len(bg_points) == n_points
    assert bg_points.crs == sample_occurrences.crs
    assert "presence" in bg_points.columns
    assert all(bg_points["presence"] == 0)



def test_orientation_of_density_array(sample_occurrences, sample_boundary):
    """Test that the density array is oriented correctly."""
    bg_points, density_raster = generate_background_points_from_data(
        occurrence_data=sample_occurrences,
        boundary=sample_boundary,
        n_background_points=100,
        background_value=0.0,
    )

    # The array could be flipped in the y direction if we haven't been careful
    # All the background points should be within 100m of an occurrence

    # Check that all the background points are within 100m of an occurrence
    for bg_point in bg_points.geometry:
        assert any(sample_occurrences.distance(bg_point) < 250)

    



def test_generate_background_points(
        tmp_path,
        sample_occurrences,
        sample_boundary,
        sample_regions,
        config,
    ):
    """Test the file-based interface."""
    # Create test data with proper spatial extent

    # Save test data
    occurrence_path = tmp_path / "occurrences.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    regions_path = tmp_path / "regions.geojson"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_occurrences.to_file(occurrence_path, driver="GeoJSON")
    sample_boundary.to_file(boundary_path, driver="GeoJSON")
    sample_regions.to_file(regions_path, driver="GeoJSON")

    # Generate background points
    bg_points_path, density_raster_path = generate_background_points(
        occurrence_data_path=occurrence_path,
        boundary_path=boundary_path,
        output_dir_for_density_raster=output_dir,
        regions_path=regions_path,
        n_background_points=config["n_background_points"],
        grid_resolution=config["grid_resolution"]
    )
    assert bg_points_path.exists()
    assert density_raster_path.exists()
    
def test_generate_background_points_with_stratification(
    sample_occurrences: gpd.GeoDataFrame,
    sample_boundary: gpd.GeoDataFrame,
    sample_regions: gpd.GeoDataFrame,
    config: dict,
):
    """Test background point generation with stratification by regions."""
    
    # Generate background points with stratification
    bg_points, density_raster = generate_background_points_from_data(
        occurrence_data=sample_occurrences,
        boundary=sample_boundary,
        regions=sample_regions,
        n_background_points=config["n_background_points"],
        background_method=BackgroundMethod.CONTRAST,
        background_value=0.3,
        sigma=1.0,
        transform_method=TransformMethod.PRESENCE
    )
    
    # Basic checks
    assert isinstance(bg_points, gpd.GeoDataFrame)
    assert len(bg_points) == config["n_background_points"]
    assert bg_points.crs == sample_regions.crs
    assert "presence" in bg_points.columns
    assert all(bg_points["presence"] == 0)

    # for each region, count the occurrences and background points
    # regions with more occurrences should have more background points

    value_counts = []
    for i, region in sample_regions.iterrows():
        geometry = region.geometry
        region_occurrences = sample_occurrences[sample_occurrences.intersects(geometry)]
        region_bg_points = bg_points[bg_points.intersects(geometry)] # type: ignore
        
        value_counts.append({
            "region_id": i,
            "region_occurrences": len(region_occurrences),
            "region_bg_points": len(region_bg_points),
        })
    # TODO: build a check that the regions with more occurrences have more background points
    # maybe compar with and without stratification?

@pytest.fixture
def simple_density_array():
    """Create a simple 3x3 density array for testing."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    da = xr.DataArray(
        data,
        coords={
            'x': np.array([0, 1, 2]),
            'y': np.array([0, 1, 2])
        },
        dims=['y', 'x']
    )
    # Set up the CRS and transform for the array
    da = da.rio.write_crs("EPSG:4326")
    # Set a simple transform (1 unit per pixel)
    transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    da.rio.write_transform(transform, inplace=True)
    return da


@pytest.fixture
def simple_regions():
    """Create two simple regions for testing."""
    regions = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 1, 1),  # Bottom left region
            box(1, 1, 2, 2)   # Top right region
        ],
        crs="EPSG:4326"
    )
    return regions


def test_basic_weighting(simple_density_array, simple_regions):
    """Test basic weighting functionality."""
    weighted = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=1.0
    )
    
    # Check that the output has the same shape as input
    assert weighted.shape == simple_density_array.shape
    
    # Check that the output is not identical to input (weights were applied)
    assert not np.array_equal(weighted, simple_density_array)
    
    # Check that no NaN values were introduced
    assert not np.isnan(weighted).any()


def test_reverse_weights(simple_density_array, simple_regions):
    """Test that reverse_weights parameter works correctly."""
    # Get weights with and without reversal
    normal = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=1.0,
        reverse_weights=False
    )
    
    reversed = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=1.0,
        reverse_weights=True
    )
    
    # Check that the results are different
    assert not np.array_equal(normal, reversed)
    
    # Check that the relative ordering is reversed
    # This is a bit of a simplification, but should work for our test case
    normal_region1 = normal[0:2, 0:2].mean()  # Bottom left region
    normal_region2 = normal[1:3, 1:3].mean()  # Top right region
    
    reversed_region1 = reversed[0:2, 0:2].mean()
    reversed_region2 = reversed[1:3, 1:3].mean()
    
    if normal_region1 > normal_region2:
        assert reversed_region1 < reversed_region2
    else:
        assert reversed_region1 > reversed_region2


def test_weight_parameter(simple_density_array, simple_regions):
    """Test that the weight parameter affects the result."""
    # Test with different weight values
    weighted_0 = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=0.0
    )
    
    weighted_1 = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=1.0
    )
    
    weighted_2 = weight_density_array_by_regions(
        simple_density_array, 
        simple_regions,
        weight=2.0
    )
    
    # Check that different weights produce different results
    assert not np.array_equal(weighted_0, weighted_1)
    assert not np.array_equal(weighted_1, weighted_2)
    
    # Check that weight=0 produces more uniform weights
    # (less variation between regions)
    std_0 = weighted_0.std()
    std_1 = weighted_1.std()
    std_2 = weighted_2.std()
    
    assert std_0 <= std_1 <= std_2


def test_background_region(simple_density_array, simple_regions):
    """Test that areas not in any region are handled correctly."""
    # Create a region that only covers part of the array
    partial_region = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)],  # Only covers bottom left
        crs="EPSG:4326"
    )
    
    weighted = weight_density_array_by_regions(
        simple_density_array, 
        partial_region,
        weight=1.0
    )
    
    # Check that the background region (not covered by any region)
    # has been weighted appropriately
    background_mean = weighted[1:3, 1:3].mean()  # Top right area
    region_mean = weighted[0:2, 0:2].mean()      # Bottom left area
    
    # The means should be different since they're in different regions
    assert not np.isclose(background_mean, region_mean)


def test_empty_regions(simple_density_array):
    """Test behavior with empty regions."""
    empty_regions = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    with pytest.raises(ValueError):
        weight_density_array_by_regions(
            simple_density_array, 
            empty_regions,
            weight=1.0
        )


def test_reserved_column_name(simple_density_array, simple_regions):
    """Test that using reserved column name raises error."""
    regions_with_reserved = simple_regions.copy()
    regions_with_reserved["_region_id"] = ["test1", "test2"]
    
    with pytest.raises(ValueError):
        weight_density_array_by_regions(
            simple_density_array, 
            regions_with_reserved,
            weight=1.0
        )


def test_constant_density(simple_regions):
    """Test behavior with constant density values."""
    constant_data = np.ones((3, 3))
    constant_array = xr.DataArray(
        constant_data,
        coords={
            'x': np.array([0, 1, 2]),
            'y': np.array([0, 1, 2])
        },
        dims=['y', 'x']
    )
    # Set up the CRS and transform for the array
    constant_array = constant_array.rio.write_crs("EPSG:4326")
    transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    constant_array.rio.write_transform(transform, inplace=True)
    
    weighted = weight_density_array_by_regions(
        constant_array, 
        simple_regions,
        weight=1.0
    )
    
    # With constant density, all regions should have the same weight
    # (after normalization)
    assert np.allclose(weighted, 1.0, atol=1e-10)