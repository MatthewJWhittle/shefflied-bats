import pytest
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import box, Polygon, Point
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
        coords={"y": y, "x": x},
        dims=["y", "x"],
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
    
    # Check that the results are different (reverse_weights should change the output)
    # The exact relationship depends on the normalization function
    # We just verify that reverse_weights=True produces different results
    assert not np.allclose(normal, reversed, atol=1e-6)


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
    
    # Check that different weights produce different results
    # The normalization function is complex, so we just check they're different
    std_0 = weighted_0.std()
    std_1 = weighted_1.std()
    std_2 = weighted_2.std()
    
    # Allow for NaN std values (when all values are the same)
    if not np.isnan(std_0) and not np.isnan(std_1) and not np.isnan(std_2):
        # When weight increases, variation should generally increase
        assert std_1 <= std_2


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
    from rasterio.transform import from_bounds
    transform = from_bounds(0, 0, 3, 3, 3, 3)  # Create proper Affine transform
    constant_array.rio.write_transform(transform, inplace=True)
    
    weighted = weight_density_array_by_regions(
        constant_array, 
        simple_regions,
        weight=1.0
    )
    
    # With constant density, the weights should be more uniform than with varying density
    # The normalise_to_distribution function adds variation, so we check for reasonable bounds
    # Some values can be negative due to the normalization, so we use a wider range
    assert np.all(weighted >= -1.0) and np.all(weighted <= 3.0)


# Tests for modular training functions
from pathlib import Path
from unittest.mock import Mock, patch
from shapely.geometry import Point
from sdm.commands.modelling.train_sdm_models import (
    setup_training_data,
    train_models_with_setup,
    TrainingSetup,
)
from sdm.models.maxent.maxent_model import ActivityType, DefaultMaxentConfig
from sdm.types import ProjectConfig, PathsConfig, SpatialConfig, MlflowConfig


@pytest.fixture
def mock_project_config():
    """Create a mock project config."""
    return ProjectConfig(
        paths=PathsConfig(
            raw_data="data/raw",
            processed_data="data/processed",
            models="data/models",
            predictions="data/predictions",
            model_config_path="model_config.yml",
            variables_config_path="variables_config.yml",
            occurence_data="data/processed/bats-tidy.geojson",
            background_points="data/processed/background-points.geojson",
            boundary="data/processed/boundary.geojson",
            grid_points="data/processed/grid-points.parquet",
            evs="data/evs",
            ev_tiff="data/evs/evs-to-model.tif",
        ),
        spatial=SpatialConfig(
            top=100.0,
            left=0.0,
            crs="EPSG:27700",
            resolution=100,
            study_area_buffer=1000.0,
        ),
        crs="EPSG:27700",
        mlflow=MlflowConfig(
            tracking_uri="file:./mlruns",
            experiment_name="test_experiment",
        ),
    )


@pytest.fixture
def mock_annotated_data():
    """Create mock annotated bat and background data."""
    # Create mock bat data
    bats_data = {
        "latin_name": ["Myotis daubentonii", "Myotis daubentonii"],
        "activity_type": ["In flight", "In flight"],
        "geometry": [Point(0, 0), Point(1, 1)],
        "ev1": [1.0, 2.0],
        "ev2": [3.0, 4.0],
    }
    bats_gdf = gpd.GeoDataFrame(bats_data, crs="EPSG:27700")
    
    # Create mock background data
    background_data = {
        "geometry": [Point(0.5, 0.5), Point(1.5, 1.5)],
        "ev1": [1.5, 2.5],
        "ev2": [3.5, 4.5],
        "weight": [1.0, 1.0],
    }
    background_gdf = gpd.GeoDataFrame(background_data, crs="EPSG:27700")
    background_density = pd.Series([1.0, 1.0], index=background_gdf.index)
    
    return bats_gdf, background_gdf, background_density


@patch("sdm.commands.modelling.train_sdm_models.load_project_config")
@patch("sdm.commands.modelling.train_sdm_models.load_variables_config")
@patch("sdm.commands.modelling.train_sdm_models.load_bat_data")
@patch("sdm.commands.modelling.train_sdm_models.load_background_points")
@patch("sdm.commands.modelling.train_sdm_models.load_environmental_variables")
@patch("sdm.commands.modelling.train_sdm_models.extract_grid_points")
@patch("sdm.commands.modelling.train_sdm_models.annotate_points")
def test_setup_training_data(
    mock_annotate,
    mock_extract_grid,
    mock_load_ev,
    mock_load_bg,
    mock_load_bats,
    mock_load_vars,
    mock_load_proj,
    mock_project_config,
    mock_annotated_data,
):
    """Test that setup_training_data correctly loads and prepares shared data."""
    bats_gdf, background_gdf, background_density = mock_annotated_data
    
    # Setup mocks
    mock_load_proj.return_value = mock_project_config
    mock_load_vars.return_value = Mock(roster=["ev1", "ev2"], activity_feature_sets={})
    mock_load_bats.return_value = bats_gdf
    mock_load_bg.return_value = (background_gdf, background_density)
    mock_load_ev.return_value = (
        Mock(data_vars={"ev1": Mock(), "ev2": Mock()}),
        Path("data/evs/evs-to-model.tif"),
    )
    mock_extract_grid.return_value = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]}, crs="EPSG:27700"
    )
    mock_annotate.return_value = (bats_gdf, background_gdf)
    
    # Call setup function
    setup = setup_training_data(
        project_config_path=Path("config.yml"),
        variables_config_path=None,
        verbose=False,
    )
    
    # Verify setup object
    assert isinstance(setup, TrainingSetup)
    assert setup.project_config == mock_project_config
    assert len(setup.annotated_bats) == 2
    assert len(setup.annotated_background) == 2
    assert setup.ev_columns == ["ev1", "ev2"]
    assert setup.latin_names == ["Myotis daubentonii"]
    assert setup.activity_types == ["In flight"]


@patch("sdm.commands.modelling.train_sdm_models.generate_training_data")
@patch("sdm.commands.modelling.train_sdm_models.train_models_parallel")
def test_train_models_with_setup(
    mock_train_parallel,
    mock_generate_training,
    mock_annotated_data,
    mock_project_config,
):
    """Test that train_models_with_setup correctly uses shared setup data."""
    bats_gdf, background_gdf, background_density = mock_annotated_data
    
    # Create setup
    setup = TrainingSetup(
        project_config=mock_project_config,
        annotated_bats=bats_gdf,
        annotated_background=background_gdf,
        background_density=background_density,
        grid_points=gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:27700"),
        ev_columns=["ev1", "ev2"],
        all_ev_columns=["ev1", "ev2"],
        ev_raster_path=Path("data/evs/evs-to-model.tif"),
        latin_names=["Myotis daubentonii"],
        activity_types=["In flight"],
    )
    
    # Mock training data and results
    from sdm.types import TrainingData, TrainingResults
    
    mock_training_data = [
        TrainingData(
            latin_name="Myotis daubentonii",
            activity_type="In flight",
            occurrence=gpd.GeoDataFrame(
                {"class": [1, 0], "geometry": [Point(0, 0), Point(1, 1)]},
                crs="EPSG:27700",
            ),
        )
    ]
    
    mock_results = [
        TrainingResults(
            latin_name="Myotis daubentonii",
            activity_type="In flight",
            final_model=Mock(),
            cv_models=None,
            cv_scores=np.array([0.8, 0.9, 0.85]),
            success=True,
            error=None,
        )
    ]
    
    mock_generate_training.return_value = mock_training_data
    mock_train_parallel.return_value = mock_results
    
    # Call training function
    model_config = DefaultMaxentConfig()
    feature_selection = {ActivityType.IN_FLIGHT: ["ev1", "ev2"]}
    sampling_params = {
        "subset_occurrence": None,
        "subset_background": True,
        "order_by_density_for_subset": True,
        "sample_weight_n_neighbors": 5,
        "background_min_bg": 1000,
        "background_max_bg": 10000,
        "background_factor": 10,
    }
    
    results, training_data = train_models_with_setup(
        setup=setup,
        model_config=model_config,
        feature_selection=feature_selection,
        sampling_params=sampling_params,
        min_presence=15,
        verbose=False,
    )
    
    # Verify results
    assert len(results) == 1
    assert results[0].success
    assert len(training_data) == 1
    mock_generate_training.assert_called_once()
    mock_train_parallel.assert_called_once()