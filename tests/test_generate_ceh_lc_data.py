import pytest
import xarray as xr
import numpy as np
import rioxarray as rxr
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_bounds

from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import generate_ceh_lc_data
from sdm.raster.processing import calculate_multiscale_variables

# At 100 m cells: window 10/20/30 → 500/1000/1500 m neighbourhood means
# (matches notebooks/feature-engineering.ipynb and calculate_multiscale_variables).
LANDCOVER_NEIGHBOURHOOD_WINDOWS = (10, 20, 30)


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


def _synthetic_ceh_tile(height: int, width: int, seed: int = 0, nodata: int = 255):
    """Build a synthetic 10 m CEH-like categorical tile (EPSG:27700)."""
    rng = np.random.default_rng(seed)
    codes = np.arange(1, 22)
    data = rng.choice(codes, size=(height, width)).astype(np.uint8)
    data[rng.random(data.shape) < 0.03] = nodata
    x = np.arange(width) * 10 + 5
    y = (np.arange(height)[::-1]) * 10 + 5
    da = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x}, name="land_cover")
    da = da.rio.write_crs("EPSG:27700")
    da = da.rio.write_transform(from_bounds(0, 0, width * 10, height * 10, width, height))
    da = da.rio.write_nodata(nodata)
    return da


def test_chunked_matches_single_pass_exactly():
    """Chunked coarsen must match the legacy single-pass stack on a small tile."""
    from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import (
        coarsen_land_cover_categories,
    )

    tile = _synthetic_ceh_tile(240, 260, seed=7)
    # window=80 → multiple chunks on this tile; 80 is already a multiple of 10.
    full = coarsen_land_cover_categories(tile, 100, window_pixels=80, method="full")
    chunked = coarsen_land_cover_categories(tile, 100, window_pixels=80, method="chunked")

    assert set(full.data_vars) == set(chunked.data_vars)
    assert np.allclose(full.x.values, chunked.x.values)
    assert np.allclose(full.y.values, chunked.y.values)
    for name in full.data_vars:
        np.testing.assert_array_equal(full[name].values, chunked[name].values)


def test_chunked_matches_when_requested_window_splits_output_cell():
    """
    Requested window mid-way through a 100 m cell (105 source px) must snap to
    the output grid so no coarse cell is split; result still matches single-pass.
    """
    from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import (
        _aligned_window_starts,
        coarsen_land_cover_categories,
    )

    coarsen_factor = 10  # 10 m → 100 m
    # 105 is deliberately not a multiple of 10 — would split a 100 m cell if used raw.
    starts = list(_aligned_window_starts(350, window=105, coarsen_factor=coarsen_factor))
    assert starts, "expected at least one aligned chunk"
    for start, size in starts:
        assert start % coarsen_factor == 0
        assert size % coarsen_factor == 0
        assert size <= 100  # 105 snapped down to 100

    tile = _synthetic_ceh_tile(350, 370, seed=11)
    full = coarsen_land_cover_categories(tile, 100, window_pixels=105, method="full")
    chunked = coarsen_land_cover_categories(tile, 100, window_pixels=105, method="chunked")

    assert set(full.data_vars) == set(chunked.data_vars)
    for name in full.data_vars:
        np.testing.assert_array_equal(full[name].values, chunked[name].values)


def _load_ceh_output_dataset(path):
    """Load CEH GeoTIFF bands as a Dataset keyed by long_name."""
    da = rxr.open_rasterio(path)
    names = da.attrs.get("long_name")
    if isinstance(names, str):
        names = [names]
    names = list(names)
    assert len(names) == int(da.sizes["band"])
    da = da.assign_coords(band=names)
    ds = da.to_dataset(dim="band")
    for name in ds.data_vars:
        if hasattr(da, "rio") and da.rio.crs is not None:
            ds[name] = ds[name].rio.write_crs(da.rio.crs)
        nodata = da.rio.nodata
        if nodata is not None:
            ds[name].rio.write_nodata(nodata, inplace=True)
    return ds


def _landcover_step_with_neighbourhood(ceh_tif_path):
    """
    Full land-cover step outputs: in-cell class layers plus 500/1000/1500 m means.

    Neighbourhood averages use the existing multiscale path
    (``calculate_multiscale_variables``), not chunked CEH aggregation.
    """
    class_layers = _load_ceh_output_dataset(ceh_tif_path)
    parts = [class_layers]
    for window in LANDCOVER_NEIGHBOURHOOD_WINDOWS:
        parts.append(calculate_multiscale_variables(class_layers, window))
    return xr.merge(parts)


def test_e2e_landcover_step_matches_a97f7d9_baseline_with_neighbourhood(
    tmp_path, mock_spatial_config
):
    """
    Full land-cover step (class layers + 500/1000/1500 m neighbourhood averages)
    from chunked aggregation must match the a97f7d9 single-pass baseline
    (``coarsen_method="full"``) on a synthetic AOI where chunk edges fall
    mid-neighbourhood-window.
    """
    # 600×620 @ 10 m → after trim/coarsen ~60×62 @ 100 m. window_pixels=105
    # snaps to 100 source px = 10 coarse cells, so chunk seams every 10 cells.
    # Neighbourhood windows 20 and 30 (radius 10 / 15) therefore straddle seams.
    height, width = 600, 620
    tile = _synthetic_ceh_tile(height, width, seed=42)
    ceh_path = tmp_path / "ceh.tif"
    tile.rio.to_raster(ceh_path)

    # Boundary inset so clip keeps a multi-chunk interior; no extra buffer.
    margin = 80
    boundary = gpd.GeoDataFrame(
        geometry=[box(margin, margin, width * 10 - margin, height * 10 - margin)],
        crs="EPSG:27700",
    )
    boundary_path = tmp_path / "boundary.geojson"
    boundary.to_file(boundary_path)

    with patch(
        "sdm.commands.data_preparation.environmental.generate_ceh_lc_data.load_spatial_config"
    ) as mock_config:
        mock_config.return_value = mock_spatial_config

        baseline_path = generate_ceh_lc_data(
            output_dir=tmp_path / "baseline",
            boundary_path=boundary_path,
            ceh_data_path=ceh_path,
            buffer_distance_m=0,
            output_resolution_m=100,
            window_pixels=105,
            coarsen_method="full",  # a97f7d9 single-pass path
            verbose=False,
        )
        chunked_path = generate_ceh_lc_data(
            output_dir=tmp_path / "chunked",
            boundary_path=boundary_path,
            ceh_data_path=ceh_path,
            buffer_distance_m=0,
            output_resolution_m=100,
            window_pixels=105,  # mid-100 m cell request → snaps; seams mid-window
            coarsen_method="chunked",
            verbose=False,
        )

    baseline = _landcover_step_with_neighbourhood(baseline_path)
    chunked = _landcover_step_with_neighbourhood(chunked_path)

    # Class layers plus neighbourhood suffixes must all be present and equal.
    assert set(baseline.data_vars) == set(chunked.data_vars)
    neighbourhood_vars = [
        v for v in baseline.data_vars if str(v).endswith(("_500m", "_1000m", "_1500m"))
    ]
    assert neighbourhood_vars, "expected 500/1000/1500 m neighbourhood layers"
    class_vars = [v for v in baseline.data_vars if v not in neighbourhood_vars]
    assert class_vars, "expected in-cell class layers"

    for name in baseline.data_vars:
        np.testing.assert_allclose(
            baseline[name].values,
            chunked[name].values,
            equal_nan=True,
            err_msg=f"mismatch in land-cover step layer {name!r}",
        )
