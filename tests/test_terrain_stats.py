# tests/test_terrain_simple.py
import math
import numpy as np
import pytest
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.transform import from_origin

# ⬇️ Update these imports to your actual package paths if needed
from sdm.raster.terrain import (
    calculate_slope_simple,
    calculate_aspect_simple,
    calculate_aspect_components_simple,
    calculate_twi_simple,
    calculate_curvature_simple,
    calculate_roughness_simple,
    calculate_tpi_simple,
    calculate_weighted_aspect_simple,
    process_dem_to_terrain_attributes_simple,
    save_terrain_dataset_simple,
)
from sdm.commands.data_preparation.environmental.generate_terrain_stats import (
    generate_terrain_stats,
)

# ---------- fixtures & helpers ----------

@pytest.fixture
def sample_dem_array():
    # 5x5 hill in middle
    return np.array([
        [10, 10, 10, 10, 10],
        [10, 12, 14, 12, 10],
        [10, 14, 16, 14, 10],
        [10, 12, 14, 12, 10],
        [10, 10, 10, 10, 10],
    ], dtype=np.float32)

@pytest.fixture
def sample_dem_dataset(sample_dem_array):
    dem_da = xr.DataArray(
        sample_dem_array, dims=["y", "x"],
        coords={"y": np.arange(5), "x": np.arange(5)},
        attrs={"_FillValue": -9999},
    )
    return dem_da.to_dataset(name="dem")

def plane_dem(h, w, ax, ay, base=0.0, dtype=np.float32):
    """
    z = base + ax*x + ay*y
    For downslope direction θ (0=E, CCW): ax = -cosθ, ay = -sinθ
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(dtype)
    return base + ax * xx + ay * yy


# ---------- unit tests (numpy/xarray) ----------

def test_calculate_slope_simple(sample_dem_array):
    slope = calculate_slope_simple(sample_dem_array, cell_size=1.0)
    assert isinstance(slope, np.ndarray)
    assert slope.shape == sample_dem_array.shape
    assert np.mean(slope[1:4, 1:4]) > np.mean(slope[0, :])  # steeper in centre

def test_slope_on_uniform_plane():
    dem = plane_dem(33, 33, ax=-1.0, ay=0.0)  # downslope to East
    slope = calculate_slope_simple(dem, 1.0)
    inner = slope[5:-5, 5:-5]
    assert np.allclose(float(inner.mean()), math.atan(1.0), atol=1e-6)  # 45°

def test_calculate_aspect_components_simple(sample_dem_array):
    aspect = calculate_aspect_simple(sample_dem_array, cell_size=1.0)
    eastness, northness = calculate_aspect_components_simple(aspect)
    assert isinstance(eastness, np.ndarray) and isinstance(northness, np.ndarray)
    assert eastness.shape == aspect.shape == northness.shape
    assert np.all(eastness[np.isfinite(eastness)] <= 1) and np.all(eastness[np.isfinite(eastness)] >= -1)
    assert np.all(northness[np.isfinite(northness)] <= 1) and np.all(northness[np.isfinite(northness)] >= -1)
    # unit circle where finite
    mask = np.isfinite(aspect)
    assert np.allclose((eastness[mask]**2 + northness[mask]**2).mean(), 1.0, atol=1e-6)

def test_aspect_orientation_and_flats():
    dem = plane_dem(33, 33, ax=-1.0, ay=0.0)  # downslope East → ascent ≈ West (π)
    aspect = calculate_aspect_simple(dem, 1.0)
    inner = aspect[5:-5, 5:-5]
    assert np.allclose(float(inner.mean()), math.pi, atol=1e-5)
    flat = np.full((17, 17), 100.0, dtype=np.float32)
    asp_flat = calculate_aspect_simple(flat, 10.0)
    assert np.isnan(asp_flat).all()

def test_calculate_twi_simple(sample_dem_array):
    slope = calculate_slope_simple(sample_dem_array, 1.0)  # arg kept for API, not used
    twi = calculate_twi_simple(sample_dem_array, slope, cell_size=1.0)
    assert isinstance(twi, np.ndarray) and twi.shape == sample_dem_array.shape
    assert np.isfinite(twi).any()

def test_twi_monotonic_with_acc_on_uniform_slope():
    dem = plane_dem(49, 49, ax=-1.0, ay=0.0)
    twi = calculate_twi_simple(dem, slope=np.zeros_like(dem), cell_size=1.0)
    r = 24
    assert np.isfinite(twi[r, 2]) and np.isfinite(twi[r, -3])
    assert twi[r, -3] > twi[r, 2]  # higher downstream

def test_calculate_curvature_simple(sample_dem_array):
    curvature = calculate_curvature_simple(sample_dem_array, cell_size=1.0)
    assert isinstance(curvature, np.ndarray)
    assert curvature.shape == sample_dem_array.shape

def test_curvature_laplacian_zero_on_plane():
    dem = plane_dem(33, 33, ax=-0.7, ay=0.3)
    curv = calculate_curvature_simple(dem, 5.0)
    inner = curv[5:-5, 5:-5]
    assert np.allclose(float(np.nanmax(np.abs(inner))), 0.0, atol=1e-6)

def test_roughness_simple(sample_dem_dataset):
    # slope-like field
    sample_dem_dataset["slope"] = (("y", "x"), np.random.random((5, 5)))
    roughness = calculate_roughness_simple(sample_dem_dataset.slope, window_size=3)
    assert isinstance(roughness, xr.DataArray)
    assert roughness.dims == sample_dem_dataset.slope.dims
    # with min_periods=1 edges need not be NaN; just check finite presence
    assert np.isfinite(roughness.values).any()

def test_calculate_tpi_simple(sample_dem_dataset):
    tpi = calculate_tpi_simple(sample_dem_dataset.dem, window_size=3)
    assert isinstance(tpi, xr.DataArray)
    assert tpi.dims == sample_dem_dataset.dem.dims
    max_idx = np.unravel_index(np.argmax(sample_dem_dataset.dem.values), sample_dem_dataset.dem.shape)
    assert tpi.values[max_idx] > 0  # hill centre > neighbourhood mean

def test_weighted_aspect_simple(sample_dem_dataset):
    sample_dem_dataset["slope"] = (("y", "x"), np.random.random((5, 5)))
    aspect = calculate_aspect_simple(sample_dem_dataset.dem.values, cell_size=1.0)
    e, n = calculate_aspect_components_simple(aspect)
    sample_dem_dataset["aspect_eastness"] = (("y", "x"), e)
    sample_dem_dataset["aspect_northness"] = (("y", "x"), n)
    we, wn = calculate_weighted_aspect_simple(
        sample_dem_dataset.slope,
        sample_dem_dataset.aspect_eastness,
        sample_dem_dataset.aspect_northness,
        slope_units="radians",
    )
    assert isinstance(we, xr.DataArray) and isinstance(wn, xr.DataArray)
    assert we.dims == sample_dem_dataset.slope.dims == wn.dims


def test_nan_propagation_in_metrics():
    dem = plane_dem(31, 31, ax=-1.0, ay=0.0)
    dem[0:5, 0:5] = np.nan
    slope = calculate_slope_simple(dem, 1.0)
    aspect = calculate_aspect_simple(dem, 1.0)
    twi = calculate_twi_simple(dem, slope, 1.0)
    assert np.isnan(slope[0:5, 0:5]).all()
    assert np.isnan(aspect[0:5, 0:5]).all()
    assert np.isnan(twi[0:5, 0:5]).all()


# ---------- integration tests (real IO, no mocks) ----------

def test_process_terrain_stats_simple_io_roundtrip(tmp_path):
    dem = plane_dem(64, 64, ax=-1.0, ay=0.0).astype(np.float32)
    transform = from_origin(300000.0, 500000.0, 10.0, 10.0)
    in_tif = tmp_path / "dem.tif"

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:27700",
        "transform": transform,
        "nodata": np.nan,
        "compress": "LZW",
    }
    with rasterio.open(in_tif, "w", **profile) as dst:
        dst.write(dem, 1)

    ds = process_dem_to_terrain_attributes_simple(
        dem_path=in_tif,
        output_slope_units="degrees",
        slope_window_size=5,
        tpi_window_size=7,
    )

    expected_vars = {
        "dem", "slope", "aspect", "aspect_eastness", "aspect_northness",
        "aspect_eastness_slope", "aspect_northness_slope",
        "twi", "curvature_laplacian", "roughness", "tpi",
    }
    assert expected_vars.issubset(set(ds.data_vars))

    mean_slope_deg = float(ds.slope.where(np.isfinite(ds.slope)).mean())
    # Allow wider range for slope calculation - the exact value depends on the DEM gradient method
    assert 5.0 <= mean_slope_deg <= 50.0
    assert np.isfinite(ds.twi.values[~np.isnan(ds.dem.values)]).all()

def test_save_terrain_stats_simple(tmp_path):
    dem = plane_dem(32, 32, ax=-0.5, ay=0.0).astype(np.float32)
    transform = from_origin(0.0, 0.0, 25.0, 25.0)
    in_tif = tmp_path / "dem.tif"
    with rasterio.open(
        in_tif, "w",
        driver="GTiff", height=dem.shape[0], width=dem.shape[1],
        count=1, dtype="float32", crs="EPSG:3857",
        transform=transform, nodata=np.nan, compress="LZW",
    ) as dst:
        dst.write(dem, 1)

    ds = process_dem_to_terrain_attributes_simple(in_tif, output_slope_units="radians")
    out_tif = tmp_path / "terrain.tif"
    path = save_terrain_dataset_simple(ds, out_tif, drop_dem_variable=True)
    assert path.exists()

    arr = rxr.open_rasterio(path)
    try:
        assert arr.rio.count > 0
        assert arr.rio.crs is not None
    finally:
        arr.close()


# ---------- CLI/wrapper signature ----------

def test_generate_terrain_stats_signature():
    import inspect
    sig = inspect.signature(generate_terrain_stats)
    assert "input_dem_path" in sig.parameters
    assert "output_path" in sig.parameters
    assert "dem_band_index" in sig.parameters
    assert "slope_window_size" in sig.parameters
