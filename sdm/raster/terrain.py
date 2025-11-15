from pathlib import Path
from typing import Tuple, Union
import logging

import numpy as np
import xarray as xr
import rioxarray as rxr

from .utils import squeeze_dataset
from ..data.terrain.stats import twi_from_array  # your fast D8+flats TWI

logger = logging.getLogger(__name__)


def _ensure_float32_nan(arr: np.ndarray, nodata: float | None = None) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if nodata is not None and not np.isnan(nodata):
        a = np.where(a == nodata, np.nan, a)
    return a


def calculate_slope(dem_data: np.ndarray, cell_size: float) -> np.ndarray:
    """Slope angle β in radians using central differences; tanβ = √(dzdx²+dzdy²)."""
    gy_img, gx = np.gradient(dem_data, cell_size)  # gy increases south
    gy = -gy_img  # flip so +y points north
    tanbeta = np.hypot(gx, gy)
    slope = np.arctan(tanbeta).astype(np.float32)
    slope[~np.isfinite(dem_data)] = np.nan
    return slope


def calculate_aspect(dem_data: np.ndarray, cell_size: float, flat_eps: float = 1e-6) -> np.ndarray:
    """
    Aspect in radians, 0 = East, increasing CCW (East→North→West→South).
    Undefined (NaN) where slope < flat_eps.
    """
    gy_img, gx = np.gradient(dem_data, cell_size)
    gy = -gy_img  # north-positive
    aspect = np.arctan2(gy, gx)
    aspect = (aspect + 2 * np.pi) % (2 * np.pi)
    tanbeta = np.hypot(gx, gy)
    aspect[tanbeta < flat_eps] = np.nan
    aspect[~np.isfinite(dem_data)] = np.nan
    return aspect.astype(np.float32)


def calculate_aspect_components(aspect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eastness = cos(aspect), Northness = sin(aspect); aspect measured from East CCW."""
    eastness = np.cos(aspect).astype(np.float32)
    northness = np.sin(aspect).astype(np.float32)
    eastness[~np.isfinite(aspect)] = np.nan
    northness[~np.isfinite(aspect)] = np.nan
    return eastness, northness


def calculate_twi(dem_data: np.ndarray, slope: np.ndarray, cell_size: float) -> np.ndarray:
    """TWI via fast D8 pipeline using provided slope parameter."""
    dem_f32 = dem_data.astype(np.float32, copy=False)
    # Convert slope (in radians) to tan(slope) for TWI calculation
    tan_slope = np.tan(slope).astype(np.float32)
    twi = twi_from_array(dem_f32, cellsize=cell_size, slope_eps=1e-6, do_fill=True, flats_tol=0.0, tan_slope=tan_slope)
    return twi


def calculate_curvature(dem_data: np.ndarray, cell_size: float) -> np.ndarray:
    """
    Laplacian curvature (κ ≈ d²z/dx² + d²z/dy²).
    If you want profile curvature, use the commented formula below.
    """
    gy_img, gx = np.gradient(dem_data, cell_size)
    gy = -gy_img
    gy2, _ = np.gradient(gy, cell_size)
    _, gx2 = np.gradient(gx, cell_size)
    curvature = (gx2 + gy2).astype(np.float32)
    curvature[~np.isfinite(dem_data)] = np.nan

    # --- Optional: profile curvature (commented)
    # tanb = np.hypot(gx, gy)
    # denom = (1.0 + tanb**2) ** 1.5
    # curvature = ((gx2 + gy2) / np.maximum(denom, 1e-12)).astype(np.float32)

    return curvature


def calculate_roughness(slope_data: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    """Roughness = rolling std of slope (in chosen units)."""
    return slope_data.rolling(x=window_size, y=window_size, center=True, min_periods=1).std()


def calculate_tpi(dem_data: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    """Topographic Position Index: elevation minus neighbourhood mean."""
    mean_elev = dem_data.rolling(x=window_size, y=window_size, center=True, min_periods=1).mean()
    return dem_data - mean_elev


def calculate_weighted_aspect(
    slope_da: xr.DataArray,
    aspect_eastness_da: xr.DataArray,
    aspect_northness_da: xr.DataArray,
    slope_units: str = "radians",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Weight aspect components by steepness. Use sin(slope) (0 on flats, 1 at 90°).
    slope_units: 'radians' | 'degrees'
    """
    if slope_units == "degrees":
        slope_rad = np.deg2rad(slope_da)
    else:
        slope_rad = slope_da

    w = np.sin(slope_rad)  # bounded, unitless
    return w * aspect_eastness_da, w * aspect_northness_da


def process_dem_to_terrain_attributes(
    dem_path: Union[str, Path],
    dem_band_index: int = 0,
    slope_window_size: int = 3,
    tpi_window_size: int = 3,
    output_slope_units: str = "radians",
) -> xr.Dataset:
    """
    Compute terrain attributes without RichDEM.
    - Slope returned in 'output_slope_units' ('radians' | 'degrees' | 'percent').
    - Aspect (radians, 0=East CCW), eastness, northness.
    - TWI (D8), Laplacian curvature, roughness (std of slope), TPI.
    """
    dem_rxr = rxr.open_rasterio(dem_path)
    dem_rxr = dem_rxr.isel(band=dem_band_index)

    # Ensure NoData -> NaN in the working array
    nodata = dem_rxr.rio.nodata
    dem_np = _ensure_float32_nan(dem_rxr.values, nodata)

    transform = dem_rxr.rio.transform()
    cell_size = abs(transform[0])  # assumes square pixels

    # Compute in radians
    slope_rad = calculate_slope(dem_np, cell_size)
    aspect = calculate_aspect(dem_np, cell_size)
    eastness, northness = calculate_aspect_components(aspect)
    twi = calculate_twi(dem_np, slope_rad, cell_size)
    curvature = calculate_curvature(dem_np, cell_size)

    # Prepare dataset (inherits CRS/transform from dem_rxr)
    ds = dem_rxr.to_dataset(name="dem")
    ds["slope"] = (("y", "x"), slope_rad)
    ds["aspect"] = (("y", "x"), aspect)
    ds["aspect_eastness"] = (("y", "x"), eastness)
    ds["aspect_northness"] = (("y", "x"), northness)
    ds["twi"] = (("y", "x"), twi)
    ds["curvature_laplacian"] = (("y", "x"), curvature)

    # Neighbourhood stats
    ds["roughness"] = calculate_roughness(ds.slope, slope_window_size)
    ds["tpi"] = calculate_tpi(ds.dem, tpi_window_size)

    # Weighted aspect uses slope in radians regardless of output unit
    we, wn = calculate_weighted_aspect(ds.slope, ds.aspect_eastness, ds.aspect_northness, "radians")
    ds["aspect_eastness_slope"] = we
    ds["aspect_northness_slope"] = wn

    # Convert the *stored* slope to requested units last
    if output_slope_units == "degrees":
        ds["slope"] = xr.apply_ufunc(np.degrees, ds.slope)
    elif output_slope_units == "percent":
        ds["slope"] = xr.apply_ufunc(lambda s: np.tan(s) * 100.0, ds.slope)

    # keep attributes/crs via rioxarray
    return ds


def save_terrain_dataset(
    terrain_ds: xr.Dataset,
    output_path: Union[str, Path],
    drop_dem_variable: bool = True,
) -> Path:
    """Save terrain dataset to GeoTIFF (multi-band)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if drop_dem_variable and "dem" in terrain_ds.data_vars:
        out = terrain_ds.drop_vars("dem")
    else:
        out = terrain_ds

    out = squeeze_dataset(out)
    # optional: set dtype/encoding per band here if you want float32:
    # for v in out.data_vars:
    #     out[v].encoding.update({"dtype": "float32", "zlevel": 6})
    out.rio.to_raster(output_path)
    return output_path
