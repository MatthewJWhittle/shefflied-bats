from pathlib import Path
from typing import Tuple, Union, Optional
import logging

import numpy as np
import xarray as xr
import richdem as rd # type: ignore # Add type ignore if linter complains about missing stubs
import rioxarray as rxr # For loading DEM

from .utils import squeeze_dataset # For save_terrain_stats

# Helper to convert xarray DataArray to RichDEM array and back, handling nodata
def _xr_to_rdarray(data_array: xr.DataArray) -> rd.rdarray:
    """Converts an xarray.DataArray to a richdem.rdarray, preserving nodata."""
    # Ensure data is float for RichDEM, common for DEMs
    # RichDEM might handle other types, but float32/64 is safest for elevation.
    if not np.issubdtype(data_array.dtype, np.floating):
        data_array = data_array.astype(np.float32)
    
    # RichDEM expects a NumPy array and a no_data value.
    # If data_array.rio.nodata is None, RichDEM might default or error.
    # It's safer to ensure a nodata value is passed if one is intended.
    # If original data had no nodata, rd.rdarray(..., no_data=None) might be okay,
    # but often DEMs have fill values.
    nodata_val = data_array.rio.nodata
    if nodata_val is None and np.issubdtype(data_array.dtype, np.floating):
        # If float and no explicit nodata, check for NaNs and use a common fill if any found
        # This is a heuristic; ideally, nodata is explicit.
        if np.isnan(data_array.data).any():
            nodata_val = -9999.0 # A common float fill value
            data_array = data_array.fillna(nodata_val) # Fill NaNs before passing to RichDEM
    elif nodata_val is not None and np.isnan(nodata_val) and not np.issubdtype(data_array.dtype, np.floating):
        # If nodata is NaN but data is int, RichDEM will fail. Choose an int fill.
        nodata_val = -9999 # Common int fill value
        # data_array should already have this value, or needs conversion.

    return rd.rdarray(data_array.data, no_data=nodata_val)

def _rdarray_to_xr(rd_array: rd.rdarray, template_xr: xr.DataArray, name: str) -> xr.DataArray:
    """Converts a richdem.rdarray back to an xarray.DataArray using a template for coords/attrs."""
    # Create DataArray, then assign coords and CRS from template
    # Ensure data type is appropriate (e.g. float32 for many derivatives)
    data = rd_array.data.astype(np.float32) if not np.issubdtype(rd_array.data.dtype, np.floating) else rd_array.data
    
    # Handle nodata from RichDEM if it's set
    # If rd_array.no_data is a value, replace it with np.nan for float outputs if appropriate
    if rd_array.no_data is not None and np.issubdtype(data.dtype, np.floating):
        data[data == rd_array.no_data] = np.nan
        nodata_to_write = np.nan
    elif rd_array.no_data is not None:
        nodata_to_write = rd_array.no_data
    else:
        nodata_to_write = np.nan # Default to NaN for float if no nodata from RichDEM

    new_da = xr.DataArray(
        data,
        coords=template_xr.coords,
        dims=template_xr.dims,
        name=name,
        attrs=template_xr.attrs.copy() # Copy attributes
    )
    new_da.rio.write_crs(template_xr.rio.crs, inplace=True)
    new_da.rio.write_transform(template_xr.rio.transform(), inplace=True)
    new_da.rio.write_nodata(nodata_to_write, inplace=True)
    return new_da

def calculate_slope_aspect_rd(
    dem_rd: rd.rdarray, 
    slope_units: str = 'radians' # 'degrees', 'percent', 'radians'
) -> Tuple[rd.rdarray, rd.rdarray]:
    slope = rd.TerrainAttribute(dem_rd, attrib=f'slope_{slope_units}')
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect') # Aspect is typically in degrees
    return slope, aspect

def calculate_aspect_components_from_rd_aspect(aspect_rd: rd.rdarray) -> Tuple[np.ndarray, np.ndarray]:
    # Aspect from RichDEM is in degrees. Convert to radians for trig functions.
    aspect_rad = np.deg2rad(aspect_rd.data)
    aspect_eastness = np.cos(aspect_rad)
    aspect_northness = np.sin(aspect_rad)
    # Handle nodata if present in aspect_rd.data
    if aspect_rd.no_data is not None:
        nodata_mask = (aspect_rd.data == aspect_rd.no_data)
        aspect_eastness[nodata_mask] = np.nan # Or another suitable nodata float
        aspect_northness[nodata_mask] = np.nan
    return aspect_eastness, aspect_northness

def calculate_twi_rd(dem_rd: rd.rdarray, slope_rd_radians: rd.rdarray) -> rd.rdarray:
    slope_data_safe = slope_rd_radians.data.copy()
    slope_data_safe[slope_data_safe < 0.001] = 0.001 # Avoid division by zero
    if slope_rd_radians.no_data is not None:
        slope_data_safe[slope_rd_radians.data == slope_rd_radians.no_data] = np.nan # Propagate nodata

    flow_acc_rd = rd.FlowAccumulation(dem_rd, method='D8')
    
    # TWI calculation, ensuring alignment and handling nodata
    # (specific area = flow_acc * cell_area, but often simplified with just flow_acc for relative TWI)
    # Assuming cell area is constant and absorbed into interpretation or scaling.
    twi_data = np.full_like(flow_acc_rd.data, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(slope_data_safe) & ~np.isnan(flow_acc_rd.data)
    if flow_acc_rd.no_data is not None:
        valid_mask &= (flow_acc_rd.data != flow_acc_rd.no_data)
    
    twi_data[valid_mask] = np.log(flow_acc_rd.data[valid_mask] / slope_data_safe[valid_mask])
    
    # Create a new rdarray for TWI
    twi_rd = rd.rdarray(twi_data, no_data=np.nan)
    # Copy georeferencing from dem_rd if necessary, though not strictly an rd.TerrainAttribute result
    twi_rd.geotransform = dem_rd.geotransform 
    # twi_rd.projection = dem_rd.projection # if available/needed
    return twi_rd

def calculate_curvature_rd(dem_rd: rd.rdarray, curvature_type: str = 'planform') -> rd.rdarray:
    # RichDEM supports: 'profile', 'planform', 'tangential', 'mean', 'total' (or just 'curvature' for general)
    return rd.TerrainAttribute(dem_rd, attrib=curvature_type + '_curvature')

# Functions operating on xarray DataArrays directly
def calculate_roughness_xr(slope_da: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    return slope_da.rolling(x=window_size, y=window_size, center=True).std()

def calculate_tpi_xr(dem_da: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    # Ensure center=True for TPI to be relative to central pixel
    mean_elevation = dem_da.rolling(x=window_size, y=window_size, center=True).mean()
    return dem_da - mean_elevation

def calculate_weighted_aspect_xr(
    slope_da: xr.DataArray, 
    aspect_eastness_da: xr.DataArray, 
    aspect_northness_da: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    weighted_eastness = slope_da * aspect_eastness_da
    weighted_northness = slope_da * aspect_northness_da
    return weighted_eastness.rename("aspect_eastness_slope"), weighted_northness.rename("aspect_northness_slope")

def process_dem_to_terrain_attributes(
    dem_path: Union[str, Path],
    dem_band_index: int = 0, # 0-indexed band for isel
    slope_window_size: int = 3,
    tpi_window_size: int = 3,
    output_slope_units: str = 'radians'
) -> xr.Dataset:
    logging.info(f"Processing DEM: {dem_path} (band index: {dem_band_index})")
    try:
        dem_xr = rxr.open_rasterio(dem_path, masked=True).isel(band=dem_band_index)
        dem_xr.rio.write_nodata(dem_xr.rio.nodata, inplace=True) # Ensure nodata is set
    except Exception as e:
        logging.error(f"Failed to load DEM from {dem_path}: {e}")
        raise

    dem_rd = _xr_to_rdarray(dem_xr)
    terrain_ds = dem_xr.to_dataset(name="dem")

    logging.info("Calculating slope and aspect...")
    slope_rd, aspect_rd = calculate_slope_aspect_rd(dem_rd, slope_units=output_slope_units)
    terrain_ds["slope"] = _rdarray_to_xr(slope_rd, dem_xr, "slope")
    
    logging.info("Calculating aspect components (eastness, northness)...")
    aspect_east_np, aspect_north_np = calculate_aspect_components_from_rd_aspect(aspect_rd)
    terrain_ds["aspect_eastness"] = _rdarray_to_xr(rd.rdarray(aspect_east_np, no_data=np.nan), dem_xr, "aspect_eastness")
    terrain_ds["aspect_northness"] = _rdarray_to_xr(rd.rdarray(aspect_north_np, no_data=np.nan), dem_xr, "aspect_northness")

    logging.info("Calculating Topographic Wetness Index (TWI)...")
    twi_rd = calculate_twi_rd(dem_rd, slope_rd) # slope_rd is already in radians if output_slope_units was 'radians'
    terrain_ds["twi"] = _rdarray_to_xr(twi_rd, dem_xr, "twi")

    logging.info("Calculating curvature (planform)...")
    # For other types: 'profile', 'mean', 'tangential'
    curvature_rd = calculate_curvature_rd(dem_rd, curvature_type='planform') 
    terrain_ds["planform_curvature"] = _rdarray_to_xr(curvature_rd, dem_xr, "planform_curvature")

    logging.info("Calculating roughness (std of slope)...")
    terrain_ds["roughness"] = calculate_roughness_xr(terrain_ds["slope"], window_size=slope_window_size)

    logging.info("Calculating Topographic Position Index (TPI)...")
    terrain_ds["tpi"] = calculate_tpi_xr(terrain_ds["dem"], window_size=tpi_window_size)

    logging.info("Calculating slope-weighted aspect components...")
    east_weighted, north_weighted = calculate_weighted_aspect_xr(
        terrain_ds["slope"], terrain_ds["aspect_eastness"], terrain_ds["aspect_northness"]
    )
    terrain_ds["aspect_eastness_slope"] = east_weighted
    terrain_ds["aspect_northness_slope"] = north_weighted
    
    # Ensure all variables have consistent nodata (e.g. np.nan for float types)
    for var_name in terrain_ds.data_vars:
        if np.issubdtype(terrain_ds[var_name].dtype, np.floating):
            terrain_ds[var_name] = terrain_ds[var_name].rio.write_nodata(np.nan, encoded=True) # encoded=True might be needed

    logging.info("Terrain attribute calculation complete.")
    return terrain_ds

def save_terrain_dataset(
    terrain_ds: xr.Dataset, 
    output_path: Union[str, Path],
    drop_dem_variable: bool = True
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_to_save = terrain_ds.copy()
    if drop_dem_variable and "dem" in dataset_to_save.data_vars:
        dataset_to_save = dataset_to_save.drop_vars("dem")
        logging.info("Dropped 'dem' variable before saving terrain stats.")
    
    dataset_to_save = squeeze_dataset(dataset_to_save)
    try:
        logging.info(f"Saving terrain dataset to: {output_path}")
        dataset_to_save.rio.to_raster(output_path)
    except Exception as e:
        logging.error(f"Failed to save terrain dataset to {output_path}: {e}")
        raise
    return output_path 