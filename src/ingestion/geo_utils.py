"Module containing utility functions for geospatial data processing."
from typing import Union

from rasterio.enums import Resampling
from affine import Affine
import rioxarray as rxr
import xarray as xr
import numpy as np


def reproject_data(
    array: xr.DataArray, crs: Union[int, str], transform: Affine, resolution: float
) -> xr.DataArray:
    """
    Reprojects the given array to a specified coordinate reference system (CRS) and transform.

    Parameters:
    array (xarray.DataArray): The input data array to be reprojected.
    crs (str): The target coordinate reference system.
    transform (affine.Affine): The affine transformation to apply.
    resolution (float): The spatial resolution.

    Returns:
        xarray.DataArray: The reprojected data array with nodata values set to NaN.
    """
    # Resampling needs to be done in two steps - not sure why
    # the first step is to reproject to a higher resolution in the target crs
    # the second step is to align it to the transform so all rasters align to the same grid
    reprojected = array.rio.reproject(
        crs, resolution=resolution, resampling=Resampling.bilinear
    )
    reprojected = reprojected.rio.reproject(
        crs, resampling=Resampling.bilinear, transform=transform
    )
    reprojected = reprojected.where(reprojected != reprojected.rio.nodata, np.nan)
    reprojected.rio.write_nodata(np.nan, inplace=True)
    return reprojected
