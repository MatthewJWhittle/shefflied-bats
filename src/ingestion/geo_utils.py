"Module containing utility functions for geospatial data processing."
from typing import Union, Tuple, List
import math

from rasterio.enums import Resampling
from affine import Affine
import rioxarray as rxr
import xarray as xr
import numpy as np


def reproject_data(
    array: Union[xr.DataArray, xr.Dataset], 
    crs: Union[int, str], 
    transform: Affine, 
    resolution: float,
    resampling: Resampling = Resampling.bilinear,
) -> Union[xr.DataArray, xr.Dataset]:
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
    if isinstance(reprojected, xr.DataArray):
        reprojected = reprojected.where(reprojected != reprojected.rio.nodata, np.nan)
        reprojected.rio.write_nodata(np.nan, inplace=True)
    elif isinstance(reprojected, xr.Dataset):
        for var in reprojected.data_vars:
            reprojected[var] = reprojected[var].where(
                reprojected[var] != reprojected[var].rio.nodata, np.nan
            ).rio.write_nodata(np.nan)
    else:
        raise ValueError("Unexpected output type from reproject method")
        
    return reprojected



class BoxTiler:
    """
    Class for tiling a bounding box into smaller tiles.
    """
    def __init__(self, 
                 tile_size: Tuple[float, float],
                 origin: Tuple[float, float],
                 ):
        """
        Initialize the BoxTiler class.
        
        Args:
            tile_size: Tuple of (width, height) for the tiles.
            origin: Tuple of (x, y) for the origin of the grid
            
        """
        self.tile_size = tile_size
        self.origin = origin
    
    def pad_and_align(
            self,
            bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Pads and aligns a bounding box to the tile grid.

        Args:
            bbox: Tuple of (minx, miny, maxx, maxy) coordinates.

        Returns:
            Tuple of (minx, miny, maxx, maxy) for the padded and aligned box.
        """
        minx, miny, maxx, maxy = bbox
        tile_w, tile_h = self.tile_size
        origin_x, origin_y = self.origin

        # Align to the tile grid
        start_x = math.floor((minx - origin_x) / tile_w) * tile_w + origin_x
        start_y = math.floor((miny - origin_y) / tile_h) * tile_h + origin_y
        end_x = math.ceil((maxx - origin_x) / tile_w) * tile_w + origin_x
        end_y = math.ceil((maxy - origin_y) / tile_h) * tile_h + origin_y

        return start_x, start_y, end_x, end_y


    def tile_bbox(
        self, 
        bbox: Tuple[float, float, float, float], 
    ) -> List[Tuple[float, float, float, float]]:
        """
        Splits a bounding box into standardized tiles.

        Creates a set of tiles that align with a global grid to ensure consistent
        tiling across different requests.

        Args:
            bbox: Tuple of (minx, miny, maxx, maxy) coordinates.
            tile_size: Tuple of (width, height) for the tiles.

        Returns:
            List of tuples, each containing (minx, miny, maxx, maxy) for a tile.
        """
        tile_w, tile_h = self.tile_size

        # Pad and align the bounding box to the tile grid
        start_x, start_y, end_x, end_y = self.pad_and_align(bbox)

        standard_tiles = []
        for y in np.arange(start_y, end_y, tile_h):
            for x in np.arange(start_x, end_x, tile_w):
                tile_bbox = (x, y, x + tile_w, y + tile_h)
                standard_tiles.append(tile_bbox)

        return standard_tiles
    

