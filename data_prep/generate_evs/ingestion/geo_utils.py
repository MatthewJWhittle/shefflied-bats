"Module containing utility functions for geospatial data processing."
from typing import Union, Tuple, List, Dict
import math
import logging



from rasterio.enums import Resampling
from affine import Affine
import rioxarray as rxr
import xarray as xr
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree



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
    

def squeeze_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Squeeze a Dataset by dropping extra dimensions."""
    for var in ds.data_vars:
        ds[var] = ds[var].squeeze()
    if "band" in ds.dims:
        ds = ds.drop_dims("band")
    return ds



def generate_point_grid(
    bbox: Tuple[float, float, float, float],
    resolution: float,
    crs: str
) -> gpd.GeoDataFrame:
    """Generate a regular grid of points within a bounding box.

    Args:
        bbox: Bounding box coordinates (minx, miny, maxx, maxy).
        resolution: Spacing between points in CRS units.
        crs: Coordinate reference system of the bounding box.

    Returns:
        GeoDataFrame containing points forming a regular grid.
    """
    xmin, ymin, xmax, ymax = bbox
    # Pad the bounding box
    width = xmax - xmin
    height = ymax - ymin
    # Pad the bounding box to make it fit the resolution
    # This ensure that the grid is aligned with the raster
    xmin -= width % resolution
    ymin -= height % resolution
    xmax += width % resolution
    ymax += height % resolution

    # Create a grid of points
    x_coords = np.arange(xmin, xmax, resolution)
    y_coords = np.arange(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    xx = xx.flatten()
    yy = yy.flatten()

    # Create a geo dataframe
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx, yy))
    grid.crs = crs

    # Add x and y columns for easy access
    grid["x"] = xx
    grid["y"] = yy

    return grid

def calculate_distances(
    feature_gdfs: Dict[str, gpd.GeoDataFrame],
    boundary: gpd.GeoDataFrame,
    resolution: int = 100
) -> xr.Dataset:
    """Calculate distance to nearest feature for each feature type.

    Args:
        feature_gdfs: Dictionary of feature name to GeoDataFrame mappings.
        boundary: GeoDataFrame containing the area of interest.
        resolution: Resolution in meters for the output distance grids.

    Returns:
        xarray Dataset containing distance rasters for each feature type.
    """
    logging.info("Calculating distance matrices at %dm resolution", resolution)
    
    bbox = tuple(boundary.total_bounds)
    grid = generate_point_grid(bbox, resolution, boundary.crs)
    logging.info("Generated point grid with %d points", len(grid))

    grid_points = np.array(grid[["x", "y"]])
    for name, gdf in feature_gdfs.items():
        logging.info("Calculating distances to %s features", name)
        feature_points = np.array(
            [[geom.x, geom.y] for geom in gdf.geometry.centroid]
        )
        tree = cKDTree(feature_points)
        grid[f"distance_to_{name}"] = tree.query(grid_points, k=1)[0]
        logging.debug("Completed distance calculation for %s", name)

    logging.info("Converting distance grid to xarray")
    distance_array = (
        grid.sort_values(["y", "x"])
        .set_index(["y", "x"])
        .to_xarray()
        .rio.write_crs(boundary.crs)
        .drop_vars(["geometry"])
    )
    # log the na values
    logging.debug("NA values for distance array: %.2f%%",
                  100 * distance_array.isnull().mean())
    logging.info("Distance matrix calculation complete")

    return distance_array
