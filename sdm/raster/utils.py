import logging
from pathlib import Path
from typing import Union, Tuple, Optional, List
import math

import geopandas as gpd
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import rioxarray as rxr # import unused but needed to enable rio attributes of xarray
import xarray as xr 
from shapely.geometry import box, Polygon




from sdm.utils.io import load_boundary, load_spatial_config

from sdm.utils.io import (
    load_boundary, 
    load_spatial_config 
)

def rasterise_gdf(gdf:gpd.GeoDataFrame, resolution:float, output_file:str, bbox=None):
    # Define the raster size and transform
    # Here, I'm assuming a 1x1 meter resolution and using the bounds of the GeoDataFrame
    if bbox is None:
        x_min, y_min, x_max, y_max = gdf.total_bounds
    else: 
        x_min, y_min, x_max, y_max = bbox

    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Create a mask: rasterize the GeoDataFrame. This gives a value of True where the geometry covers a square
    mask = geometry_mask(gdf.geometry, transform=transform, invert=True, out_shape=(height, width))
    # Convert the boolean mask to uint8 (or another supported data type)
    mask = mask.astype('uint8')

    # Write the mask to a raster file
    with rio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs=gdf.crs,
        transform=transform,
        nodata=0,
    ) as dest:
        dest.write(mask, 1)

    return output_file



def aggregate_raster(input_file, output_file, scale_factor):
    with rio.open(input_file) as src:
        # Read the data
        data = src.read(1)
        
        # Calculate the shape of the destination array
        dst_height = int(data.shape[0] // scale_factor)
        dst_width = int(data.shape[1] // scale_factor)
        
        # Aggregate data to new resolution
        aggregated_data = data.reshape((dst_height, scale_factor, dst_width, scale_factor)).sum(axis=(1, 3))
        
        # Create destination transform
        dst_transform = src.transform * src.transform.scale(
            (src.width / dst_width),
            (src.height / dst_height)
        )
        
        # Write the new raster
        with rio.open(
            output_file,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=aggregated_data.dtype,
            crs=src.crs,
            transform=dst_transform,
        ) as dest:
            dest.write(aggregated_data, 1)




# Generate a grid of points
def generate_point_grid(
    bbox: Tuple[float, float, float, float],
    resolution: float,
    crs: Union[str, int, dict]
) -> gpd.GeoDataFrame:
    """
    Generates a grid of points within a given bounding box and resolution.

    Args:
        bbox (tuple): A tuple of four floats representing the bounding box coordinates in the order (xmin, ymin, xmax, ymax).
        resolution (float): The resolution of the grid, in the same units as the bounding box coordinates.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the generated grid of points.

    """
    xmin, ymin, xmax, ymax = bbox
    x_coords = np.arange(xmin + resolution / 2, xmax, resolution)
    y_coords = np.arange(ymin + resolution / 2, ymax, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.flatten(), yy.flatten()), crs=crs)
    grid["x"] = xx.flatten()
    grid["y"] = yy.flatten()
    return grid



def tile_bounding_box(xmin, ymin, xmax, ymax, tile_shape, resolution):
    width = int(tile_shape[0] * resolution)
    height = int(tile_shape[1] * resolution)

    # Calculate padding needed for x and y dimensions
    x_padding = width - ((xmax - xmin) % width)
    y_padding = height - ((ymax - ymin) % height)

    # Update xmax and ymax to include padding
    xmax += x_padding
    ymax += y_padding

    x_tiles = (xmax - xmin) // width
    y_tiles = (ymax - ymin) // height

    tiles = []
    for i in range(int(y_tiles)):
        for j in range(int(x_tiles)):
            tile_xmin = xmin + j * width
            tile_xmax = tile_xmin + width
            tile_ymin = ymin + i * height
            tile_ymax = tile_ymin + height
            tiles.append((tile_xmin, tile_ymin, tile_xmax, tile_ymax))

    return tiles

def reproject_data(
    array: Union[xr.DataArray, xr.Dataset],
    crs: Union[str, int, dict],
    transform: Affine,
    resolution: float,
    resampling: Resampling = Resampling.bilinear,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Reprojects the given array to a specified coordinate reference system (CRS) and transform.
    """
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
    def __init__(self, 
                 tile_size: Tuple[float, float],
                 origin: Tuple[float, float],
                 ):
        self.tile_size_x = tile_size[0]
        self.tile_size_y = tile_size[1]
        self.origin_x = origin[0]
        self.origin_y = origin[1]
    
    def pad_and_align(
            self,
            bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = bbox
        # Align to tile grid based on origin and tile size
        xmin = self.origin_x + math.floor((xmin - self.origin_x) / self.tile_size_x) * self.tile_size_x
        ymin = self.origin_y + math.floor((ymin - self.origin_y) / self.tile_size_y) * self.tile_size_y
        xmax = self.origin_x + math.ceil((xmax - self.origin_x) / self.tile_size_x) * self.tile_size_x
        ymax = self.origin_y + math.ceil((ymax - self.origin_y) / self.tile_size_y) * self.tile_size_y
        return xmin, ymin, xmax, ymax

    def tile_bbox(
        self, 
        bbox: Tuple[float, float, float, float], 
    ) -> list[Tuple[float, float, float, float]]: # Changed List to list
        xmin, ymin, xmax, ymax = self.pad_and_align(bbox)
        tiles = []
        for cur_y in np.arange(ymin, ymax, self.tile_size_y):
            for cur_x in np.arange(xmin, xmax, self.tile_size_x):
                tiles.append((cur_x, cur_y, cur_x + self.tile_size_x, cur_y + self.tile_size_y))
        return tiles
    
def squeeze_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Squeeze a Dataset by dropping extra dimensions."""
    for var in ds.data_vars:
        ds[var] = ds[var].squeeze()
    if "band" in ds.dims:
        ds = ds.drop_dims("band")
    return ds



def calculate_offset(x: float, y: float, origin: Tuple[float,float], resolution: float) -> Tuple[float, float]:
    """Calculates the offset of a point from an origin based on a given resolution."""
    x_origin, y_origin = origin
    offset_x = (x - x_origin) % resolution
    offset_y = (y - y_origin) % resolution
    return offset_x, offset_y

def align_to_grid(
    x: np.ndarray, 
    y: np.ndarray, 
    origin: Tuple[float,float], 
    resolution: float, 
    decimals: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligns coordinates to a grid defined by an origin and resolution."""
    x_origin, y_origin = origin
    offset_x = (x - float(x_origin)) % float(resolution)
    offset_y = (y - float(y_origin)) % float(resolution)
    aligned_x = x - offset_x
    aligned_y = y - offset_y
    aligned_x = np.round(aligned_x, decimals=decimals)
    aligned_y = np.round(aligned_y, decimals=decimals)
    return aligned_x, aligned_y

def points_to_grid_squares(
    points: gpd.GeoSeries, 
    grid_size: float = 100.0, 
    origin: Tuple[float, float] = (0.0, 0.0)
) -> gpd.GeoSeries:
    """
    Converts a GeoSeries of points to a GeoSeries of grid squares aligned to an origin.
    """
    if not isinstance(points, gpd.GeoSeries):
        raise TypeError("Input 'points' must be a GeoPandas GeoSeries.")
    if points.empty:
        return gpd.GeoSeries([], crs=points.crs) # Linter might flag this
    if not all(points.geom_type == 'Point'):
         raise ValueError("All geometries in 'points' must be of type Point.")

    x_coords = np.asarray(points.x) # Explicit cast to ndarray
    y_coords = np.asarray(points.y) # Explicit cast to ndarray

    x_mins, y_mins = align_to_grid(x_coords, y_coords, origin, grid_size, decimals=1) 
    
    x_maxs = x_mins + grid_size
    y_maxs = y_mins + grid_size

    grid_squares_geoms: List[Polygon] = [] # Explicit type for the list
    for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs):
        grid_squares_geoms.append(box(x_min, y_min, x_max, y_max))
    
    # Linter might still complain about gpd.GeoSeries constructor.
    return gpd.GeoSeries(grid_squares_geoms, crs=points.crs, index=points.index)




def construct_transform_shift_bounds(bounds: tuple, resolution: float) -> tuple[Affine, tuple]:
    """
    Construct a transform based on the bounds and resolution.

    Parameters:
    bounds (tuple): The bounds of the boundary.
    resolution (float): The spatial resolution.

    Returns:
    tuple: A tuple containing the transform and the bounds (shifted to be a factor of the resolution).
    """
    # get the left and the top
    xmin, ymin, xmax, ymax = bounds
    # move the left to the left so it is a factor of the resolution
    xmin = xmin - (xmin % resolution)
    # move the top to the top so it is a factor of the resolution
    ymax = ymax + (resolution - (ymax % resolution))
    # move the right to the right so it is a factor of the resolution
    xmax = xmax + (resolution - (xmax % resolution))
    # move the bottom to the bottom so it is a factor of the resolution
    ymin = ymin - (ymin % resolution)

    # calculate the widht and height
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    transform = from_bounds(
        west=xmin, south=ymin, east=xmax, north=ymax, width=width, height=height
    )
    return transform, (xmin, ymin, xmax, ymax)

def generate_model_grid(
    boundary_path: Union[str, Path], resolution: Optional[int] = None
) -> Tuple[xr.DataArray, Tuple]:
    """Generate a model grid based on the study area boundary.

    Args:
        boundary_path: Path to the boundary file
        resolution: Resolution for the grid, uses spatial config if None

    Returns:
        Tuple of (grid, bounds) where grid is an xarray DataArray
        and bounds is a tuple of (minx, miny, maxx, maxy)
    """
    # Load boundary
    boundary = load_boundary(boundary_path)
    # Load spatial config
    spatial_config = load_spatial_config()

    if resolution is None:
        grid_resolution: int = spatial_config["resolution"]
    else:
        grid_resolution = resolution

    # Get model transform and bounds
    _, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), grid_resolution
    )

    # Create coordinate arrays
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx + grid_resolution / 2, maxx, grid_resolution)
    y_coords = np.arange(maxy - grid_resolution / 2, miny, -grid_resolution)

    # Create empty grid with coordinates
    grid = xr.DataArray(
        np.zeros((len(y_coords), len(x_coords))),
        coords={"y": y_coords, "x": x_coords},
        dims=["y", "x"],
    )
    grid = grid.rio.write_crs(spatial_config["crs"])

    return grid, bounds