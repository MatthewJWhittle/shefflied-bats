import json
from typing import Union
from pathlib import Path

import geopandas as gpd
from rasterio.transform import from_bounds
from affine import Affine


def load_boundary(
    filepath : Union[str, Path],
    buffer_distance: Union[float, int] = 7000,
    target_crs: Union[str, int] = "EPSG:27700",
) -> gpd.GeoDataFrame:
    """
    Loads a boundary from a file and applies a buffer to its geometry.

    Parameters:
    filepath (str): The path to the file containing the boundary data.
    buffer_distance (float): The buffer distance to apply to the boundary geometry (in units of target_crs).
    target_crs (str): The target coordinate reference system (CRS) to reproject the boundary to.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the boundary with buffered geometry.
    """
    boundary = gpd.read_file(filepath)
    boundary = boundary.to_crs(target_crs)
    boundary["geometry"] = boundary.buffer(buffer_distance)
    return boundary


def load_spatial_config() -> dict:
    with open("config/spatial.json") as f:
        spatial_config = json.load(f)
    
    assert isinstance(spatial_config["resolution"], int), "Resolution must be an integer."

    return spatial_config


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


def load_boundary_and_transform(
        boundary_path: Union[str, Path],
        buffer_distance: Union[float, int] = 7000,
) -> tuple[gpd.GeoDataFrame, Affine, tuple, dict]:
    """
    Load the boundary and construct the model transform.
    """
    spatial_config = load_spatial_config()
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs=spatial_config["crs"]
    )
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), spatial_config["resolution"]
    )
    return boundary, model_transform, bounds, spatial_config