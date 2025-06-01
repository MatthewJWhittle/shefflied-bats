from pathlib import Path
from typing import Union, Optional, Tuple

import xarray as xr
import numpy as np
from rasterio.transform import from_bounds
from rasterio.transform import Affine


from sdm.utils.io import load_boundary, load_spatial_config


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