"""
Module for generating background points for species distribution modeling.
"""

from pathlib import Path
import logging
import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
import rioxarray as rxr
from typing import Union, Tuple, Optional

from data_prep.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)
from data_prep.utils.config import setup_logging


def generate_model_grid(
    boundary_path: Union[str, Path], resolution: Optional[float] = None
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
        resolution: int = spatial_config["resolution"]
    # Get model transform and bounds
    _, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), resolution
    )

    # Create coordinate arrays
    minx, miny, maxx, maxy = bounds
    x_coords = np.arange(minx + resolution / 2, maxx, resolution)
    y_coords = np.arange(maxy - resolution / 2, miny, -resolution)

    # Create empty grid with coordinates
    grid = xr.DataArray(
        np.zeros((len(y_coords), len(x_coords))),
        coords={"y": y_coords, "x": x_coords},
        dims=["y", "x"],
    )
    grid = grid.rio.write_crs(spatial_config["crs"])

    return grid, bounds


def generate_background_points(
    occurrence_data_path: Union[str, Path],
    boundary_path: Union[str, Path],
    output_dir: Union[str, Path],
    n_background_points: int = 10000,
    background_p: float = 0.1,
    sigma: float = 1.5,
    resolution: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Generate background points based on density-smoothed occurrence data.

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON or Parquet)
        boundary_path: Path to boundary data
        output_dir: Directory to save outputs
        n_background_points: Number of background points to generate
        background_p: Minimum probability for areas with few/no occurrences
        sigma: Sigma value for Gaussian smoothing
        resolution: Resolution of the model grid

    Returns:
        GeoDataFrame of background points
    """
    output_dir = Path(output_dir)
    boundary_path = Path(boundary_path)
    # Load occurrence data
    if str(occurrence_data_path).endswith(".parquet"):
        occurrences = gpd.read_parquet(occurrence_data_path)
    else:
        occurrences = gpd.read_file(occurrence_data_path)

    # Ensure occurrences are in target CRS
    spatial_config = load_spatial_config()
    boundary = load_boundary(boundary_path, buffer_distance=0)
    occurrences = occurrences.to_crs(spatial_config["crs"])

    # Generate model grid
    grid, bounds = generate_model_grid(boundary_path, resolution)

    # Get grid coordinates
    grid_x = grid.coords["x"].values
    grid_y = grid.coords["y"].values

    # Calculate resolution
    resolution_x = grid_x[1] - grid_x[0]
    resolution_y = grid_y[0] - grid_y[1]  # Y is descending

    # Create bounding box and filter occurrences
    bbox_poly = box(*bounds)
    occurrences = occurrences[occurrences.intersects(bbox_poly)]

    logging.info(
        f"Using %s occurrence points to generate density surface", len(occurrences)
    )

    # Generate density array based on histogram of occurrences
    point_counts, bin_x, bin_y = np.histogram2d(
        occurrences.geometry.x,
        occurrences.geometry.y,
        bins=(grid_x, grid_y[::-1]),  # Reverse y to match grid order
    )

    # Apply Gaussian smoothing
    point_counts = gaussian_filter(point_counts, sigma=sigma)

    # Convert to xarray
    array_x = bin_x[:-1]
    array_y = bin_y[:-1]
    density_array = xr.DataArray(
        point_counts, coords={"x": array_x, "y": array_y}
    ).transpose("y", "x")
    density_array = density_array.rio.write_crs(spatial_config["crs"])

    # Set minimum probability for areas with few/no occurrences
    floor_probability = np.max(density_array) * background_p
    density_array = density_array.where(
        density_array > floor_probability, floor_probability
    )

    # Normalize the density array to sum to 1
    density_array = density_array / density_array.sum()

    ## Write the density array to a raster
    density_array.rio.to_raster(output_dir / "bat_density.tif")

    # Convert to dataframe for sampling
    density_df = density_array.to_dataframe(name="density")
    density_df = density_df.reset_index()

    # Sample points based on density
    logging.info(f"Sampling %s background points", n_background_points)
    samples_idx = np.random.choice(
        density_df.index.values,
        size=n_background_points,
        p=density_array.values.flatten(),
    )
    samples = density_df[["x", "y"]].loc[samples_idx]

    # Add random jitter within grid cells
    x_range = (-resolution_x / 2, resolution_x / 2)
    y_range = (-resolution_y / 2, resolution_y / 2)
    x_noise = np.random.uniform(*x_range, size=n_background_points)
    y_noise = np.random.uniform(*y_range, size=n_background_points)
    samples["x"] = samples["x"] + x_noise
    samples["y"] = samples["y"] + y_noise

    # Convert to GeoDataFrame
    background_points = gpd.GeoDataFrame(
        samples,
        geometry=gpd.points_from_xy(samples.x, samples.y),
        crs=spatial_config["crs"],
    )
    # sjoin with boundary to remove points outside
    background_points = gpd.sjoin(
        background_points, boundary[["geometry"]], how="inner", predicate="intersects"
    )
    background_points = background_points.drop(columns="index_right")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save density raster for inspection
    density_array.rio.to_raster(output_dir / "bat_density.tif")

    # Save background points
    background_points.to_parquet(output_dir / "background-points.parquet")
    background_points.to_file(
        output_dir / "background-points.geojson", driver="GeoJSON"
    )

    logging.info(f"Background points saved to %s", output_dir)
    return background_points


def main(
    occurrence_data_path: Union[str, Path],
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    output_dir: Union[str, Path] = "data/processed",
    n_background_points: int = 40000,
    background_p: float = 0.1,
    resolution: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Main function to generate background points.

    Args:
        occurrence_data_path: Path to occurrence data
        boundary_path: Path to boundary data
        output_dir: Directory to save outputs
        n_background_points: Number of background points to generate
        background_p: Minimum probability for areas with few/no occurrences
        resolution: Resolution of the model grid

    Returns:
        GeoDataFrame of background points
    """
    setup_logging()
    logging.info("Generating background points...")

    background_points = generate_background_points(
        occurrence_data_path,
        boundary_path,
        output_dir,
        n_background_points,
        background_p,
        resolution=resolution,
    )

    logging.info(f"Generated %s background points", len(background_points))
    return background_points


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate background points for SDM")
    parser.add_argument(
        "occurrence_path", type=str, help="Path to occurrence data (GeoJSON or Parquet)"
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="data/processed/boundary.geojson",
        help="Path to boundary file",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of background points to generate",
    )
    parser.add_argument(
        "--background-p",
        type=float,
        default=0.1,
        help="Minimum probability for areas with few/no occurrences",
    )

    args = parser.parse_args()

    main(
        args.occurrence_path,
        args.boundary,
        args.output,
        args.n_points,
        args.background_p,
    )
