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
from typing import Union, Tuple, Optional, Literal

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
    background_method: Literal["contrast", "percentile", "scale", "fixed", "binary"] = "contrast",
    background_value: float = 0.3,
    sigma: float = 1.5,
    resolution: Optional[float] = None,
    transform_method: Literal["log", "sqrt", "presence", "cap", "rank"] = "log",
    cap_percentile: float = 90.0,
) -> gpd.GeoDataFrame:
    """Generate background points based on density-smoothed occurrence data.

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON or Parquet)
        boundary_path: Path to boundary data
        output_dir: Directory to save outputs
        n_background_points: Number of background points to generate
        background_method: Method for setting the minimum background probability:
            - "contrast": Set floor based on contrast ratio (most intuitive method)
                A value of 0 means only sample where occurrences exist
                A value of 1 means uniform sampling everywhere
            - "percentile": Set floor to the value at background_value percentile (of ALL values)
            - "scale": Set floor to background_value * max value
            - "fixed": Use background_value as a literal fixed probability
            - "binary": Use presence/absence only (ignores background_value)
        background_value: Value to use with background_method:
            - If "contrast": Ratio (0-1) controlling concentration (default 0.3)
            - If "percentile": Percentile threshold (0-100)
            - If "scale": Proportion of maximum value (0-1)
            - If "fixed": Absolute probability value
            - If "binary": Ignored
        sigma: Sigma value for Gaussian smoothing
        resolution: Resolution of the model grid
        transform_method: Method to transform occurrence counts to reduce bias from Poisson distribution:
            - "log": Log-transform counts (log(1+x))
            - "sqrt": Square root transform counts
            - "presence": Convert to binary presence/absence
            - "cap": Cap counts at a percentile threshold
            - "rank": Use rank-based normalization
        cap_percentile: When using "cap" method, percentile at which to cap counts (default: 90)

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

    # Transform counts to address Poisson distribution issues
    if transform_method == "log":
        # Log-transform to reduce effect of high counts (adding 1 to handle zeros)
        logging.info("Applying log transformation to point counts")
        point_counts = np.log1p(point_counts)
    elif transform_method == "sqrt":
        # Square root transformation (less aggressive than log)
        logging.info("Applying square root transformation to point counts")
        point_counts = np.sqrt(point_counts)
    elif transform_method == "presence":
        # Convert to binary presence/absence
        logging.info("Converting counts to binary presence/absence")
        point_counts = (point_counts > 0).astype(float)
    elif transform_method == "cap":
        # Cap counts at a percentile threshold
        logging.info(f"Capping counts at {cap_percentile}th percentile")
        non_zero = point_counts[point_counts > 0]
        if len(non_zero) > 0:
            cap_value = np.percentile(non_zero, cap_percentile)
            point_counts = np.minimum(point_counts, cap_value)
    elif transform_method == "rank":
        # Rank-based normalization
        logging.info("Applying rank-based normalization")
        from scipy.stats import rankdata
        point_counts_flat = point_counts.flatten()
        ranks = rankdata(point_counts_flat) / len(point_counts_flat)
        point_counts = ranks.reshape(point_counts.shape)
    else:
        logging.warning(f"Unknown transform method: {transform_method}, using raw counts")

    # Apply Gaussian smoothing
    point_counts = gaussian_filter(point_counts, sigma=sigma)

    # Convert to xarray
    array_x = bin_x[:-1]
    array_y = bin_y[:-1]
    density_array = xr.DataArray(
        point_counts, coords={"x": array_x, "y": array_y}
    ).transpose("y", "x")
    density_array = density_array.rio.write_crs(spatial_config["crs"])

    # Save the raw density array for inspection
    raw_density = density_array.copy()
    raw_density.rio.to_raster(output_dir / "bat_density_raw.tif")

    # Save stats on density array before applying floor
    original_max = density_array.max().item()
    original_mean = density_array.mean().item()
    original_non_zero_mean = density_array.where(density_array > 0).mean().item()
    original_min = density_array.min().item()
    
    # Apply the background probability floor in a more intuitive way
    if background_method == "contrast":
        # NEW METHOD: Use a contrast ratio to control concentration
        # background_value ranges from 0 (concentrated) to 1 (uniform)
        if background_value < 0 or background_value > 1:
            logging.warning(f"Contrast value should be between 0 and 1, got {background_value}. Clamping to range [0, 1].")
            background_value = max(0, min(1, background_value))
        
        # When background_value = 0, floor_probability = 0 (concentrated)
        # When background_value = 1, floor_probability = max_value (uniform)
        floor_probability = original_max * background_value
        
        logging.info(f"Using contrast method: floor = max * {background_value:.4f} = {floor_probability:.8f}")
    elif background_method == "percentile":
        # Calculate percentile using ALL values, not just non-zero
        all_values = density_array.values.flatten()
        floor_probability = np.percentile(all_values, background_value)
        logging.info(f"Using {background_value}th percentile of ALL values: {floor_probability:.8f}")
    elif background_method == "scale":
        # Original behavior: background as a fraction of max
        floor_probability = np.max(density_array) * background_value
        logging.info(f"Using {background_value:.4f} * max for background: {floor_probability:.8f}")
    elif background_method == "fixed":
        # Use background_value as a literal fixed value
        floor_probability = background_value
        logging.info(f"Using fixed background value: {floor_probability:.8f}")
    elif background_method == "binary":
        # Use binary presence/absence (converts to 1 where there's data, retains 0 elsewhere)
        density_array = density_array > 0
        # No need to add background floor for binary
        floor_probability = None
        logging.info("Using binary presence/absence - no background floor applied")
    else:
        # Default to contrast method
        floor_probability = original_max * 0.3  # Default 0.3 contrast
        logging.info(f"Using default contrast method (0.3): {floor_probability:.8f}")
    
    # Apply the floor probability if not using binary method
    if background_method != "binary" and floor_probability is not None:
        density_array = density_array.where(density_array > floor_probability, floor_probability)
        
        # Log statistics about how the floor changed the distribution
        with_floor_max = density_array.max().item()
        with_floor_mean = density_array.mean().item()
        with_floor_min = density_array.min().item()
        
        logging.info(f"Density stats before floor - min: {original_min:.8f}, mean: {original_mean:.8f}, non-zero mean: {original_non_zero_mean:.8f}, max: {original_max:.8f}")
        logging.info(f"Density stats after floor - min: {with_floor_min:.8f}, mean: {with_floor_mean:.8f}, max: {with_floor_max:.8f}")
        
        # Calculate and log how much of probability mass is in background vs. occurrences
        if with_floor_max > 0:
            # Calculate percent of total probability that's from background
            total_sum = np.sum(density_array.values)
            floor_cells = np.sum(density_array.values == floor_probability)
            floor_contribution = floor_cells * floor_probability
            background_percent = (floor_contribution / total_sum) * 100
            logging.info(f"Background accounts for {background_percent:.1f}% of total probability")
            logging.info(f"Concentration ratio (max/min): {with_floor_max/with_floor_min:.1f}x")

    # Normalize the density array to sum to 1
    density_array = density_array / density_array.sum()

    ## Write the density array to a raster
    density_array.rio.to_raster(output_dir / "bat_density.tif")

    # Convert to dataframe for sampling", "sqrt", "presence", "cap", "rank"] = "log",
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
    background_method: str = "contrast",
    background_value: float = 0.3,
    resolution: Optional[float] = None,
    transform_method: str = "log",
    cap_percentile: float = 90.0,
) -> gpd.GeoDataFrame:
    """Main function to generate background points.

    Args:
        occurrence_data_path: Path to occurrence data
        boundary_path: Path to boundary data
        output_dir: Directory to save outputs
        n_background_points: Number of background points to generate
        background_method: Method for setting background probability ("contrast", "percentile", "scale", "fixed", "binary")
        background_value: Value to use with chosen background method
        resolution: Resolution of the model grid
        transform_method: Method to transform counts ("log", "sqrt", "presence", "cap", "rank")
        cap_percentile: When using "cap" method, percentile at which to cap counts

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
        background_method=background_method,
        background_value=background_value,
        resolution=resolution,
        transform_method=transform_method,
        cap_percentile=cap_percentile,
    )

    logging.info(f"Generated %s background points", len(background_points))
    return background_points


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate background points for SDM")
    parser.add_argument(
        "--occurrence_path", 
        default="data/processed/bats-tidy.geojson",
        type=str, 
        help="Path to occurrence data (GeoJSON or Parquet)"
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="data/processed/boundary.geojson",
        help="Path to boundary file",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed", 
        help="Output directory"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of background points to generate",
    )
    parser.add_argument(
        "--background-method",
        type=str,
        default="contrast",
        choices=["contrast", "percentile", "scale", "fixed", "binary"],
        help="Method for setting the background floor",
    )
    parser.add_argument(
        "--background-value",
        type=float,
        default=0.0005,
        help="Value to use with background method (contrast, percentile, scale factor, or fixed value)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="log",
        choices=["log", "sqrt", "presence", "cap", "rank"],
        help="Method to transform counts to address Poisson distribution bias",
    )
    parser.add_argument(
        "--cap-percentile",
        type=float,
        default=90.0,
        help="When using 'cap' transform, percentile at which to cap counts",
    )

    args = parser.parse_args()

    main(
        args.occurrence_path,
        args.boundary,
        args.output,
        args.n_points,
        background_method=args.background_method,
        background_value=args.background_value,
        transform_method=args.transform,
        cap_percentile=args.cap_percentile,
    )
