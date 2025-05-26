from pathlib import Path
from typing import Union, Tuple, Optional, Literal
import logging

import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
import rioxarray as rxr # Required for .rio accessor, even if not directly called

from sdm.raster.utils import generate_model_grid
from sdm.utils.io import load_boundary, load_spatial_config

# Note: setup_logging() would typically be called by the script using this library function.

def generate_background_points(
    occurrence_data_path: Union[str, Path],
    boundary_path: Union[str, Path],
    output_dir_for_density_raster: Path, # For saving intermediate density raster
    n_background_points: int = 10000,
    background_method: Literal[
        "contrast", "percentile", "scale", "fixed", "binary"
    ] = "contrast",
    background_value: float = 0.3,
    sigma: float = 1.5,
    grid_resolution: Optional[int] = None, # Renamed from 'resolution' for clarity
    transform_method: Literal["log", "sqrt", "presence", "cap", "rank"] = "log",
    cap_percentile: float = 90.0,
) -> gpd.GeoDataFrame:
    """Generate background points based on density-smoothed occurrence data.

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON or Parquet).
        boundary_path: Path to boundary data.
        output_dir_for_density_raster: Directory to save intermediate density raster.
        n_background_points: Number of background points to generate.
        background_method: Method for setting the minimum background probability.
        background_value: Value to use with background_method.
        sigma: Sigma value for Gaussian smoothing.
        grid_resolution: Resolution of the model grid in CRS units. If None, uses project default.
        transform_method: Method to transform occurrence counts.
        cap_percentile: Percentile for 'cap' transform_method.

    Returns:
        GeoDataFrame of background points.
    """
    occurrence_data_path = Path(occurrence_data_path)
    boundary_path = Path(boundary_path)
    output_dir_for_density_raster.mkdir(parents=True, exist_ok=True)

    # Load occurrence data
    logging.info(f"Loading occurrence data from: {occurrence_data_path}")
    if occurrence_data_path.suffix == ".parquet":
        occurrences = gpd.read_parquet(occurrence_data_path)
    elif occurrence_data_path.suffix in [".geojson", ".gpkg"]:
        occurrences = gpd.read_file(occurrence_data_path)
    else:
        raise ValueError(f"Unsupported occurrence data format: {occurrence_data_path.suffix}")

    # Load spatial config and boundary to determine target CRS for the grid
    spatial_config = load_spatial_config()
    project_crs = spatial_config["crs"]
    # Load boundary without buffer first, just to get its CRS if needed or for context
    # The grid generation will use its own boundary loading with project_crs
    # occurrences = occurrences.to_crs(project_crs) # Ensure occurrences match project CRS early

    logging.info(f"Generating model grid at {grid_resolution or spatial_config.get('resolution', 'default')}m resolution, CRS: {project_crs}")
    grid, grid_bounds = generate_model_grid(boundary_path, project_crs=project_crs, resolution=grid_resolution)

    # Ensure occurrences are in the same CRS as the grid for histogramming
    if occurrences.crs != grid.rio.crs:
        logging.info(f"Re-projecting occurrences from {occurrences.crs} to {grid.rio.crs}")
        occurrences = occurrences.to_crs(grid.rio.crs)

    # Filter occurrences to those within the (potentially larger) grid bounds
    bbox_poly = box(*grid_bounds)
    occurrences_in_grid = occurrences[occurrences.intersects(bbox_poly)].copy() # Use .copy() to avoid SettingWithCopyWarning

    logging.info(
        f"Using {len(occurrences_in_grid)} occurrence points (within grid bounds) to generate density surface."
    )
    if len(occurrences_in_grid) == 0:
        logging.error("No occurrence points found within the grid bounds. Cannot generate background points.")
        # Return an empty GeoDataFrame or raise an error
        return gpd.GeoDataFrame(columns=['geometry'], crs=project_crs) 

    # Generate density array based on histogram of occurrences
    # Bins for histogram2d need to be edges. Grid coords are centers. Adjust.
    res_x = grid.rio.resolution()[0]
    res_y = grid.rio.resolution()[1] # Usually negative
    
    # Histogram bins should be pixel edges
    hist_bins_x = np.arange(grid.x.min() - res_x/2, grid.x.max() + res_x/2 + abs(res_x)/2, res_x) 
    # For y, which is usually descending, adjust accordingly for edges
    hist_bins_y = np.arange(grid.y.min() - abs(res_y)/2, grid.y.max() + abs(res_y)/2 + abs(res_y)/2, abs(res_y))
    # Ensure y bins are in ascending order for np.histogram2d if y coords were descending
    if grid.y.values[0] > grid.y.values[-1]: # y is descending
         hist_bins_y = np.sort(hist_bins_y) # ascending for histogram2d

    point_counts, _, _ = np.histogram2d(
        occurrences_in_grid.geometry.x.values,
        occurrences_in_grid.geometry.y.values,
        bins=(hist_bins_x, hist_bins_y),
    )
    # Histogram2d output needs to be transposed if y was descending in grid
    # And y bins were flipped for histogram2d. The output shape matches (bins_x-1, bins_y-1).
    # We want it to match (grid_y_coords, grid_x_coords)
    if grid.y.values[0] > grid.y.values[-1]: # If original y was descending
        point_counts = np.flipud(point_counts) # Flip to match geo-spatial orientation (top-left origin)
    point_counts = point_counts.T # Transpose to (y,x) like grid
    
    # Transform counts
    if transform_method == "log":
        logging.info("Applying log transformation to point counts")
        point_counts = np.log1p(point_counts)
    # ... (other transform methods from original code) ...
    elif transform_method == "sqrt":
        logging.info("Applying square root transformation to point counts")
        point_counts = np.sqrt(point_counts)
    elif transform_method == "presence":
        logging.info("Converting counts to binary presence/absence")
        point_counts = (point_counts > 0).astype(float)
    elif transform_method == "cap":
        logging.info("Capping counts at %sth percentile", cap_percentile)
        non_zero = point_counts[point_counts > 0]
        if len(non_zero) > 0:
            cap_value = np.percentile(non_zero, cap_percentile)
            point_counts = np.minimum(point_counts, cap_value)
        else:
            logging.warning("No non-zero counts to cap, skipping cap transformation.")
    elif transform_method == "rank":
        logging.info("Applying rank-based normalization")
        from scipy.stats import rankdata # Local import is fine for less common dependency
        point_counts_flat = point_counts.flatten()
        ranks = rankdata(point_counts_flat, method='average') / len(point_counts_flat) # Normalize ranks
        point_counts = ranks.reshape(point_counts.shape)
    else:
        logging.warning(f"Unknown transform method: {transform_method}, using raw counts.")

    # Apply Gaussian smoothing
    logging.info(f"Applying Gaussian smoothing with sigma={sigma}")
    smoothed_counts = gaussian_filter(point_counts, sigma=sigma)

    # Convert to xarray DataArray, aligning with the model grid
    density_array = xr.DataArray(
        smoothed_counts, 
        coords=grid.coords, # Use the model grid's coordinates
        dims=grid.dims,
        name="occurrence_density"
    )
    density_array = density_array.rio.write_crs(grid.rio.crs)
    density_array.rio.write_transform(grid.rio.transform(), inplace=True)
    density_array.rio.write_nodata(np.nan, inplace=True) # Ensure nodata for consistency

    # Save the raw density array for inspection
    density_output_path = output_dir_for_density_raster / "occurrence_density_surface.tif"
    logging.info(f"Saving occurrence density surface to: {density_output_path}")
    density_array.rio.to_raster(density_output_path)

    # Determine probability floor based on chosen method
    original_max = float(density_array.max().item())
    floor_probability: float

    if background_method == "contrast":
        if not 0 <= background_value <= 1:
            logging.warning(f"Contrast value {background_value} out of [0,1] range. Clamping.")
            background_value = max(0, min(1, background_value))
        floor_probability = original_max * background_value
        logging.info(f"Using contrast method: floor = max * {background_value:.4f} = {floor_probability:.8f}")
    # ... (other background methods from original code) ...
    elif background_method == "percentile":
        all_values = density_array.data.flatten()
        floor_probability = float(np.percentile(all_values[~np.isnan(all_values)], background_value))
        logging.info(f"Using {background_value}th percentile of ALL valid values: {floor_probability:.8f}")
    elif background_method == "scale":
        floor_probability = original_max * background_value
        logging.info(f"Using {background_value:.4f} * max for background: {floor_probability:.8f}")
    elif background_method == "fixed":
        floor_probability = background_value
        logging.info(f"Using fixed background value: {floor_probability:.8f}")
    elif background_method == "binary":
        floor_probability = 0 # For binary, effectively sample anywhere with non-zero density after floor
        density_array = (density_array > 1e-9).astype(float) # Make binary based on minimal presence
        logging.info("Using binary method (density > 0 indicates presence for sampling pool)")
    else:
        raise ValueError(f"Unknown background method: {background_method}")

    # Apply floor and normalize to create sampling probability surface
    prob_surface = density_array.clip(min=floor_probability)
    # Handle sum == 0 case for normalization
    sum_probs = prob_surface.sum()
    if sum_probs > 1e-9: # Check if sum is effectively non-zero
        prob_surface = prob_surface / sum_probs
    else:
        logging.warning("Sum of probabilities is near zero. Sampling will be uniform over valid cells if any, or fail.")
        # If all probabilities are zero, make it uniform over the valid grid cells (where density_array was not NaN)
        # This is a fallback to prevent division by zero if all values are clipped to zero or near zero.
        valid_cells_mask = ~np.isnan(density_array.data)
        if np.any(valid_cells_mask):
            prob_surface = xr.where(valid_cells_mask, 1.0, 0.0) # Uniform over valid cells
            prob_surface = prob_surface / prob_surface.sum() # Normalize again
        else:
            logging.error("No valid cells to sample from after applying probability floor. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(columns=['geometry'], crs=project_crs)

    # Flatten probability surface and coordinates for sampling
    flat_probs = prob_surface.data.flatten()
    # Remove NaNs from probabilities and corresponding coordinates before sampling
    valid_indices = ~np.isnan(flat_probs)
    flat_probs = flat_probs[valid_indices]
    
    # If all probabilities became NaN or zero after filtering
    if flat_probs.sum() < 1e-9 or len(flat_probs) == 0:
        logging.error("No valid probabilities left to sample from. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(columns=['geometry'], crs=project_crs)
    
    # Normalize again after removing NaNs if any were introduced by clipping/operations on nodata cells
    flat_probs = flat_probs / flat_probs.sum()

    coords_x, coords_y = np.meshgrid(grid.x.data, grid.y.data)
    flat_x = coords_x.flatten()[valid_indices]
    flat_y = coords_y.flatten()[valid_indices]

    # Sample points
    logging.info(f"Sampling {n_background_points} background points...")
    if len(flat_x) < n_background_points:
        logging.warning(f"Number of available unique locations ({len(flat_x)}) is less than requested background points ({n_background_points}). Sampling with replacement or all available points.")
        # Decide if sampling with replacement or just taking all available unique points
        # For now, let's sample with replacement if fewer unique locations than points needed.
        # Or, one could choose to sample min(len(flat_x), n_background_points) without replacement.
        # If sampling all available points without replacement:
        # chosen_indices = np.random.choice(len(flat_x), size=len(flat_x), replace=False, p=flat_probs)
        # If sampling n_background_points with replacement (if n_background_points > len(flat_x)):
        chosen_indices = np.random.choice(len(flat_x), size=n_background_points, replace=True, p=flat_probs)
    else:
        chosen_indices = np.random.choice(len(flat_x), size=n_background_points, replace=False, p=flat_probs)

    background_x = flat_x[chosen_indices]
    background_y = flat_y[chosen_indices]

    # Create GeoDataFrame
    bg_points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(background_x, background_y),
        crs=grid.rio.crs # Use grid CRS which is project_crs
    )
    # Add a column indicating these are background points
    bg_points_gdf["presence"] = 0 

    logging.info(f"Generated {len(bg_points_gdf)} background points.")
    return bg_points_gdf 