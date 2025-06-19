from pathlib import Path
from typing import Union, Tuple, Optional, Literal
import logging
from enum import StrEnum
import json

import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
import rioxarray as rxr  # Required for .rio accessor, even if not directly called

from sdm.utils.io import load_boundary, load_spatial_config
from sdm.raster import generate_model_grid


class TransformMethod(StrEnum):
    LOG = "log"
    SQRT = "sqrt"
    PRESENCE = "presence"
    CAP = "cap"
    RANK = "rank"


def transform_point_counts(
    point_counts: np.ndarray,
    transform_method: TransformMethod,
    cap_percentile: float = 90.0,
) -> np.ndarray:
    """
    Transform point counts based on the transform method.

    Args:
        point_counts: 2D numpy array of point counts.
        transform_method: Method to transform point counts.
        cap_percentile: Percentile to cap point counts at.

    Returns:
        Transformed point counts.
    """
    if transform_method == TransformMethod.LOG:
        logging.info("Applying log transformation to point counts")
        point_counts = np.log1p(point_counts)
    elif transform_method == TransformMethod.SQRT:
        logging.info("Applying square root transformation to point counts")
        point_counts = np.sqrt(point_counts)
    elif transform_method == TransformMethod.PRESENCE:
        logging.info("Converting counts to binary presence/absence")
        point_counts = (point_counts > 0).astype(float)
    elif transform_method == TransformMethod.CAP:
        logging.info("Capping counts at %sth percentile", cap_percentile)
        non_zero = point_counts[point_counts > 0]
        if len(non_zero) > 0:
            cap_value = np.percentile(non_zero, cap_percentile)
            point_counts = np.minimum(point_counts, cap_value)
        else:
            logging.warning("No non-zero counts to cap, skipping cap transformation.")
    elif transform_method == TransformMethod.RANK:
        logging.info("Applying rank-based normalization")
        from scipy.stats import (
            rankdata,
        )  # Local import is fine for less common dependency

        point_counts_flat = point_counts.flatten()
        ranks = rankdata(point_counts_flat, method="average") / len(
            point_counts_flat
        )  # Normalize ranks
        point_counts = ranks.reshape(point_counts.shape)
    else:
        logging.warning(
            f"Unknown transform method: {transform_method}, using raw counts."
        )

    return point_counts


class BackgroundMethod(StrEnum):
    CONTRAST = "contrast"
    PERCENTILE = "percentile"
    SCALE = "scale"
    FIXED = "fixed"
    BINARY = "binary"


def calculate_floor_probability(
    density_array: xr.DataArray,
    background_method: BackgroundMethod,
    background_value: float,
) -> float:
    """
    Calculate the floor probability for background points.

    This is the probability that a background point will be sampled from a location the grid
    without the influence of the occurrence data.

    Args:
        density_array: The density array of the occurrence data.
        background_method: The method to use to calculate the floor probability.
        background_value: The value to use with the background method.

    Returns:
        The floor probability.
    """
    # Determine probability floor based on chosen method
    original_max = float(density_array.max().item())
    floor_probability: float

    if background_method == BackgroundMethod.CONTRAST:
        if not 0 <= background_value <= 1:
            logging.warning(
                f"Contrast value {background_value} out of [0,1] range. Clamping."
            )
            background_value = max(0, min(1, background_value))
        floor_probability = original_max * background_value
        logging.info(
            f"Using contrast method: floor = max * {background_value:.4f} = {floor_probability:.8f}"
        )

    elif background_method == BackgroundMethod.PERCENTILE:
        all_values = density_array.data.flatten()
        floor_probability = float(
            np.percentile(all_values[~np.isnan(all_values)], background_value)
        )
        logging.info(
            f"Using {background_value}th percentile of ALL valid values: {floor_probability:.8f}"
        )
    elif background_method == BackgroundMethod.SCALE:
        floor_probability = original_max * background_value
        logging.info(
            f"Using {background_value:.4f} * max for background: {floor_probability:.8f}"
        )
    elif background_method == BackgroundMethod.FIXED:
        floor_probability = background_value
        logging.info(f"Using fixed background value: {floor_probability:.8f}")
    elif background_method == BackgroundMethod.BINARY:
        floor_probability = 0  # For binary, effectively sample anywhere with non-zero density after floor
        density_array = (density_array > 1e-9).astype(
            float
        )  # Make binary based on minimal presence
        logging.info(
            "Using binary method (density > 0 indicates presence for sampling pool)"
        )
    else:
        raise ValueError(f"Unknown background method: {background_method}")

    return floor_probability


def sample_probability_surface(
    prob_surface: xr.DataArray,
    n_background_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample background points from a probability surface.
    """
    # Flatten probability surface and coordinates for sampling
    flat_probs = prob_surface.data.flatten()
    # Remove NaNs from probabilities and corresponding coordinates before sampling
    valid_indices = ~np.isnan(flat_probs)
    flat_probs = flat_probs[valid_indices]

    # If all probabilities became NaN or zero after filtering
    if flat_probs.sum() < 1e-9 or len(flat_probs) == 0:
        logging.error(
            "No valid probabilities left to sample from. Returning empty GeoDataFrame."
        )
        raise ValueError(
            "No valid probabilities left to sample from. Returning empty GeoDataFrame."
        )

    # Normalize again after removing NaNs if any were introduced by clipping/operations on nodata cells
    flat_probs = flat_probs / flat_probs.sum()

    coords_x, coords_y = np.meshgrid(prob_surface.x.data, prob_surface.y.data)
    flat_x = coords_x.flatten()[valid_indices]
    flat_y = coords_y.flatten()[valid_indices]

    # Sample points
    logging.info(f"Sampling {n_background_points} background points...")
    if len(flat_x) < n_background_points:
        logging.warning(
            f"Number of available unique locations ({len(flat_x)}) is less than requested background points ({n_background_points}). Sampling with replacement or all available points."
        )
        # Decide if sampling with replacement or just taking all available unique points
        # For now, let's sample with replacement if fewer unique locations than points needed.
        # Or, one could choose to sample min(len(flat_x), n_background_points) without replacement.
        # If sampling all available points without replacement:
        # chosen_indices = np.random.choice(len(flat_x), size=len(flat_x), replace=False, p=flat_probs)
        # If sampling n_background_points with replacement (if n_background_points > len(flat_x)):
        chosen_indices = np.random.choice(
            len(flat_x), size=n_background_points, replace=True, p=flat_probs
        )
    else:
        chosen_indices = np.random.choice(
            len(flat_x), size=n_background_points, replace=False, p=flat_probs
        )

    background_x = flat_x[chosen_indices]
    background_y = flat_y[chosen_indices]
    probabilities = flat_probs[chosen_indices]

    return background_x, background_y, probabilities

def normalise_to_distribution(
        x: np.ndarray,
        mean: float = 1.0,
        std: float = 1.0,
        eps: float = 1e-9,
) -> np.ndarray:
    """
    Normalise an array to have a mean of mean and a standard deviation of std.
    """
    if len(x) == 1 or x.std() < eps:
        return np.array([mean])
    else:
        x_norm = (x - x.mean()) / (x.std() + eps)
        x_transformed = (x_norm - mean) / std + mean
        return x_transformed

def weight_density_array_by_regions(
    density_array: xr.DataArray,
    regions: gpd.GeoDataFrame,
    weight: float = 1.0,
    reverse_weights: bool = False,
) -> xr.DataArray:
    """
    Weight the density array by regions.

    This function will weight the density array by the number of occurrences in each region.

    Args:
        density_array: The density array to weight.
        regions: The regions to weight the density array by.
        weight: The weight to apply to the density array. Influences how much the density array is
            weighted by the number of occurrences in each region. 1.0 means no weighting. 0.0 means no weighting.
        reverse_weights: Whether to reverse the weights. If True, regions with higher weights will have lower density.
    Returns:
        The weighted density array.
    """
    # make a copy of the regions so we don't modify the original
    regions = regions.copy()

    if "_region_id" in regions.columns:
        raise ValueError(
            "'_region_id' is a reserved column name for internal use. Please rename the column."
        )
    else:
        regions["_region_id"] = [f"region_{i}" for i in range(len(regions))]

    ones = xr.ones_like(density_array).squeeze()
    ones.rio.write_nodata(0, inplace=True)
    region_masks = []
    region_ids = []
    for idx, row in regions.iterrows():
        # create a mask for the region
        mask = ones.rio.clip([row.geometry], density_array.rio.crs, drop=False)
        region_masks.append(mask)
        region_ids.append(row["_region_id"])

    region_masks = xr.concat(region_masks, dim="region_id")
    # get the background region where all regions are 0
    background_region = region_masks.sum(dim="region_id") == 0
    region_masks = xr.concat([region_masks, background_region], dim="region_id")

    region_masks.coords["region_id"] = region_ids + ["background"]
    region_masks.rio.write_nodata(0, inplace=True)

    # Calculate density sums for each region
    region_weights = (density_array * region_masks)
    region_sums = region_weights.sum(dim=["x", "y"])

    # Normalize the region sums
    region_sums = normalise_to_distribution(region_sums, mean=1.0, std=weight)

    # Reverse the density so that regions with higher weights have lower density
    if reverse_weights:
        region_sums = 1.0 - region_sums + 1

    # Create a weight array by broadcasting the region weights to the full grid
    weight_array = xr.zeros_like(density_array)

    for region_id, weight in zip(region_masks.coords["region_id"], region_sums):
        weight_array = weight_array + (region_masks.sel(region_id=region_id) * weight)

    # Apply the weights to the density array
    density_array = density_array * weight_array

    return density_array


def generate_background_points_from_data(
    occurrence_data: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    regions: Optional[gpd.GeoDataFrame] = None,
    n_background_points: int = 10000,
    background_method: BackgroundMethod = BackgroundMethod.CONTRAST,
    background_value: float = 0.3,
    sigma: float = 1.5,
    grid_resolution: int = 100,  # Renamed from 'resolution' for clarity
    transform_method: TransformMethod = TransformMethod.LOG,
    cap_percentile: float = 90.0,
    region_weighting_factor: float = 1.0,
    reverse_weights: bool = False,
    clip_to_boundary: bool = True,
) -> Tuple[gpd.GeoDataFrame, xr.DataArray]:
    """Generate background points based on density-smoothed occurrence data.

    Args:
        occurrence_data: GeoDataFrame of occurrence data.
        boundary: GeoDataFrame of boundary data.
        regions: GeoDataFrame of regions to weight the density array by.
        n_background_points: Number of background points to generate.
        background_method: Method for setting the minimum background probability.
        background_value: Value to use with background_method.
        sigma: Sigma value for Gaussian smoothing.
        grid_resolution: Resolution of the model grid in CRS units. If None, uses project default.
        transform_method: Method to transform occurrence counts.
        cap_percentile: Percentile for 'cap' transform_method.
        region_weighting_factor: Factor to weight the density array by the number of occurrences in each region.
        reverse_weights: Whether to reverse the weights. If True, regions with higher weights will have lower density.
    Returns:
        GeoDataFrame of background points.
    """

    # Load occurrence data
    logging.info(f"Loading occurrence data from: {occurrence_data}")

    # Load spatial config and boundary to determine target CRS for the grid
    spatial_config = load_spatial_config()
    project_crs = spatial_config["crs"]
    # Load boundary without buffer first, just to get its CRS if needed or for context
    # The grid generation will use its own boundary loading with project_crs
    # occurrences = occurrences.to_crs(project_crs) # Ensure occurrences match project CRS early

    logging.info(
        f"Generating model grid at {grid_resolution or spatial_config.get('resolution', 'default')}m resolution, CRS: {project_crs}"
    )
    grid, grid_bounds = generate_model_grid(
        boundary, project_crs=project_crs, resolution=grid_resolution
    )

    # Ensure occurrences are in the same CRS as the grid for histogramming
    if occurrence_data.crs != grid.rio.crs:
        logging.info(
            f"Re-projecting occurrences from {occurrence_data.crs} to {grid.rio.crs}"
        )
        occurrence_data = occurrence_data.to_crs(grid.rio.crs)

    # Check Geometry of occurrence_data is point
    if not all(occurrence_data.geometry.type == "Point"):
        logging.warning(
            "Occurrence data is not a point geometry. Converting to point geometry."
        )
        raise ValueError(
            "Occurrence data is not a point geometry. Converting to point geometry."
        )

    # Filter occurrences to those within the (potentially larger) grid bounds
    bbox_poly = box(*grid_bounds)
    occurrences_in_grid = occurrence_data[
        occurrence_data.intersects(bbox_poly)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    logging.info(
        f"Using {len(occurrences_in_grid)} occurrence points (within grid bounds) to generate density surface."
    )
    if len(occurrences_in_grid) == 0:
        logging.error(
            "No occurrence points found within the grid bounds. Cannot generate background points."
        )
        # Return an empty GeoDataFrame or raise an error
        raise ValueError(
            "No occurrence points found within the grid bounds. Cannot generate background points."
        )

    # Generate density array based on histogram of occurrences
    # Bins for histogram2d need to be edges. Grid coords are centers. Adjust.
    res_x = grid.rio.resolution()[0]
    res_y = grid.rio.resolution()[1]  # Usually negative

    # Histogram bins should be pixel edges
    hist_bins_x = np.arange(
        grid.x.min() - res_x / 2, grid.x.max() + res_x / 2 + abs(res_x) / 2, res_x
    )
    # For y, which is usually descending, adjust accordingly for edges
    hist_bins_y = np.arange(
        grid.y.min() - abs(res_y) / 2,
        grid.y.max() + abs(res_y) / 2 + abs(res_y) / 2,
        abs(res_y),
    )
    # Ensure y bins are in ascending order for np.histogram2d if y coords were descending
    if grid.y.values[0] > grid.y.values[-1]:  # y is descending
        hist_bins_y = np.sort(hist_bins_y)  # ascending for histogram2d

    point_counts, _, _ = np.histogram2d(
        occurrences_in_grid.geometry.x.values,
        occurrences_in_grid.geometry.y.values,
        bins=(hist_bins_x, hist_bins_y),
    )
    # Always transpose to (y, x) for raster alignment
    point_counts = point_counts.T

    # Only flip if y-coordinates are descending (i.e., north is at the bottom)
    if grid.y.values[0] > grid.y.values[-1]:
        point_counts = np.flipud(point_counts)

    # Transform counts
    point_counts = transform_point_counts(
        point_counts, transform_method, cap_percentile
    )

    # Apply Gaussian smoothing
    logging.info(f"Applying Gaussian smoothing with sigma={sigma}")
    smoothed_counts = gaussian_filter(point_counts, sigma=sigma)

    # Convert to xarray DataArray, aligning with the model grid
    density_array = xr.DataArray(
        smoothed_counts,
        coords=grid.coords,  # Use the model grid's coordinates
        dims=grid.dims,
        name="occurrence_density",
    )
    density_array = density_array.rio.write_crs(grid.rio.crs)
    density_array.rio.write_transform(grid.rio.transform(), inplace=True)
    density_array.rio.write_nodata(
        np.nan, inplace=True
    )  # Ensure nodata for consistency

    floor_probability = calculate_floor_probability(
        density_array, background_method, background_value
    )

    # Apply floor and normalize to create sampling probability surface
    prob_surface = density_array.clip(min=floor_probability)


    if regions is not None:
        density_array = weight_density_array_by_regions(
            density_array, regions, weight=region_weighting_factor, reverse_weights=reverse_weights
        )
    
    # Handle sum == 0 case for normalization
    sum_probs = prob_surface.sum()
    if sum_probs > 1e-9:  # Check if sum is effectively non-zero
        prob_surface = prob_surface / sum_probs
    else:
        logging.warning(
            "Sum of probabilities is near zero. Sampling will be uniform over valid cells if any, or fail."
        )
        # If all probabilities are zero, make it uniform over the valid grid cells (where density_array was not NaN)
        # This is a fallback to prevent division by zero if all values are clipped to zero or near zero.
        valid_cells_mask = ~np.isnan(density_array.data)
        if np.any(valid_cells_mask):
            prob_surface = xr.where(
                valid_cells_mask, 1.0, 0.0
            )  # Uniform over valid cells
            prob_surface = prob_surface / prob_surface.sum()  # Normalize again
        else:
            logging.error(
                "No valid cells to sample from after applying probability floor. Returning empty GeoDataFrame."
            )
            raise ValueError(
                "No valid cells to sample from after applying probability floor. Returning empty GeoDataFrame."
            )

    background_x, background_y, probabilities = sample_probability_surface(
        prob_surface, n_background_points
    )

    # Create GeoDataFrame
    bg_points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(background_x, background_y),
        crs=grid.rio.crs,  # Use grid CRS which is project_crs
    )
    # Add a column indicating these are background points
    bg_points_gdf["presence"] = 0
    bg_points_gdf["density"] = probabilities

    logging.info(f"Generated {len(bg_points_gdf)} background points.")

    # clip the background points to the boundary
    if clip_to_boundary:
        bg_points_gdf = bg_points_gdf[bg_points_gdf.intersects(boundary.union_all())]


    return bg_points_gdf, density_array


def generate_background_points(
    occurrence_data_path: Union[str, Path],
    boundary_path: Union[str, Path],
    output_dir_for_density_raster: Path,  # For saving intermediate density raster
    regions_path: Optional[Union[str, Path]] = None,
    n_background_points: int = 10000,
    background_method: BackgroundMethod = BackgroundMethod.CONTRAST,
    background_value: float = 0.3,
    sigma: float = 1.5,
    grid_resolution: int = 100,
    transform_method: TransformMethod = TransformMethod.LOG,
    cap_percentile: float = 90.0,
    region_weighting_factor: float = 1.0,
    write_parameters: bool = False,
    reverse_weights: bool = False,
) -> Tuple[Path, Path]:
    """
    A wrapper around generate_background_points_from_data that has a file interface and output

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON or Parquet).
        boundary_path: Path to boundary data.
        output_dir_for_density_raster: Directory to save intermediate density raster.
        regions_path: Path to regions data.
        n_background_points: Number of background points to generate.
        background_method: Method for setting the minimum background probability.
        background_value: Value to use with background_method.
        sigma: Sigma value for Gaussian smoothing.
        grid_resolution: Resolution of the model grid in CRS units. If None, uses project default.
        transform_method: Method to transform occurrence counts.
        cap_percentile: Percentile for 'cap' transform_method.
        region_weighting_factor: Factor to weight the density array by the number of occurrences in each region.
        write_parameters: Whether to write the parameters to a JSON file.
        reverse_weights: Whether to reverse the weights. If True, regions with higher weights will have lower density.

    Returns:
        Tuple[Path, Path]: Path to background points GeoJSON file and path to density raster.
        The background points are saved as a GeoJSON file and the density raster is saved as a TIFF file.
    """
    occurrence_data = gpd.read_file(occurrence_data_path)
    boundary = gpd.read_file(boundary_path)
    if regions_path is not None:
        regions = gpd.read_file(regions_path)
    else:
        regions = None

    background_points, density_raster = generate_background_points_from_data(
        occurrence_data=occurrence_data,
        boundary=boundary,
        regions=regions,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        grid_resolution=grid_resolution,
        transform_method=transform_method,
        cap_percentile=cap_percentile,
        region_weighting_factor=region_weighting_factor,
        reverse_weights=reverse_weights,
    )

    background_points_path = output_dir_for_density_raster / "background_points.geojson"
    background_points.to_file(background_points_path, driver="GeoJSON")

    density_raster_path = (
        output_dir_for_density_raster / "occurrence_density_surface.tif"
    )
    density_raster.rio.to_raster(density_raster_path)

    if write_parameters:
        parameters_path = (
            output_dir_for_density_raster / "background-pointparameters.json"
        )
        with open(parameters_path, "w") as f:
            json.dump(
                {
                    "background_method": background_method,
                    "background_value": background_value,
                    "sigma": sigma,
                    "grid_resolution": grid_resolution,
                    "transform_method": transform_method,
                    "cap_percentile": cap_percentile,
                    "n_background_points": n_background_points,
                },
                f,
            )

    return background_points_path, density_raster_path
