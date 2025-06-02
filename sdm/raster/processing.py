from pathlib import Path
import xarray as xr
import rioxarray as rxr
import logging
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Optional, Union, List
from tempfile import NamedTemporaryFile
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .utils import generate_point_grid, rasterise_gdf


def load_dataset(path: Path):
    data = rxr.open_rasterio(path)
    # Extract the band name
    long_name = data.attrs["long_name"]
    if type(long_name) == str:
        long_name = [long_name]
    else:
        long_name = list(long_name)

    # prefix the band name with the file name
    long_name = [f"{path.stem}_{name}" for name in long_name]

    # Rename the band dimension and convert to a dataset
    data.coords["band"] = long_name
    return data.to_dataset(dim="band")


def load_evs(ev_folder: Path):
    # list the tifs
    ev_tifs = list(ev_folder.glob("*.tif"))

    evs = [load_dataset(path) for path in ev_tifs]

    evs = xr.merge(evs)

    return evs


def interpolate_nas(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dataset.sortby("y")
    dataset = dataset.interpolate_na(dim="y")
    dataset = dataset.interpolate_na(dim="x")

    return dataset


def calculate_multiscale_variables(dataset: xr.Dataset, window: int) -> xr.Dataset:
    original_names = dataset.data_vars

    vars = (
        dataset.rolling(x=window, y=window, center=True)
        .mean(skipna=True)
        .rename(
            {name: f"{name}_{round((window/2) * 100)}m" for name in dataset.data_vars}
        )
    )
    # Assign nodata from the original dataset
    for new_name, old_name in zip(vars.data_vars, original_names):
        vars[new_name].rio.write_nodata(
            dataset[old_name].rio.nodata, inplace=True
        )


    return vars


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
        # Ensure geometry is not empty and contains points
        if gdf.empty or not gdf.geometry.apply(lambda geom: geom.type == 'Point').all():
            logging.warning("Skipping %s due to empty or non-point geometry", name)
            grid[f"distance_to_{name}"] = np.nan
            continue
        
        feature_points = np.array(
            [[geom.x, geom.y] for geom in gdf.geometry]
        )
        if feature_points.size == 0:
            logging.warning("Skipping %s due to no valid feature points after extraction", name)
            grid[f"distance_to_{name}"] = np.nan
            continue

        tree = cKDTree(feature_points)
        grid[f"distance_to_{name}"] = tree.query(grid_points, k=1)[0]
        logging.debug("Completed distance calculation for %s", name)

    logging.info("Converting distance grid to xarray")
    # Select only original columns + new distance columns for to_xarray()
    # to avoid issues if 'geometry' column still exists with mixed types.
    columns_to_convert = [col for col in grid.columns if col.startswith('distance_to_') or col in ['x', 'y']]
    
    if not all(col in grid.columns for col in ['x', 'y']):
        logging.error("x or y columns missing in grid before to_xarray()")
        # Handle error appropriately, maybe return empty Dataset or raise
        return xr.Dataset() # Or raise an error
        
    distance_array = (
        grid[columns_to_convert]
        .sort_values(["y", "x"])
        .set_index(["y", "x"])
        .to_xarray()
        .rio.write_crs(boundary.crs)
    )
    # Drop 'geometry' if it was accidentally carried over and is now a coord
    if 'geometry' in distance_array.coords:
        distance_array = distance_array.drop_vars(['geometry'])

    # log the na values
    logging.debug("NA values for distance array: %.2f%%",
                  100 * distance_array.isnull().any().mean() if distance_array.data_vars else 0)
    logging.info("Distance matrix calculation complete")

    return distance_array


def calculate_feature_cover(
    feature_gdfs: Dict[str, gpd.GeoDataFrame],
    boundary: gpd.GeoDataFrame,
    target_resolution: int = 100,
    base_resolution: int = 10
) -> xr.Dataset:
    """Calculate feature coverage density for each feature type.

    Args:
        feature_gdfs: Dictionary of feature name to GeoDataFrame mappings.
        boundary: GeoDataFrame containing the area of interest.
        target_resolution: Resolution in meters for the output raster.
        base_resolution: Resolution for initial rasterization before aggregation.

    Returns:
        xarray Dataset containing coverage density for each feature type.
    """
    logging.info("Calculating feature cover at %dm resolution, from %dm base",
                 target_resolution, base_resolution)

    if target_resolution % base_resolution != 0:
        raise ValueError("target_resolution must be a multiple of base_resolution.")

    boundary_union: Polygon = boundary.unary_union
    scale_factor = target_resolution // base_resolution
    
    # Ensure the boundary GeoDataFrame has a CRS
    if boundary.crs is None:
        raise ValueError("Boundary GeoDataFrame must have a CRS defined.")

    def calculate_cover_for_gdf(gdf: gpd.GeoDataFrame, name: str) -> xr.Dataset:
        logging.debug("Processing cover for %s (%d features)", name, len(gdf))
        if gdf.empty:
            logging.warning(f"GeoDataFrame for {name} is empty. Skipping cover calculation.")
            # Create an empty DataArray or Dataset that can be merged
            # This needs to match the expected structure from non-empty cases after reprojection
            # For now, returning an empty Dataset. This might need refinement based on how merge handles it.
            return xr.Dataset()

        # Ensure input GDF has the same CRS as the boundary for consistent rasterization
        if gdf.crs != boundary.crs:
            logging.info(f"Reprojecting {name} from {gdf.crs} to {boundary.crs}")
            gdf = gdf.to_crs(boundary.crs)

        with NamedTemporaryFile(suffix=".tif") as f:
            # Use the rasterise_gdf from species_sdm.raster.utils
            rasterise_gdf(
                gdf, 
                resolution=base_resolution, 
                output_file=f.name, 
                bbox=boundary_union.bounds
            )
            # Open with rioxarray, specify chunks for potentially large intermediate rasters
            cover_raster: xr.DataArray = rxr.open_rasterio(f.name, chunks='auto').squeeze(drop=True)
            
            # Ensure nodata is handled if necessary, though rasterise_gdf sets a nodata value
            # If rasterise_gdf output has explicit nodata, rioxarray should handle it.
            # cover_raster = cover_raster.where(cover_raster != cover_raster.rio.nodata) # Example if needed

            if scale_factor > 1:
                cover_area = cover_raster.coarsen(
                    x=scale_factor, y=scale_factor, boundary="trim"
                ).sum()
            else:
                cover_area = cover_raster # No coarsening if scale_factor is 1

            logging.debug("NA values for %s cover: %.2f%%", name,
                          100 * float(cover_area.isnull().mean()) if cover_area.size > 0 else 0)
        # Return as a Dataset, a_dataset.to_dataset(name=name) if it's a DataArray
        return cover_area.rename(name).to_dataset() 

    cover_datasets = []
    for name, gdf_item in feature_gdfs.items():
        logging.info("Calculating cover for %s", name)
        # Make a copy to avoid modifying original GDFs if reprojected
        # Ensure it's a GeoDataFrame
        current_gdf = gpd.GeoDataFrame(gdf_item.copy())
        processed_cover = calculate_cover_for_gdf(current_gdf, name)
        if processed_cover.data_vars: # Only append if it's not an empty dataset
            cover_datasets.append(processed_cover)

    if not cover_datasets:
        logging.warning("No feature cover datasets were generated.")
        return xr.Dataset() # Return empty dataset if nothing was processed

    feature_cover = xr.merge(cover_datasets)
    logging.info("Feature cover calculation complete")
    
    # Clip to the exact boundary geometry after merging
    # Ensure feature_cover has spatial dims and CRS set for clipping
    if feature_cover.rio.crs is None and boundary.crs is not None:
         feature_cover = feature_cover.rio.write_crs(boundary.crs)
    if not feature_cover.rio.dims: # Check if spatial dims are set
        # Attempt to infer common spatial dims if possible, or raise error
        # This is a fallback, ideally rasters should have spatial dims set earlier
        if 'x' in feature_cover.coords and 'y' in feature_cover.coords:
            feature_cover = feature_cover.rio.set_spatial_dims(x_dim='x', y_dim='y')
        else:
            logging.error("Cannot clip feature_cover: spatial dimensions not set or inferrable.")
            return feature_cover # Return unclipped or raise

    # Ensure boundary geometry is valid for clipping
    if not boundary_union.is_valid:
        logging.warning("Boundary union geometry is not valid, attempting to buffer by 0 to fix.")
        boundary_union = boundary_union.buffer(0)
        if not boundary_union.is_valid:
            logging.error("Boundary union geometry is still not valid after buffer(0). Clipping might fail or be incorrect.")

    # Perform clipping if feature_cover has data variables
    if feature_cover.data_vars:
        try:
            feature_cover_clipped = feature_cover.rio.clip([boundary_union], crs=boundary.crs, all_touched=True)
            return feature_cover_clipped
        except Exception as e:
            logging.error(f"Failed to clip feature_cover: {e}. Returning unclipped data.")
            return feature_cover
    else:
        return feature_cover # Return as is if no data variables (e.g., all inputs were empty)


def calculate_distance_to_geom(
    geom: BaseGeometry,
    boundary_gdf: gpd.GeoDataFrame,
    grid_bounds: tuple,
    resolution: float,
    var_name: str = "distance",
) -> xr.Dataset:
    """
    Create a grid of points within the boundary and calculate the distance to the geometry.
    Assumes boundary_gdf and grid_bounds are in the same CRS.
    """
    if not isinstance(geom, BaseGeometry):
        raise ValueError("Input geom must be a Shapely BaseGeometry (e.g., Polygon, MultiPolygon, LineString)")
    
    points_gdf = generate_point_grid(
        bbox=grid_bounds, resolution=resolution, crs=boundary_gdf.crs
    )
    # points_gdf.reset_index(drop=True, inplace=True) # reset_index might not be needed if generate_point_grid returns a clean index

    # Calculate the distance to the geometry
    # Ensure geom is in the same CRS as points_gdf if it has a CRS attribute and is a GeoSeries/GeoDataFrame
    # For a raw shapely geom, assume it is.
    distances = points_gdf.geometry.distance(geom)
    # distances.reset_index(drop=True, inplace=True)
    points_gdf[var_name] = distances
    logging.debug("Missing values for %s: %.2f%%", var_name, round(points_gdf[var_name].isna().mean() * 100, 2) if points_gdf[var_name].size > 0 else 0)

    # Reshape the distances to a grid
    logging.info("Converting %s grid to xarray", var_name)
    distance_ds = (
        points_gdf[[var_name, 'x', 'y']] # Select only necessary columns
        .sort_values(["y", "x"])
        .set_index(["y", "x"])
        .to_xarray()
    )
    # CRS should be set based on the points_gdf.crs
    if boundary_gdf.crs:
        distance_ds = distance_ds.rio.write_crs(boundary_gdf.crs)
    
    # The original function returned a dataset and also dropped 'geometry'.
    # to_xarray() on selected columns won't have 'geometry' unless 'x','y' were geom.
    # If var_name is the only data variable, it's fine.
    return distance_ds


def summarise_raster_metrics(
    data_array: xr.DataArray, 
    target_resolution: int, 
    var_name_prefix: str = "summary",
    boundary_handling: str = "pad" # or "trim"
) -> xr.Dataset:
    """
    Summarise a DataArray by calculating mean, min, max, and standard deviation 
    after coarsening to a target resolution.

    Args:
        data_array (xr.DataArray): The input data array (single variable).
        target_resolution (int): The target resolution for coarsening.
        var_name_prefix (str): Prefix for the output variable names in the Dataset.
        boundary_handling (str): How to handle boundaries when coarsening ('pad', 'trim').
    """
    if not isinstance(data_array, xr.DataArray):
        raise TypeError(f"Input must be an xarray.DataArray, got {type(data_array)}")

    current_res_x, current_res_y = data_array.rio.resolution()
    current_res = int(abs(current_res_x)) # Assume square pixels and take absolute
    
    if current_res == 0:
        raise ValueError("Could not determine current resolution from data_array, or it is zero.")

    # Ensure target_resolution is coarser than current_resolution
    if target_resolution < current_res:
        logging.warning(
            f"Target resolution {target_resolution}m is finer than or equal to current raster resolution {current_res}m. "
            f"Skipping coarsening, calculating stats on original resolution."
        )
        # Calculate stats on original data if target is not coarser
        data_mean = data_array.mean()
        data_min = data_array.min()
        data_max = data_array.max()
        data_std = data_array.std()
        scale_factor = 1
    elif target_resolution == current_res:
        logging.info(
            f"Target resolution {target_resolution}m is the same as current raster resolution {current_res}m. "
            f"Calculating stats on original resolution without coarsening."
        )
        data_mean = data_array.mean()
        data_min = data_array.min()
        data_max = data_array.max()
        data_std = data_array.std()
        scale_factor = 1 # Conceptually
    else:
        scale_factor = target_resolution // current_res # Integer division
        if scale_factor <= 1: # Should not happen if target_resolution > current_res
            logging.warning(
                f"Calculated scale factor {scale_factor} is <= 1 (target: {target_resolution}, current: {current_res}). "
                f"Calculating stats on original resolution."
            )
            data_mean = data_array.mean()
            data_min = data_array.min()
            data_max = data_array.max()
            data_std = data_array.std()
        else:
            logging.info(f"Coarsening from {current_res}m to {target_resolution}m (factor: {scale_factor})...")
            data_coarse = data_array.coarsen(x=scale_factor, y=scale_factor, boundary=boundary_handling, coord_func="mean")
            data_mean = data_coarse.mean()
            data_min = data_coarse.min()
            data_max = data_coarse.max()
            data_std = data_coarse.std()

    summary_ds = xr.Dataset(
        {
            f"{var_name_prefix}_mean_{target_resolution}m": data_mean,
            f"{var_name_prefix}_min_{target_resolution}m": data_min,
            f"{var_name_prefix}_max_{target_resolution}m": data_max,
            f"{var_name_prefix}_std_{target_resolution}m": data_std,
        }
    )
    # Attempt to copy spatial attributes if they exist and are meaningful after aggregation
    if hasattr(data_array, 'rio'):
        summary_ds = summary_ds.rio.write_crs(data_array.rio.crs)
        # Transform might need adjustment based on coarsening, or use original if extent is similar
        # For simplicity, assign the transform of the (potentially) coarsened mean array
        summary_ds = summary_ds.rio.write_transform(data_mean.rio.transform() if hasattr(data_mean, 'rio') else None)

    return summary_ds


def create_binary_raster_from_category(
    source_raster: xr.DataArray, 
    category_value: Union[int, float],
    output_var_name: str,
    nodata_val: Optional[Union[int, float]] = np.nan # Value to assign where source is nodata
) -> xr.Dataset:
    """
    Creates a binary raster (0 or 1) indicating the presence of a specific category value 
    in the source raster. Preserves nodata values from the source.

    Args:
        source_raster (xr.DataArray): Input raster with category values.
        category_value (Union[int, float]): The specific category value to map to 1.
        output_var_name (str): Name for the output variable in the returned Dataset.
        nodata_val: Value to use for nodata in the output raster where source has nodata.
                   Defaults to np.nan. If source is integer and np.nan is used, output becomes float.
                   
    Returns:
        xr.Dataset: A Dataset containing the binary raster layer.
    """
    if not isinstance(source_raster, xr.DataArray):
        raise TypeError("source_raster must be an xarray.DataArray.")

    # Initialize output as all zeros, with the type of nodata_val or float if np.nan
    if np.isnan(nodata_val):
        binary_array = xr.zeros_like(source_raster, dtype=np.float64)
    else: # Try to match nodata_val type or source_raster type if nodata_val is not np.nan
        binary_array = xr.zeros_like(source_raster, dtype=type(nodata_val) if nodata_val is not None else source_raster.dtype)

    # Where source_raster equals category_value, set binary_array to 1
    binary_array = binary_array.where(source_raster != category_value, 1)

    # Handle nodata: where source_raster has its original nodata, set binary_array to nodata_val
    # This assumes source_raster.rio.nodata is correctly set or np.isnan can identify its nodata.
    if source_raster.rio.nodata is not None:
        binary_array = binary_array.where(source_raster != source_raster.rio.nodata, nodata_val)
    elif np.issubdtype(source_raster.dtype, np.floating):
        binary_array = binary_array.where(~np.isnan(source_raster), nodata_val)
    # If integer type with no explicit nodata, it's harder to guess. Assume all values are valid.

    binary_ds = binary_array.to_dataset(name=output_var_name)
    # Copy spatial attributes
    if hasattr(source_raster, 'rio'):
        binary_ds = binary_ds.rio.write_crs(source_raster.rio.crs)
        binary_ds = binary_ds.rio.write_transform(source_raster.rio.transform())
        if nodata_val is not None:
             binary_ds[output_var_name].rio.write_nodata(nodata_val, inplace=True)

    return binary_ds


def aggregate_categorical_rasters(
    categorical_raster_ds: xr.Dataset, 
    aggregation_map: Dict[str, List[str]],
    categories_to_drop: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Aggregates variables in a Dataset (representing categorical rasters) based on a mapping.
    Optionally drops specified original or newly aggregated categories.

    Args:
        categorical_raster_ds (xr.Dataset): Input Dataset where each data variable is a raster layer 
                                            (e.g., binary indicator for a land cover type).
        aggregation_map (Dict[str, List[str]]): Dictionary mapping new aggregated category names 
                                                 to a list of existing variable names in the Dataset.
        categories_to_drop (Optional[List[str]]): List of variable names (original or newly aggregated)
                                                  to drop from the final Dataset.
    Returns:
        xr.Dataset: Dataset with aggregated and optionally pruned raster layers.
    """
    if not isinstance(categorical_raster_ds, xr.Dataset):
        raise TypeError("categorical_raster_ds must be an xarray.Dataset.")

    processed_ds = categorical_raster_ds.copy()

    for new_cat_name, existing_vars_to_sum in aggregation_map.items():
        # Filter for variables that actually exist in the current dataset state
        vars_present = [var for var in existing_vars_to_sum if var in processed_ds.data_vars]
        if not vars_present:
            logging.warning(f"For aggregation '{new_cat_name}', no source variables found: {existing_vars_to_sum}. Skipping.")
            continue
        
        logging.info(f"Aggregating '{new_cat_name}' from: {vars_present}")
        # Sum the DataArrays for the present variables to create the new aggregated category
        # Ensure they are aligned if they weren't already (should be if from same source grid)
        # .to_array().sum() is a robust way if dimensions are consistent.
        processed_ds[new_cat_name] = processed_ds[vars_present].to_array(dim="category_source").sum(dim="category_source", skipna=True)
        
        # Drop the original variables that were summed into the new category
        processed_ds = processed_ds.drop_vars(vars_present)

    if categories_to_drop:
        vars_to_actually_drop = [var for var in categories_to_drop if var in processed_ds.data_vars]
        if vars_to_actually_drop:
            logging.info(f"Dropping specified categories: {vars_to_actually_drop}")
            processed_ds = processed_ds.drop_vars(vars_to_actually_drop)
            
    return processed_ds
