# Placeholder for occurrence data processing functions

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
import numpy as np # For filter_gdf_to_grid

# Example function (to be replaced with actual logic from notebook)
def clean_occurrence_data(raw_occurrence_path: Path, output_path: Path) -> None:
    logging.info(f"Processing raw occurrence data from {raw_occurrence_path}")
    # df = pd.read_csv(raw_occurrence_path) # Or other format
    # ... data cleaning and filtering ...
    # gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
    # gdf.to_parquet(output_path)
    logging.info(f"Cleaned occurrence data saved to {output_path}")
    pass

def filter_bats_data(
    gdf: gpd.GeoDataFrame, 
    genus: str = None, 
    latin_name: str = None, 
    activity_type: str = None
) -> gpd.GeoDataFrame:
    """Filters a GeoDataFrame of bat occurrence data based on taxonomic or activity criteria."""
    # It's good practice to work on a copy if the original GDF might be used elsewhere
    # However, the original function didn't, so for direct porting, we also don't initially.
    # Consider adding .copy() if modifications in place are not desired by caller.
    # gdf = gdf.copy() 
    
    if genus:
        if "genus" not in gdf.columns:
            logging.warning("'genus' column not found in GeoDataFrame. Cannot filter by genus.")
        else:
            gdf = gdf[gdf.genus == genus]
    if latin_name:
        if "latin_name" not in gdf.columns:
            logging.warning("'latin_name' column not found in GeoDataFrame. Cannot filter by latin_name.")
        else:
            gdf = gdf[gdf.latin_name == latin_name]
    if activity_type:
        if "activity_type" not in gdf.columns:
            logging.warning("'activity_type' column not found in GeoDataFrame. Cannot filter by activity_type.")
        else:
            gdf = gdf[gdf.activity_type == activity_type]
    return gdf


def filter_points_to_grid_cells(
    points_gdf: gpd.GeoDataFrame, 
    grid_gdf: gpd.GeoDataFrame, 
    grid_id_col: str = "grid_id", # Column in grid_gdf to use as unique cell ID
    tolerance: float = 50.0 # Max distance for sjoin_nearest
) -> gpd.GeoDataFrame:
    """
    Filters a GeoDataFrame of points to keep only one point per cell of a reference grid.
    It performs a spatial join to associate points with grid cells and then 
    deduplicates to keep one point per grid cell (based on grid_id_col from grid_gdf).

    Args:
        points_gdf (gpd.GeoDataFrame): Points to filter.
        grid_gdf (gpd.GeoDataFrame): Reference grid cells. Must have a unique ID column `grid_id_col`.
        grid_id_col (str): Name of the unique ID column in `grid_gdf` (e.g., 'grid_index', 'cell_id').
                           This column's values will be transferred to points_gdf as `grid_id_col`.
        tolerance (float): Maximum distance for the sjoin_nearest operation. 
                           Points further than this from any grid cell centroid/geometry might be dropped 
                           depending on sjoin_nearest behavior if no match is found within tolerance.

    Returns:
        gpd.GeoDataFrame: Filtered points_gdf with an added 'grid_id_col' (renamed from grid_gdf's index_right 
                          or `grid_id_col` if that was already present) and duplicates per grid cell removed.
                          Points not joining to any grid cell within tolerance might be dropped.
    """
    if grid_id_col not in grid_gdf.columns:
        # If grid_id_col is not a real column, assume user wants to use grid_gdf.index
        # and name the resulting column grid_id_col.
        # sjoin_nearest will use grid_gdf.index as index_right by default.
        logging.info(f"'{grid_id_col}' not in grid_gdf columns. Using grid_gdf.index for unique cell IDs.")
        grid_to_join = grid_gdf.copy() # Avoid modifying original
        # No need to create a column from index if sjoin_nearest uses it directly as index_right
    else:
        # If grid_id_col *is* a column, ensure it's unique for proper deduplication later.
        if not grid_gdf[grid_id_col].is_unique:
            logging.warning(
                f"Values in '{grid_id_col}' of grid_gdf are not unique. Deduplication might be unpredictable."
            )
        grid_to_join = grid_gdf[[grid_id_col, grid_gdf.geometry.name]] # Keep only necessary columns for join
    
    # Perform spatial join. `index_right` will refer to the index of `grid_to_join`.
    # If `grid_id_col` was an actual column in `grid_to_join`, its values are also brought over.
    points_on_grid = gpd.sjoin_nearest(
        points_gdf, 
        grid_to_join, 
        how="left", 
        distance_col="_distance_to_grid_cell",
        max_distance=tolerance 
    )

    # Handle points that didn't join (outside tolerance or no grid cells)
    # These will have NaN for columns from grid_to_join, including index_right.
    # We typically want to drop these points.
    original_len = len(points_on_grid)
    points_on_grid.dropna(subset=["index_right"], inplace=True) # index_right is key from sjoin
    if len(points_on_grid) < original_len:
        logging.info(f"Dropped {original_len - len(points_on_grid)} points that did not join to any grid cell within tolerance {tolerance}.")

    # `index_right` is the index from `grid_gdf` (or `grid_to_join`). This is our unique grid cell identifier.
    # Rename `index_right` to the desired `grid_id_col` name for clarity in the output points_gdf.
    # This column will be used for deduplication.
    if "index_right" in points_on_grid.columns: # Should always be true after sjoin
        points_on_grid.rename(columns={"index_right": grid_id_col}, inplace=True)
    else:
        # This case should ideally not happen if sjoin was successful and created index_right
        logging.error("'index_right' column not found after sjoin_nearest. Cannot deduplicate by grid cell.")
        return points_on_grid # Or raise error

    # Drop duplicate points within the same grid cell, keeping the first one encountered.
    # The choice of which point to keep (first, last, etc.) can be important.
    # Original code used drop_duplicates(subset="index_right", inplace=True) which implies keeping first.
    original_len_before_dedup = len(points_on_grid)
    points_on_grid.drop_duplicates(subset=[grid_id_col], keep='first', inplace=True)
    if len(points_on_grid) < original_len_before_dedup:
        logging.info(f"Deduplicated {original_len_before_dedup - len(points_on_grid)} points to keep one per grid cell.")

    # Clean up the distance column if it was added and is no longer needed
    if "_distance_to_grid_cell" in points_on_grid.columns:
        points_on_grid.drop(columns=["_distance_to_grid_cell"], inplace=True)
        
    return points_on_grid 