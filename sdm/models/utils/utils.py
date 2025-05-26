import logging
from typing import List, Tuple # Corrected from list, tuple

import geopandas as gpd
import pandas as pd
import elapid as ela # For stack_geodataframes, distance_weights

# Assuming filter_gdf_to_grid will be in species_sdm.occurrence.processing
# from species_sdm.occurrence.processing import filter_gdf_to_grid 
# For now, to avoid circular dependency if this file is created first, 
# we might need to pass grid_gdf to filter_gdf_to_grid if it stays external,
# or include a simplified version here if it's tightly coupled.
# Let's assume it will be imported for now.

logger = logging.getLogger(__name__)

def calculate_background_points(
    n_presences: int, 
    min_bg: int = 1000, 
    max_bg: int = 10000, 
    factor: int = 10
) -> int:
    """
    Calculate a recommended number of background (pseudo-absence) points
    based on the number of presence records.
    
    Args:
        n_presences (int): Number of presence records.
        min_bg (int): Minimum number of background points.
        max_bg (int): Maximum number of background points.
        factor (int): Multiplier factor for presences.
        
    Returns:
        int: Calculated number of background points.
    """
    n_bg = max(min(n_presences * factor, max_bg), min_bg)
    logger.info(f"Recommended background points: {n_bg} (presences: {n_presences})")
    return int(n_bg)

def prepare_occurrence_data(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    background_density: pd.Series, # Assumes density is pre-calculated and passed
    grid_gdf: gpd.GeoDataFrame, # Used for filtering by grid_index
    input_vars: List[str],
    drop_na: bool = True,
    sample_weight_n_neighbors: int = 5,
    filter_to_grid: bool = True, # Whether to filter points to the grid and remove overlaps
    subset_background: bool = True, # Whether to subset background points
    order_by_density_for_subset: bool = True, # For background subsetting
) -> gpd.GeoDataFrame:
    """
    Prepares presence and background GeoDataFrames for model training.
    Includes filtering to grid, removing overlaps, subsetting background points, 
    and calculating sample weights.

    Args:
        presence_gdf: GeoDataFrame of presence points with EV data.
        background_gdf: GeoDataFrame of background points with EV data.
        background_density: Series with density values for background points, indexed like background_gdf.
        grid_gdf: GeoDataFrame representing the modelling grid cells, must have 'grid_index' or similar unique ID per cell.
                  (This function currently expects `filter_gdf_to_grid` to add 'grid_index'.)
        input_vars: List of column names for environmental variables.
        drop_na: If True, drop rows with NA values in input_vars.
        sample_weight_n_neighbors: Number of neighbors for distance-based sample weighting.
        filter_to_grid: If True, filter points not falling into grid cells and remove background points in presence cells.
        subset_background: If True, subset background points based on n_presences.
        order_by_density_for_subset: If True and subset_background is True, order background points by density for subsetting.

    Returns:
        A GeoDataFrame ready for model training (stacked, with class labels and sample weights).
    """
    # Make copies to avoid modifying original DataFrames
    presence_gdf = presence_gdf.copy()
    background_gdf = background_gdf.copy()
    background_density = background_density.copy()

    if not all(col in presence_gdf.columns for col in input_vars):
        raise ValueError("Not all input_vars found in presence_gdf columns.")
    if not all(col in background_gdf.columns for col in input_vars):
        raise ValueError("Not all input_vars found in background_gdf columns.")

    if filter_to_grid:
        logger.info("Filtering points to grid and removing overlaps...")
        # This part relies on filter_gdf_to_grid adding an 'index_right' or 'grid_index' column
        # and that grid_gdf has a unique index to be used as 'grid_index'.
        # Let's assume filter_gdf_to_grid is available and works as expected.
        # from species_sdm.occurrence.processing import filter_gdf_to_grid # Ideal import location
        
        # Placeholder for filter_gdf_to_grid if not yet available - this is complex logic.
        # For now, we assume this step correctly assigns a 'grid_id' based on grid_gdf
        # and filters points outside any grid cell.
        # Simplified conceptual filtering (actual sjoin and duplicate drop is more involved):
        # presence_gdf = filter_gdf_to_grid(presence_gdf, grid_gdf, tolerance=...)
        # background_gdf = filter_gdf_to_grid(background_gdf, grid_gdf, tolerance=...)
        
        # This step implies `filter_gdf_to_grid` adds a column like `grid_index` (from `grid_gdf.index` or a specific ID col)
        # For demonstration, let's assume a simplified direct spatial join if filter_gdf_to_grid is not yet moved/defined.
        # This is NOT a robust replacement for the original filter_gdf_to_grid logic.
        if 'grid_index' not in presence_gdf.columns or 'grid_index' not in background_gdf.columns:
             logger.warning("'grid_index' not found in GDFs. Skipping overlap removal. filter_gdf_to_grid logic needs to be integrated.")
        else:
            # Drop background points that share a grid_index with presence points
            original_bg_len = len(background_gdf)
            background_gdf = background_gdf[
                ~background_gdf["grid_index"].isin(presence_gdf["grid_index"])
            ]
            logger.info(f"Removed {original_bg_len - len(background_gdf)} background points overlapping with presence grid cells.")
            
            # Re-filter density series to match remaining background points
            background_density = background_density.loc[background_gdf.index]

            # Drop the grid_index column after use if it was temporarily added
            # presence_gdf = presence_gdf.drop(columns=["grid_index"], errors='ignore')
            # background_gdf = background_gdf.drop(columns=["grid_index"], errors='ignore')

    if subset_background:
        n_bg_calculated = calculate_background_points(len(presence_gdf))
        logger.info(f"Subsetting background points to approximately {n_bg_calculated}.")
        
        if order_by_density_for_subset and not background_density.empty:
            logger.info("Subsetting background points by density.")
            # Ensure background_density index aligns with background_gdf for sorting
            valid_indices = background_density.index.intersection(background_gdf.index)
            background_gdf = background_gdf.loc[valid_indices]
            background_density = background_density.loc[valid_indices]
            
            # Sort by density and take top n_bg
            background_gdf = background_gdf.loc[
                background_density.sort_values(ascending=False).index
            ]
            background_gdf = background_gdf.head(n_bg_calculated)
        elif not background_density.empty:
            logger.info("Subsetting background points randomly.")
            n_bg_sample = min(n_bg_calculated, len(background_gdf))
            if n_bg_sample > 0 :
                background_gdf = background_gdf.sample(n=n_bg_sample, random_state=42) # Add random_state for reproducibility
            else:
                logger.warning("No background points available for random sampling after filtering.")
                background_gdf = background_gdf.iloc[0:0] # Empty gdf with same columns
        else:
            logger.warning("Background density not provided or empty; cannot subset by density. Performing random sampling if points exist.")
            n_bg_sample = min(n_bg_calculated, len(background_gdf))
            if n_bg_sample > 0:
                 background_gdf = background_gdf.sample(n=n_bg_sample, random_state=42)
            else:
                logger.warning("No background points available for random sampling.")
                background_gdf = background_gdf.iloc[0:0]
        logger.info(f"Number of background points after subsetting: {len(background_gdf)}.")

    # Select final columns (input_vars + geometry for stacking and weighting)
    # Ensure 'geometry' is always present
    cols_to_keep = list(set(input_vars + [presence_gdf.geometry.name]))
    presence_gdf = presence_gdf[cols_to_keep]
    background_gdf = background_gdf[cols_to_keep]

    if drop_na:
        logger.info("Dropping rows with NA values from presence and background sets.")
        original_pres_len = len(presence_gdf)
        original_bg_len = len(background_gdf)
        presence_gdf.dropna(subset=input_vars, inplace=True)
        background_gdf.dropna(subset=input_vars, inplace=True)
        logger.info(f"Removed {original_pres_len - len(presence_gdf)} NA rows from presences.")
        logger.info(f"Removed {original_bg_len - len(background_gdf)} NA rows from background.")

    if presence_gdf.empty:
        logger.warning("No presence points remaining after pre-processing. Cannot proceed.")
        # Return an empty GeoDataFrame with expected columns if needed by downstream code
        # Or raise an error.
        return gpd.GeoDataFrame(columns=cols_to_keep + ["class", "sample_weight"], geometry="geometry", crs=presence_gdf.crs or background_gdf.crs or "EPSG:4326")
    if background_gdf.empty and subset_background: # Only warn if we expected background points
        logger.warning("No background points remaining after pre-processing.")
        # Depending on model requirements, this might be an error or acceptable (e.g. presence-only models)
        # For MaxEnt, background points are crucial.

    logger.info("Stacking presence and background points.")
    # Ensure columns are exactly the same for concatenation, elapid.stack_geodataframes might handle this
    # For safety, align columns if they might differ due to prior processing steps
    # common_cols = list(set(presence_gdf.columns) & set(background_gdf.columns))
    # presence_gdf = presence_gdf[common_cols]
    # background_gdf = background_gdf[common_cols]
    
    occurrence_stacked = ela.stack_geodataframes(
        presence_gdf, 
        background_gdf, 
        add_class_label=True # Adds 'class' column (1 for presence, 0 for background)
    )
    
    logger.info(f"Calculating sample weights (n_neighbors={sample_weight_n_neighbors}).")
    # distance_weights requires all points (presence and background) to be in the same GDF
    occurrence_stacked["sample_weight"] = ela.distance_weights(
        occurrence_stacked, n_neighbors=sample_weight_n_neighbors
    )

    logger.info(f"Prepared occurrence data: {len(occurrence_stacked)} total points.")
    return occurrence_stacked 