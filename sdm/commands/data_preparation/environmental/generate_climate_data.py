import logging
from pathlib import Path
from typing import Dict, Union, List

import geopandas as gpd # For type hinting boundary_gdf

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.data.climate import (
    fetch_worldclim_datasets,
    reproject_climate_datasets,
    assign_climate_variable_names, # This might be integrated or skipped if datasets are named well by loader
    write_climate_data,
    calculate_climate_statistics
)

def generate_climate_data(
    output_dir: Path,
    boundary_path: Path,
    worldclim_cache_dir: Path,
    variables: List[str] = ["bio", "tavg", "prec", "wind"], # Default variables from original script
    run_stats: bool = False,
    verbose: bool = False
) -> Dict[str, Path]:
    """
    Downloads, processes, and saves WorldClim climate data layers.
    
    Steps include:
    1. Loading study area boundary and spatial configuration.
    2. Fetching specified WorldClim variables (e.g., bioclim, tavg, prec, wind), caching downloads.
    3. Reprojecting datasets to the model's CRS and resolution.
    4. (Optionally) Assigning descriptive names to bands/variables within datasets.
    5. Writing processed climate layers to GeoTIFF files.
    6. (Optionally) Calculating and logging basic statistics.

    Args:
        output_dir: Directory to save output climate TIFF files.
        boundary_path: Path to the boundary file for clipping and context.
        worldclim_cache_dir: Directory to cache downloaded WorldClim files.
        variables: List of WorldClim variables to download (e.g., bio, tavg, prec, srad, wind, tmin, tmax).
        run_stats: Calculate and log basic statistics for downloaded variables.
        verbose: Enable verbose logging.

    Returns:
        Dictionary mapping variable names to their output file paths.
    """
    setup_logging(verbose=verbose)
    logging.info(f"Starting climate data generation. Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    worldclim_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load boundary and spatial parameters (CRS, transform, resolution)
    try:
        boundary_gdf, model_transform, _, spatial_config = load_boundary_and_transform(boundary_path)
    except FileNotFoundError:
        logging.error(f"Boundary file not found at: {boundary_path}. Cannot proceed.")
        raise
    except Exception as e:
        logging.error(f"Error loading boundary or spatial config: {e}")
        raise

    model_crs = boundary_gdf.crs
    model_resolution = spatial_config["resolution"]

    logging.info(f"Fetching WorldClim variables: {variables}")
    raw_climate_datasets = fetch_worldclim_datasets(
        variables=variables,
        boundary_gdf=boundary_gdf, # Pass boundary for potential clipping in loader
        cache_folder=worldclim_cache_dir
    )

    if not raw_climate_datasets:
        logging.error("No climate datasets were fetched. Exiting.")
        raise ValueError("No climate datasets were fetched")

    logging.info("Reprojecting climate datasets to model grid...")
    reprojected_climate_datasets = reproject_climate_datasets(
        datasets=raw_climate_datasets,
        target_crs=model_crs,
        target_transform=model_transform,
        target_resolution=model_resolution
    )
    
    # assign_climate_variable_names was used in original to set long_name attributes
    # The ClimateData._set_band_names and tidy_long_name in loaders.py, 
    # and the revised assign_climate_variable_names in climate_processing.py 
    # aim to handle naming. If bands are correctly named by these, this step might be for refinement.
    logging.info("Assigning/verifying climate variable names...")
    named_climate_datasets = assign_climate_variable_names(reprojected_climate_datasets)

    logging.info("Writing climate datasets to GeoTIFF files...")
    output_file_paths = write_climate_data(
        climate_datasets=named_climate_datasets, 
        output_dir=output_dir
    )

    if run_stats:
        logging.info("Calculating climate statistics...")
        calculate_climate_statistics(named_climate_datasets, output_dir)

    logging.info("Climate data generation finished.")
    logging.info(f"Output files: {list(output_file_paths.values())}")
    return output_file_paths 