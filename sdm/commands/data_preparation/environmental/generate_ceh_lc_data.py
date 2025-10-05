import logging
from pathlib import Path

import xarray as xr
import rioxarray as rxr # For direct rio operations if needed, though utils are preferred
import numpy as np
from rasterio.enums import Resampling # For specifying resampling method

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary, load_spatial_config
from sdm.data.landcover import get_ceh_land_cover_codes_v2023, define_broad_habitat_categories
from sdm.raster.processing import create_binary_raster_from_category, aggregate_categorical_rasters
# The reproject_data utility can be used if its parameterization fits.
# Original script used lc_processed.rio.reproject directly with resampling=0 (NearestNeighbor).
# Our reproject_data uses Resampling.bilinear by default, but accepts a resampling arg.
from sdm.raster.utils import construct_transform_shift_bounds

def generate_ceh_lc_data(
    output_dir: Path,
    boundary_path: Path,
    ceh_data_path: Path,
    buffer_distance_m: float = 7000,  # Match original default
    output_resolution_m: int = 100,
    verbose: bool = False
) -> Path:
    """
    Process CEH land cover data based on a given boundary.
    
    Args:
        output_dir: Directory where the output data will be saved
        boundary_path: Path to the boundary GeoJSON file
        ceh_data_path: Path to the CEH land cover data file
        buffer_distance_m: Buffer distance in meters to add around the boundary
        output_resolution_m: Target output resolution in meters for the processed land cover EV
        verbose: Enable verbose logging
        
    Returns:
        Path to the output file
        
    This function performs the following steps:
    1. Loads the boundary from the specified path
    2. Loads the CEH land cover data
    3. Clips the data to the boundary with buffer
    4. Converts the land cover data into category layers
    5. Coarsens the data to a lower resolution (e.g., 100m)
    6. Performs feature engineering (aggregates categories)
    7. Reprojects to the model CRS and writes the output
    """
    setup_logging(verbose=verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load land cover data first to get its CRS
    logging.info("Loading CEH land cover data from %s", ceh_data_path)
    land_cover = rxr.open_rasterio(ceh_data_path)
    
    # Load and process boundary - project to land cover CRS first
    logging.info("Loading boundary from %s", boundary_path)
    boundary = load_boundary(
        boundary_path, buffer_distance=0, target_crs=land_cover.rio.crs
    )
    boundary["geometry"] = boundary.geometry.buffer(buffer_distance_m)
    
    # Clip land cover to buffered boundary using land cover's CRS
    logging.info("Clipping land cover to boundary")
    land_cover = land_cover.rio.clip_box(*boundary.total_bounds, crs=land_cover.rio.crs)
    land_cover = land_cover.where(land_cover != land_cover.rio.nodata, np.nan)
    land_cover.rio.write_nodata(np.nan, inplace=True)
    
    # Create category layers
    logging.info("Converting land cover to category layers")
    land_cover_key = get_ceh_land_cover_codes_v2023()
    land_cover_categories = [
        create_binary_raster_from_category(land_cover[0], int(key), label, nodata_val=np.nan)
        for key, label in land_cover_key.items()
    ]
    lc_stack = xr.merge(land_cover_categories)
    
    # Calculate area per pixel for the original resolution (typically 10m)
    original_resolution = abs(land_cover.rio.resolution()[0])  # Get absolute resolution
    area_per_pixel = original_resolution * original_resolution
    lc_stack = lc_stack * area_per_pixel  # Convert to area units (m²)
    
    # Coarsen to specified resolution (e.g. 10m -> 100m)
    coarsen_factor = int(output_resolution_m / original_resolution)
    logging.info("Coarsening data by factor of %d to %dm", coarsen_factor, output_resolution_m)
    
    lc_coarse = lc_stack.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").sum(skipna=False)
    lc_coarse = lc_coarse.astype(np.float32)
    
    # Perform feature engineering
    logging.info("Performing feature engineering")
    broad_habitat_categories = define_broad_habitat_categories()
    categories_to_drop = ["Inland rock", "Marine, Littoral", "Freshwater"]
    lc_processed = aggregate_categorical_rasters(lc_coarse, aggregation_map=broad_habitat_categories, categories_to_drop=categories_to_drop)
    
    # Get spatial config and construct transform for final reprojection
    spatial_config = load_spatial_config()
    model_transform, _bounds = construct_transform_shift_bounds(
        tuple(boundary.to_crs(spatial_config["crs"]).total_bounds), output_resolution_m
    )
    
    # Reproject to model CRS using nearest neighbor (appropriate for categorical data)
    lc_projected = lc_processed.rio.reproject(
        spatial_config["crs"], 
        transform=model_transform,
        resampling=Resampling.nearest  # Use nearest neighbor for categorical data
    )

    # Write output
    output_path = output_dir / f"ceh-land-cover-{output_resolution_m}m.tif"
    logging.info("Writing data to %s", output_path)
    lc_projected.rio.to_raster(output_path)
    
    return output_path 