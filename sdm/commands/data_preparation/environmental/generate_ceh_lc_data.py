import logging
from pathlib import Path
from typing import Union # Not strictly needed

import xarray as xr
import rioxarray as rxr # For direct rio operations if needed, though utils are preferred
import numpy as np
from rasterio.enums import Resampling # For specifying resampling method

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.data.landcover import get_ceh_land_cover_codes_v2023, define_broad_habitat_categories
from sdm.raster.processing import create_binary_raster_from_category, aggregate_categorical_rasters
# The reproject_data utility can be used if its parameterization fits.
# Original script used lc_processed.rio.reproject directly with resampling=0 (NearestNeighbor).
# Our reproject_data uses Resampling.bilinear by default, but accepts a resampling arg.
from sdm.raster.utils import reproject_data, squeeze_dataset, load_spatial_config, construct_transform_shift_bounds

def generate_ceh_lc_data(
    output_dir: Path,
    boundary_path: Path,
    ceh_data_path: Path,
    buffer_distance_m: float = 1000,
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
    
    # Load and process boundary
    logging.info(f"Loading boundary from {boundary_path}")
    boundary, transform, _bounds, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance_m
    )
    crs = spatial_config["crs"]
    # Load land cover data
    logging.info(f"Loading CEH land cover data from {ceh_data_path}")
    land_cover = rxr.open_rasterio(ceh_data_path)
    
    # Project boundary to land cover CRS
    boundary.to_crs(crs, inplace=True)
    boundary["geometry"] = boundary.geometry.buffer(buffer_distance_m)
    
    # Clip land cover to buffered boundary
    logging.info("Clipping land cover to boundary")
    land_cover = land_cover.rio.clip_box(*boundary.total_bounds, crs=crs)
    land_cover = land_cover.where(land_cover != land_cover.rio.nodata, np.nan)
    land_cover.rio.write_nodata(np.nan, inplace=True)
    
    # Create category layers
    logging.info("Converting land cover to category layers")
    land_cover_key = get_ceh_land_cover_codes_v2023()
    land_cover_categories = [
        create_binary_raster_from_category(land_cover[0], int(key), label)
        for key, label in land_cover_key.items()
    ]
    lc_stack = xr.merge(land_cover_categories)
    
    # Convert to area and coarsen to lower resolution
    logging.info(f"Coarsening data to {output_resolution_m}m")
    area_per_pixel = output_resolution_m * output_resolution_m
    lc_stack = lc_stack * area_per_pixel  # Convert to area units (m²)
    
    # Coarsen to specified resolution (e.g. 10m -> 100m)
    target_resolution = output_resolution_m
    coarsen_factor_x = int(output_resolution_m / land_cover.rio.resolution[0])
    coarsen_factor_y = int(output_resolution_m / land_cover.rio.resolution[1])

    lc_coarse = lc_stack.coarsen(x=coarsen_factor_x, y=coarsen_factor_y, boundary="trim").sum(skipna=False)
    lc_coarse = lc_coarse.astype(np.float32)
    
    # Perform feature engineering
    logging.info("Performing feature engineering")
    broad_habitat_categories = define_broad_habitat_categories()
    categories_to_drop = ["Inland rock", "Marine, Littoral", "Freshwater"]
    lc_processed = aggregate_categorical_rasters(lc_coarse, aggregation_map=broad_habitat_categories, categories_to_drop=categories_to_drop)
    
    # Reproject to model CRS
    lc_projected = reproject_data(
        array=lc_processed,
        crs=crs,
        transform=transform,
        resolution=output_resolution_m,
        resampling=Resampling.bilinear,
    )

    # Write output
    output_path = output_dir / f"ceh-land-cover-{target_resolution}m.tif"
    logging.info(f"Writing data to {output_path}")
    lc_projected.rio.to_raster(output_path)
    
    return output_path 