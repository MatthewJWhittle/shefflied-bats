from pathlib import Path
from typing import Dict, Union, List
import logging

import xarray as xr
import numpy as np
import rioxarray as rxr

from data_prep.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)
from data_prep.utils.config import setup_logging


def ceh_lc_types() -> Dict[str, str]:
    """
    Return a mapping of CEH land cover codes to descriptive names.
    
    Returns:
        Dict[str, str]: Dictionary mapping land cover codes to human-readable names
    """
    return {
        "1": "Broadleaved woodland",
        "2": "Coniferous woodland",
        "3": "Arable",
        "4": "Improved grassland",
        "5": "Neutral grassland",
        "6": "Calcareous grassland",
        "7": "Acid grassland",
        "8": "Fen, Marsh and Swamp",
        "9": "Heather and shrub",
        "10": "Heather grassland",
        "11": "Bog",
        "12": "Inland rock",
        "13": "Saltwater",
        "14": "Freshwater",
        "15": "Supralittoral rock",
        "16": "Supralittoral sediment",
        "17": "Littoral rock",
        "18": "Littoral sediment",
        "19": "Saltmarsh",
        "20": "Urban",
        "21": "Suburban",
    }


def get_land_cover_category(array: xr.DataArray, key: int, label: str) -> xr.Dataset:
    """
    Convert land cover array to binary category array (1 where category matches, 0 elsewhere).
    
    Args:
        array: Input land cover array
        key: Land cover type ID
        label: Land cover type name
        
    Returns:
        xr.Dataset: Dataset with binary indicators for the specified land cover type
    """
    # Generate an array of zeros the same shape as the input array
    cat_array = xr.zeros_like(array, dtype=np.float64)
    # Set the output array to 1 where it matches the category int key
    cat_array = cat_array.where(array != key, 1)
    # Where the input array has missing values, set the output array to missing
    cat_array = cat_array.where(~np.isnan(array), np.nan)
    # Convert it to a dataset with the habitat category as the name
    cat_array = cat_array.to_dataset(name=label)
    return cat_array


def create_broad_habitat_categories() -> Dict[str, List[str]]:
    """
    Define broad habitat categories as aggregations of specific land cover types.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping broad habitat names to lists of specific habitat types
    """
    return {
        "Grassland": [
            "Neutral grassland",
            "Calcareous grassland",
            "Acid grassland",
        ],
        "Marine, Littoral": [
            "Saltwater",
            "Supralittoral rock",
            "Supralittoral sediment",
            "Littoral rock",
            "Littoral sediment",
            "Saltmarsh",
        ],
        "Upland Heathland": ["Heather and shrub", "Heather grassland"],
        "Wetland": ["Bog", "Fen, Marsh and Swamp"],
    }


def combine_habitats(land_cover_data: xr.Dataset) -> xr.Dataset:
    """
    Aggregate land cover categories into broader habitat types and remove unnecessary categories.
    
    Args:
        land_cover_data: Dataset containing land cover categories
        
    Returns:
        xr.Dataset: Dataset with aggregated habitat categories
    """
    result = land_cover_data.copy()
    
    # Define broad habitats
    broad_habitats = create_broad_habitat_categories()
    
    # Add up the area of each habitat type in each broad habitat
    # Add that variable to the data and remove the sub-habitat variables
    for broad_habitat, habitat_types in broad_habitats.items():
        valid_types = [ht for ht in habitat_types if ht in result.data_vars]
        if valid_types:
            result[broad_habitat] = result[valid_types].to_array().sum(axis=0)
            result = result.drop_vars(valid_types)
    
    # Remove categories that are unlikely to be useful for the model
    categories_to_drop = ["Inland rock", "Marine, Littoral", "Freshwater"]
    for category in categories_to_drop:
        if category in result.data_vars:
            result = result.drop_vars(category)
            
    return result


def main(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
    ceh_data_path: Union[str, Path] = "data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif",
    resolution: int = 10,
    coarsen_factor: int = 10,
) -> Path:
    """
    Main function to process CEH land cover data based on a given boundary.
    
    Args:
        output_dir: Directory where the output data will be saved
        boundary_path: Path to the boundary GeoJSON file
        buffer_distance: Buffer distance in meters to add around the boundary
        ceh_data_path: Path to the CEH land cover data file
        resolution: Resolution in meters of the input data
        coarsen_factor: Factor by which to coarsen the data (e.g., 10 to go from 10m to 100m)
        
    Returns:
        Path: Path to the output file
        
    This function performs the following steps:
    1. Loads the boundary from the specified path
    2. Loads the CEH land cover data
    3. Clips the data to the boundary with buffer
    4. Converts the land cover data into category layers
    5. Coarsens the data to a lower resolution (e.g., 100m)
    6. Performs feature engineering (aggregates categories)
    7. Reprojects to the model CRS and writes the output
    """
    setup_logging()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process boundary
    logging.info(f"Loading boundary from {boundary_path}")
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs="EPSG:27700"
    )
    
    # Load land cover data
    logging.info(f"Loading CEH land cover data from {ceh_data_path}")
    land_cover = rxr.open_rasterio(ceh_data_path)
    
    # Project boundary to land cover CRS
    boundary = boundary.to_crs(land_cover.rio.crs)
    boundary["geometry"] = boundary.geometry.buffer(buffer_distance)
    
    # Clip land cover to buffered boundary
    logging.info("Clipping land cover to boundary")
    land_cover = land_cover.rio.clip_box(*boundary.total_bounds, crs=boundary.crs)
    land_cover = land_cover.where(land_cover != land_cover.rio.nodata, np.nan)
    land_cover.rio.write_nodata(np.nan, inplace=True)
    
    # Create category layers
    logging.info("Converting land cover to category layers")
    land_cover_key = ceh_lc_types()
    land_cover_categories = [
        get_land_cover_category(land_cover[0], int(key), label)
        for key, label in land_cover_key.items()
    ]
    lc_stack = xr.merge(land_cover_categories)
    
    # Convert to area and coarsen to lower resolution
    logging.info(f"Coarsening data by factor of {coarsen_factor}")
    area_per_pixel = resolution * resolution
    lc_stack = lc_stack * area_per_pixel  # Convert to area units (m²)
    
    # Coarsen to specified resolution (e.g. 10m -> 100m)
    target_resolution = resolution * coarsen_factor
    lc_coarse = lc_stack.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").sum(skipna=False)
    
    # Perform feature engineering
    logging.info("Performing feature engineering")
    lc_processed = combine_habitats(lc_coarse)
    
    # Reproject to model CRS
    spatial_config = load_spatial_config()
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), target_resolution
    )
    
    lc_processed = lc_processed.rio.reproject(
        spatial_config["crs"], 
        transform=model_transform,
        resampling=0  # Nearest neighbor
    )
    
    # Write output
    output_path = output_dir / f"ceh-land-cover-{target_resolution}m.tif"
    logging.info(f"Writing data to {output_path}")
    lc_processed.rio.to_raster(output_path)
    
    return output_path



if __name__ == "__main__":
    main()
