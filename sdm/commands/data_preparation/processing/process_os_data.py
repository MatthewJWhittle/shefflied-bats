import logging
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import box

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary, load_spatial_config
from sdm.data.os import generate_parquets, process_roads
from sdm.raster.processing import calculate_feature_cover, calculate_distances
from sdm.raster.utils import construct_transform_shift_bounds, reproject_data, squeeze_dataset

def process_os_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance: float = 7000,
    load_from_shp: bool = False,
    verbose: bool = False
) -> Tuple[Path, Path]:
    """Process Ordnance Survey data to generate environmental variables.

    This function processes OS data to create two main outputs:
    1. Feature coverage density (percentage of area covered by each feature type)
    2. Distance to nearest feature (for each feature type)

    Args:
        output_dir: Directory where output rasters will be saved.
        boundary_path: Path to GeoJSON file defining the area of interest.
        buffer_distance: Distance in meters to buffer the boundary.
        load_from_shp: Whether to load OS data from shapefiles instead of cached parquet files.
        verbose: Enable verbose logging.

    Returns:
        Tuple containing:
        - cover_path: Path to the feature cover raster
        - distance_path: Path to the distance matrix raster

    Raises:
        FileNotFoundError: If input files are not found.
        ValueError: If boundary or buffer distance are invalid.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logging.info("Starting OS data processing pipeline")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", output_dir)

    logging.info("Loading boundary from %s with %dm buffer", boundary_path, buffer_distance)
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs="EPSG:27700"
    )
    
    logging.info("Loading spatial configuration")
    spatial_config = load_spatial_config()
    boundary.to_crs(spatial_config["crs"], inplace=True)
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), spatial_config["resolution"]
    )

    # Load OS data
    datasets = ["Building", "Water", "Woodland", "Road"]
    parquet_paths = generate_parquets(
        datasets, dir="data/raw/big-files/os-data", boundary=box(*bounds), overwrite=load_from_shp
    )
    os_data = {
        name: gpd.read_parquet(path) for name, path in zip(datasets, parquet_paths)
    }

    # Process roads
    major_roads, minor_roads = process_roads(os_data["Road"])

    # Prepare feature datasets
    feature_gdfs = {
        "major_roads": major_roads,
        "minor_roads": minor_roads,
        "woodland": os_data["Woodland"],
        "water": os_data["Water"],
        "buildings": os_data["Building"],
    }

    # Calculate and save feature cover
    feature_cover = calculate_feature_cover(feature_gdfs, boundary)
    feature_cover = reproject_data(
        feature_cover,
        spatial_config["crs"],
        transform=model_transform,
        resolution=spatial_config["resolution"],
    )
    logging.info("Writing feature cover raster")
    cover_path = output_dir / "os-feature-cover.tif"
    # write it as a dataset to keep band names
    feature_cover = squeeze_dataset(feature_cover) # type: ignore
    feature_cover.rio.to_raster(cover_path)

    # Calculate and save distance matrices
    distance_array = calculate_distances(feature_gdfs, boundary)
    distance_array = reproject_data(
        distance_array,
        spatial_config["crs"],
        transform=model_transform,
        resolution=spatial_config["resolution"],
    )
    logging.info("Writing distance matrix raster")
    distance_path = output_dir / "os-distance-to-feature.tif"
    distance_array = squeeze_dataset(distance_array) # type: ignore
    distance_array.rio.to_raster(
        distance_path
    )

    logging.info("OS data processing complete")
    logging.info("Output files saved to: %s", output_dir)
    return cover_path, distance_path 