import logging
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional

import typer
import geopandas as gpd
import pandas as pd
import xarray as xr

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary, bbox_filter # Removed load_spatial_config, construct_transform_shift_bounds
from sdm.data.loaders import load_os_shps, generate_parquets
from sdm.data.transforms import process_roads
from sdm.raster.processing import calculate_feature_cover, calculate_distances 
# Removed reproject_data, squeeze_dataset from here as they are in raster.utils, not directly called by this script's main.
# rasterise_gdf is also in raster.utils and called by calculate_feature_cover.

# Get the project configuration
# from species_sdm.utils import load_config # Not used directly in this main function
# config = load_config()
# SPATIAL_CONFIG = config["spatial"] # Example

app = typer.Typer()

@app.command()
def main(
    output_dir: Path = typer.Option("data/evs", help="Directory to save output EV files."),
    boundary_path: Path = typer.Option("data/processed/boundary.geojson", help="Path to the boundary file."),
    buffer_distance: float = typer.Option(7000, help="Buffer distance for the boundary."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
    load_from_shp: bool = typer.Option(False, help="Load OS data directly from SHP instead of expecting Parquet files (slower).")
) -> None:
    """
    Processes Ordnance Survey (OS) data to generate environmental variables:
    - OS VectorMap Local Buildings: cover
    - OS VectorMap Local Roads: cover (major/minor), distance to (major/minor)
    - OS VectorMap Local Water: cover, distance to
    - OS Open BuiltUpAreas: cover, distance to
    - OS Open Greenspace: cover, distance to
    """
    setup_logging(verbose=verbose)
    logging.info("Starting OS data processing workflow.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load boundary
    # For loading spatial config, it's assumed to be part of the main config or handled by load_boundary if it needs CRS.
    # The original process_os_data.py directly called load_spatial_config and construct_transform_shift_bounds.
    # These are now encapsulated within species_sdm library functions or loaded from a main config.
    # Here, we primarily need the boundary geometry and its CRS.
    # load_boundary_and_transform from io.py might be useful if the transform is needed directly here.
    # However, individual components usually get this from config or parameters.

    boundary_gdf = load_boundary(boundary_path, buffer_distance=buffer_distance)
    boundary_poly = boundary_gdf.unary_union # For filtering

    # Define OS datasets to process
    # Parquet files are preferred for speed. SHP loading is an option.
    os_datasets_to_load = [
        "Building", 
        "FunctionalSite", # Contains BuiltUpArea, Greenspace - will need filtering
        "Road", 
        "Water"
    ]
    
    source_os_data_dir = Path("data/raw/big-files/os-vector-map") # Default raw OS data location
    processed_os_data_dir = Path("data/processed/os-data") # Default for parquets

    if load_from_shp:
        logging.info("Generating Parquet files from SHP (load_from_shp=True)")
        os_parquet_paths = generate_parquets(
            datasets=os_datasets_to_load,
            output_dir=processed_os_data_dir,
            source_data_dir=source_os_data_dir,
            boundary=boundary_poly, # Filter during parquet generation
            overwrite=False # Be cautious with overwriting
        )
    else:
        logging.info("Using existing Parquet files (load_from_shp=False)")
        os_parquet_paths = [processed_os_data_dir / f"os-{name}.parquet" for name in os_datasets_to_load]
        # Check if all expected parquets exist
        missing_parquets = [p for p in os_parquet_paths if not p.exists()]
        if missing_parquets:
            logging.error(f"Missing parquet files: {missing_parquets}. Run with --load-from-shp or ensure they exist.")
            raise typer.Exit(code=1)

    # Load data from Parquet
    # Create a dictionary to hold the GeoDataFrames
    os_data_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    for name, path in zip(os_datasets_to_load, os_parquet_paths):
        logging.info(f"Loading {name} from {path}")
        # The bbox_filter is useful here if parquets are large and cover a much wider area.
        # For now, assuming parquets are already reasonably filtered or manageable in size.
        os_data_gdfs[name] = gpd.read_parquet(path) #, filters=bbox_filter(boundary_poly.bounds))


    # --- Feature Specific Processing ---
    
    # BuiltUpArea and Greenspace are within FunctionalSite
    functional_site_gdf = os_data_gdfs.get("FunctionalSite")
    built_up_areas_gdf = gpd.GeoDataFrame()
    greenspace_gdf = gpd.GeoDataFrame()

    if functional_site_gdf is not None and not functional_site_gdf.empty:
        built_up_areas_gdf = functional_site_gdf[functional_site_gdf["DESCRIPTIVETERM"] == "Built Up Area"].copy()
        greenspace_gdf = functional_site_gdf[functional_site_gdf["CLASS"] == "Greenspace"].copy()
        logging.info(f"Extracted {len(built_up_areas_gdf)} BuiltUpArea features.")
        logging.info(f"Extracted {len(greenspace_gdf)} Greenspace features.")
    else:
        logging.warning("FunctionalSite data not loaded or empty, BuiltUpArea and Greenspace will be empty.")

    # Roads: major and minor
    roads_gdf = os_data_gdfs.get("Road")
    major_roads_gdf, minor_roads_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame() # Initialize as empty
    if roads_gdf is not None and not roads_gdf.empty:
        major_roads_gdf, minor_roads_gdf = process_roads(roads_gdf)
    else:
        logging.warning("Road data not loaded or empty.")

    # Prepare dictionary for cover and distance calculations
    features_for_calc: Dict[str, gpd.GeoDataFrame] = {
        "os_buildings": os_data_gdfs.get("Building", gpd.GeoDataFrame()),
        "os_water": os_data_gdfs.get("Water", gpd.GeoDataFrame()),
        "os_built_up_area": built_up_areas_gdf,
        "os_greenspace": greenspace_gdf,
        "os_major_roads": major_roads_gdf,
        "os_minor_roads": minor_roads_gdf,
    }
    # Filter out any empty GDFs before passing to calculations to avoid issues
    features_for_calc = {name: gdf for name, gdf in features_for_calc.items() if gdf is not None and not gdf.empty}
    if not features_for_calc:
        logging.error("No valid features loaded for OS data processing. Exiting.")
        raise typer.Exit(code=1)

    # --- Calculate Feature Cover ---
    logging.info("Calculating OS feature cover...")
    os_feature_cover_ds = calculate_feature_cover(
        feature_gdfs=features_for_calc,
        boundary=boundary_gdf, # Pass the full boundary GeoDataFrame
        target_resolution=100 # Example resolution, make configurable if needed
    )
    cover_output_path = output_dir / "os-feature-cover.tif"
    os_feature_cover_ds.rio.to_raster(cover_output_path) # Removed kötős, compression can be added later
    logging.info(f"Saved OS feature cover to {cover_output_path}")

    # --- Calculate Distance to Features ---
    logging.info("Calculating distance to OS features...")
    # Select only point/line features for meaningful distance calculation if needed, or ensure gdf is appropriate
    # For example, distance to polygons often means distance to their boundary/centroid
    # calculate_distances expects point features for its cKDTree approach
    # Here, we might need to convert polygon centroids or use a different distance method for polygons.
    # For simplicity, the original `calculate_distances` (now in raster.processing) used centroids implicitly.
    # This part might need refinement based on desired distance metric for polygons.
    
    # We will calculate distance to: Water, BuiltUpArea, Greenspace, MajorRoads, MinorRoads
    # Buildings are polygons, distance to them might be less standard with centroid approach.
    # Let's select relevant features for distance.
    features_for_distance = {
        name: gdf for name, gdf in features_for_calc.items() 
        if name not in ["os_buildings"] # Exclude buildings for now, or handle appropriately
    }
    
    if features_for_distance: # Check if dictionary is not empty
        os_distance_ds = calculate_distances(
            feature_gdfs=features_for_distance,
            boundary=boundary_gdf, # Pass the full boundary GeoDataFrame
            resolution=100 # Example resolution
        )
        distance_output_path = output_dir / "os-distance-to-feature.tif"
        os_distance_ds.rio.to_raster(distance_output_path) # Removed kötős, compression can be added later
        logging.info(f"Saved OS distance to features to {distance_output_path}")
    else:
        logging.warning("No features selected for distance calculation.")
        distance_output_path = None


    logging.info("OS data processing workflow finished.")
    logging.info(f"Outputs: Cover: {cover_output_path}, Distances: {distance_output_path if distance_output_path else 'N/A'}")

if __name__ == "__main__":
    app() 