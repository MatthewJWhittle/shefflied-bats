import logging
from pathlib import Path
from typing import Union

import typer
import geopandas as gpd
import xarray as xr
# import rioxarray as rxr # Not directly used in main, but by functions it calls

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform # Gets boundary, transform, bounds, spatial_config
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.data.spatial import calculate_coastal_distance

app = typer.Typer()

@app.command()
def main(
    boundary_path: Path = typer.Option(
        "data/processed/boundary.geojson", 
        help="Path to the boundary file (e.g., GeoJSON) for the study area."
    ),
    output_dir: Path = typer.Option(
        "data/evs", 
        help="Directory to save the output coastal_distance.tif file."
    ),
    bgs_geocoast_shp_path: Path = typer.Option(
        "data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp",
        help="Path to the BGS GeoCoast Authority Area Inundation shapefile."
    ),
    buffer_dist_km_for_sea: float = typer.Option(
        10.0, help="Buffer distance in km to create the initial 'sea' zone from coastline."
    ),
    simplify_tolerance_m: float = typer.Option(
        1000.0, help="Simplification tolerance in meters for coastal geometry processing."
    ),
    min_sea_area_km2: float = typer.Option(
        1.0, help="Minimum area in km^2 for a sea polygon to be retained after explosion."
    ),
    distance_calc_resolution_factor: int = typer.Option(
        10, help="Factor to multiply base model resolution for distance calculation (e.g., 10x for faster coarse calculation)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Generates a coastal distance raster layer using BGS GeoCoast data.

    The process involves:
    1. Loading and buffering the study area boundary.
    2. Loading BGS GeoCoast inundation data.
    3. Processing the coastal data to define a 'sea' polygon (dissolve, simplify, buffer, difference).
    4. Calculating distance from a grid to this 'sea' polygon.
    5. Reprojecting, squeezing, and clipping the resulting distance raster.
    6. Saving the final coastal_distance.tif.
    """
    setup_logging(verbose=verbose)
    logging.info("Creating coastal distance dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load boundary and get model grid parameters
    # Note: load_boundary_and_transform also provides spatial_config which includes resolution and CRS
    boundary_gdf, model_transform, grid_bounds, spatial_config = load_boundary_and_transform(
        boundary_path
        # Buffer distance for boundary_gdf is handled by load_boundary_and_transform if its default is used
        # or if a different buffer for the EV generation itself is needed, it can be passed here.
    )
    model_crs = boundary_gdf.crs # CRS from the loaded (and possibly reprojected) boundary
    model_resolution = spatial_config["resolution"]

    logging.info("Loading BGS GeoCoast data from: %s", bgs_geocoast_shp_path)
    if not bgs_geocoast_shp_path.exists():
        logging.error(f"BGS GeoCoast shapefile not found: {bgs_geocoast_shp_path}")
        raise typer.Exit(code=1)
        
    coast_gdf = gpd.read_file(bgs_geocoast_shp_path)
    coast_gdf = coast_gdf.to_crs(model_crs) # Ensure same CRS as boundary

    logging.info("Processing coastal geometry (dissolve, simplify)...")
    coast_gdf_processed = coast_gdf.dissolve(dropna=False)
    coast_gdf_processed["geometry"] = coast_gdf_processed.simplify(simplify_tolerance_m)

    logging.info("Creating 'sea' zone polygon...")
    # Buffer the coast and calculate a difference to create a 'sea' zone
    # Convert buffer_dist_km_for_sea from km to meters (assuming CRS is in meters)
    buffer_m = buffer_dist_km_for_sea * 1000 
    sea_zone = (
        coast_gdf_processed.buffer(buffer_m) # Use geometry from processed gdf
        .simplify(simplify_tolerance_m)
        .difference(coast_gdf_processed.geometry.iloc[0]) # Difference with the (single) dissolved geometry
    )
    # Explode the polygon and drop anything with an area less than min_sea_area_km2
    # Convert min_sea_area_km2 to m^2
    min_area_m2 = min_sea_area_km2 * 1_000_000
    sea_exploded_gdf = sea_zone.explode(index_parts=True).to_frame(name='geometry')
    sea_filtered_gdf = sea_exploded_gdf[sea_exploded_gdf.area > min_area_m2]
    
    if sea_filtered_gdf.empty:
        logging.error("No 'sea' polygons remaining after filtering by area. Cannot calculate coastal distance.")
        raise typer.Exit(code=1)
    sea_polygon = sea_filtered_gdf.unary_union

    logging.info("Calculating distances to 'sea' zone...")
    # Calculate the distance to the coast at a potentially coarser resolution to speed things up
    distance_calc_resolution = model_resolution * distance_calc_resolution_factor
    coastal_distance_xr = calculate_coastal_distance(
        geom=sea_polygon,
        boundary_gdf=boundary_gdf, # Pass the GeoDataFrame for CRS and context
        grid_bounds=grid_bounds,    # Use the bounds from load_boundary_and_transform
        resolution=distance_calc_resolution,
        var_name="distance_to_coast",
    )

    logging.info("Reprojecting coastal distance data to model grid...")
    coastal_distance_xr = reproject_data(
        coastal_distance_xr, 
        crs=model_crs, 
        transform=model_transform, 
        resolution=model_resolution
    )

    logging.info("Squeezing dataset...")
    coastal_distance_xr = squeeze_dataset(coastal_distance_xr)

    logging.info("Masking data to boundary...")
    coastal_distance_xr = coastal_distance_xr.rio.clip(
        [boundary_gdf.unary_union], crs=model_crs, all_touched=True # Use all_touched for consistency
    )

    output_path = output_dir / "coastal_distance.tif"
    logging.info(f"Saving coastal distance data to {output_path}...")
    coastal_distance_xr.rio.to_raster(output_path) # Add compression options if desired

    logging.info(f"Coastal distance dataset created successfully: {output_path}")

if __name__ == "__main__":
    app() 