import logging
from pathlib import Path

import geopandas as gpd

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.data.spatial import calculate_coastal_distance


def load_and_process_coast_data(
    bgs_geocoast_shp_path: Path,
    model_crs: str,
    simplify_tolerance_m: float = 1000.0
) -> gpd.GeoDataFrame:
    """Load and process BGS GeoCoast data."""
    logging.info("Loading BGS GeoCoast data from: %s", bgs_geocoast_shp_path)
    if not bgs_geocoast_shp_path.exists():
        logging.error("BGS GeoCoast shapefile not found: %s", bgs_geocoast_shp_path)
        raise FileNotFoundError(f"BGS GeoCoast shapefile not found: {bgs_geocoast_shp_path}")
        
    coast_gdf = gpd.read_file(bgs_geocoast_shp_path)
    coast_gdf = coast_gdf.to_crs(model_crs)

    logging.info("Processing coastal geometry (dissolve, simplify)...")
    coast_gdf_processed = coast_gdf.dissolve(dropna=False)
    coast_gdf_processed["geometry"] = coast_gdf_processed.simplify(simplify_tolerance_m)
    
    return coast_gdf_processed


def create_sea_zone_polygon(
    coast_gdf_processed: gpd.GeoDataFrame,
    buffer_dist_km_for_sea: float = 10.0,
    simplify_tolerance_m: float = 1000.0,
    min_sea_area_km2: float = 1.0
) -> gpd.GeoSeries:
    """Create sea zone polygon from processed coast data."""
    logging.info("Creating 'sea' zone polygon...")
    buffer_m = buffer_dist_km_for_sea * 1000 
    sea_zone = (
        coast_gdf_processed.buffer(buffer_m)
        .simplify(simplify_tolerance_m)
        .difference(coast_gdf_processed.geometry.iloc[0])
    )
    min_area_m2 = min_sea_area_km2 * 1_000_000
    sea_exploded_gdf = sea_zone.explode(index_parts=True).to_frame(name='geometry')
    sea_filtered_gdf = sea_exploded_gdf[sea_exploded_gdf.area > min_area_m2]
    
    if sea_filtered_gdf.empty:
        logging.error("No 'sea' polygons remaining after filtering by area. Cannot calculate coastal distance.")
        raise ValueError("No 'sea' polygons remaining after filtering by area")
    
    return sea_filtered_gdf.union_all()


def generate_coastal_distance(
    boundary_path: Path = Path("data/processed/boundary.geojson"),
    output_dir: Path = Path("data/evs"),
    bgs_geocoast_shp_path: Path = Path("data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp"),
    buffer_dist_km_for_sea: float = 10.0,
    simplify_tolerance_m: float = 1000.0,
    min_sea_area_km2: float = 1.0,
    distance_calc_resolution_factor: int = 10,
    verbose: bool = False
) -> Path:
    """
    Generates a coastal distance raster layer using BGS GeoCoast data.

    The process involves:
    1. Loading and buffering the study area boundary.
    2. Loading BGS GeoCoast inundation data.
    3. Processing the coastal data to define a 'sea' polygon (dissolve, simplify, buffer, difference).
    4. Calculating distance from a grid to this 'sea' polygon.
    5. Reprojecting, squeezing, and clipping the resulting distance raster.
    6. Saving the final coastal_distance.tif.

    Args:
        boundary_path: Path to the boundary file (e.g., GeoJSON) for the study area.
        output_dir: Directory to save the output coastal_distance.tif file.
        bgs_geocoast_shp_path: Path to the BGS GeoCoast Authority Area Inundation shapefile.
        buffer_dist_km_for_sea: Buffer distance in km to create the initial 'sea' zone from coastline.
        simplify_tolerance_m: Simplification tolerance in meters for coastal geometry processing.
        min_sea_area_km2: Minimum area in km^2 for a sea polygon to be retained after explosion.
        distance_calc_resolution_factor: Factor to multiply base model resolution for distance calculation.
        verbose: Enable verbose logging.

    Returns:
        Path to the generated coastal distance raster file.

    Raises:
        FileNotFoundError: If input files are not found.
        ValueError: If no sea polygons remain after filtering.
    """
    setup_logging(verbose=verbose)
    logging.info("Creating coastal distance dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load boundary and get model grid parameters
    boundary_gdf, model_transform, grid_bounds, spatial_config = load_boundary_and_transform(
        boundary_path
    )
    model_crs = boundary_gdf.crs
    model_resolution = spatial_config["resolution"]

    # Load and process coastal data
    coast_gdf_processed = load_and_process_coast_data(
        bgs_geocoast_shp_path, model_crs, simplify_tolerance_m
    )
    
    # Create sea zone polygon
    sea_polygon = create_sea_zone_polygon(
        coast_gdf_processed, buffer_dist_km_for_sea, simplify_tolerance_m, min_sea_area_km2
    )

    logging.info("Calculating distances to 'sea' zone...")
    distance_calc_resolution = model_resolution * distance_calc_resolution_factor
    coastal_distance_xr = calculate_coastal_distance(
        geom=sea_polygon,
        boundary=boundary_gdf,
        bounds=grid_bounds,
        resolution=distance_calc_resolution,
        name="distance_to_coast",
    )

    logging.info("Reprojecting coastal distance data to model grid...")
    coastal_distance_xr = reproject_data(
        array=coastal_distance_xr,
        crs=model_crs, 
        transform=model_transform, 
        resolution=model_resolution
    )

    logging.info("Squeezing dataset...")
    coastal_distance_xr = squeeze_dataset(ds=coastal_distance_xr)

    logging.info("Masking data to boundary...")
    coastal_distance_xr = coastal_distance_xr.rio.clip(
        [boundary_gdf.union_all()], crs=model_crs, all_touched=True
    )

    output_path = output_dir / "coastal_distance.tif"
    logging.info("Saving coastal distance data to %s...", output_path)
    coastal_distance_xr.rio.to_raster(output_path)

    logging.info("Coastal distance dataset created successfully: %s", output_path)
    return output_path 