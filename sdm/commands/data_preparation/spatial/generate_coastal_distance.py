import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from sdm.data.coastline_download import (
    coastline_source_is_polyline,
    load_mean_high_water_polylines,
    resolve_coastline_source,
)
from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.raster.processing import calculate_distance_to_geom as calculate_coastal_distance


def load_and_process_coast_data(
    coast_path: Path,
    model_crs: str,
    simplify_tolerance_m: float = 1000.0,
    *,
    use_polyline_source: bool = False,
    clip_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """Load and process coastline geometry for distance calculation."""
    logging.info("Loading coastline data from: %s", coast_path)
    if not coast_path.exists():
        logging.error("Coastline source not found: %s", coast_path)
        raise FileNotFoundError(f"Coastline source not found: {coast_path}")

    if use_polyline_source:
        coast_gdf = load_mean_high_water_polylines(
            coast_path,
            target_crs=model_crs,
            clip_bbox=clip_bbox,
            simplify_tolerance_m=simplify_tolerance_m,
        )
        logging.info(
            "Loaded %d mean high water polyline features (CODE 0071)",
            len(coast_gdf),
        )
        return coast_gdf

    coast_gdf = gpd.read_file(coast_path)
    coast_gdf = coast_gdf.to_crs(model_crs)

    logging.info("Processing coastal polygon geometry (dissolve, simplify)...")
    coast_gdf_processed = coast_gdf.dissolve(dropna=False)
    coast_gdf_processed["geometry"] = coast_gdf_processed.simplify(
        simplify_tolerance_m
    )

    return coast_gdf_processed


def create_sea_zone_polygon(
    coast_gdf_processed: gpd.GeoDataFrame,
    buffer_dist_km_for_sea: float = 10.0,
    simplify_tolerance_m: float = 1000.0,
    min_sea_area_km2: float = 1.0,
) -> BaseGeometry:
    """Create sea zone polygon from processed BGS-style coast polygons."""
    logging.info("Creating 'sea' zone polygon...")
    buffer_m = buffer_dist_km_for_sea * 1000
    sea_zone = (
        coast_gdf_processed.buffer(buffer_m)
        .simplify(simplify_tolerance_m)
        .difference(coast_gdf_processed.geometry.iloc[0])
    )
    min_area_m2 = min_sea_area_km2 * 1_000_000
    sea_exploded_gdf = sea_zone.explode(index_parts=True).to_frame(name="geometry")
    sea_filtered_gdf = sea_exploded_gdf[sea_exploded_gdf.area > min_area_m2]

    if sea_filtered_gdf.empty:
        logging.error(
            "No 'sea' polygons remaining after filtering by area. "
            "Cannot calculate coastal distance."
        )
        raise ValueError("No 'sea' polygons remaining after filtering by area")

    return sea_filtered_gdf.union_all()


def build_distance_target_geometry(
    coast_gdf_processed: gpd.GeoDataFrame,
    *,
    use_polyline_source: bool,
    buffer_dist_km_for_sea: float = 10.0,
    simplify_tolerance_m: float = 1000.0,
    min_sea_area_km2: float = 1.0,
) -> BaseGeometry:
    """Build the target geometry for distance-to-coast calculation."""
    if use_polyline_source:
        logging.info("Using mean high water polylines for distance-to-coast")
        return coast_gdf_processed.geometry.union_all()

    return create_sea_zone_polygon(
        coast_gdf_processed,
        buffer_dist_km_for_sea=buffer_dist_km_for_sea,
        simplify_tolerance_m=simplify_tolerance_m,
        min_sea_area_km2=min_sea_area_km2,
    )


def generate_coastal_distance(
    boundary_path: Path = Path("data/processed/boundary.geojson"),
    output_dir: Path = Path("data/evs"),
    coastline_path: Optional[Path] = None,
    bgs_geocoast_shp_path: Optional[Path] = None,
    buffer_dist_km_for_sea: float = 10.0,
    simplify_tolerance_m: float = 1000.0,
    min_sea_area_km2: float = 1.0,
    distance_calc_resolution_factor: int = 10,
    live_download: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Generates a coastal distance raster layer.

    Default live source: OS Boundary-Line mean high water polylines (CODE 0071)
    via the OS OpenData Downloads API. BGS GeoCoast remains an optional manual
    override for coastal-specific studies.

    The process involves:
    1. Loading and buffering the study area boundary.
    2. Resolving coastline geometry (Boundary-Line MHW, optional BGS GeoCoast, or OSM fallback).
    3. Building a distance target (MHW polylines or BGS sea-zone polygon).
    4. Calculating distance from a grid to that target geometry.
    5. Reprojecting, squeezing, and clipping the resulting distance raster.
    6. Saving the final coastal_distance.tif.

    Args:
        boundary_path: Path to the boundary file (e.g., GeoJSON) for the study area.
        output_dir: Directory to save the output coastal_distance.tif file.
        coastline_path: Optional explicit coastline path (Boundary-Line or BGS GeoCoast).
        bgs_geocoast_shp_path: Deprecated alias for ``coastline_path`` (BGS manual override).
        buffer_dist_km_for_sea: Buffer distance in km for BGS sea-zone polygon creation.
        simplify_tolerance_m: Simplification tolerance in meters for coastal geometry.
        min_sea_area_km2: Minimum sea polygon area (km²) retained for BGS processing.
        distance_calc_resolution_factor: Factor to multiply base model resolution for distance calc.
        live_download: Download OS Boundary-Line when no local coastline file is present.
        verbose: Enable verbose logging.

    Returns:
        Path to the generated coastal distance raster file.

    Raises:
        FileNotFoundError: If input files are not found.
        ValueError: If no sea polygons remain after BGS filtering.
    """
    setup_logging(verbose=verbose)
    logging.info("Creating coastal distance dataset...")
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_path = coastline_path or bgs_geocoast_shp_path

    boundary_gdf, model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path)
    )
    model_crs = boundary_gdf.crs
    model_resolution = spatial_config["resolution"]
    boundary_bounds = tuple(boundary_gdf.total_bounds)

    coast_path, source_kind = resolve_coastline_source(
        explicit_path,
        live_download=live_download,
        fallback_bbox=boundary_bounds,
    )
    use_polyline_source = coastline_source_is_polyline(source_kind, coast_path)

    # Boundary-Line MHW is GB-wide: do not clip to the AOI bbox (inland sites need
    # distant coast geometry). OSM fallback uses an expanded bbox in coastline_download.
    coast_gdf_processed = load_and_process_coast_data(
        coast_path,
        model_crs,
        simplify_tolerance_m,
        use_polyline_source=use_polyline_source,
    )

    distance_target = build_distance_target_geometry(
        coast_gdf_processed,
        use_polyline_source=use_polyline_source,
        buffer_dist_km_for_sea=buffer_dist_km_for_sea,
        simplify_tolerance_m=simplify_tolerance_m,
        min_sea_area_km2=min_sea_area_km2,
    )

    logging.info(
        "Calculating distances to coast (%s source: %s)...",
        "polyline" if use_polyline_source else "sea-zone",
        source_kind,
    )
    distance_calc_resolution = model_resolution * distance_calc_resolution_factor
    coastal_distance_xr = calculate_coastal_distance(
        geom=distance_target,
        boundary_gdf=boundary_gdf,
        grid_bounds=grid_bounds,
        resolution=distance_calc_resolution,
        var_name="distance_to_coast",
    )

    logging.info("Reprojecting coastal distance data to model grid...")
    coastal_distance_xr = reproject_data(
        array=coastal_distance_xr,
        crs=model_crs,
        transform=model_transform,
        resolution=model_resolution,
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
