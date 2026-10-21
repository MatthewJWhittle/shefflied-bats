"""
Core spatial data processing functionality.
"""

import logging
from pathlib import Path
from typing import Optional, List

import geopandas as gpd

from sdm.data.ons_download import (
    DEFAULT_YORKSHIRE_COUNTY_NAMES,
    county_name_column,
    resolve_counties_file,
)

logger = logging.getLogger(__name__)

def create_boundary(
    counties_file: Optional[Path] = None,
    county_names: Optional[List[str]] = None,
    target_crs: str = "EPSG:27700",
    simplify_tolerance: Optional[float] = 100.0,
    live_download: bool = True,
    *,
    clip_to_land: bool = True,
    coastline_path: Optional[Path] = None,
    coastline_cache_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Create a boundary from UK counties. If no counties specified, creates Yorkshire boundary.

    When ``clip_to_land`` is True (default), the dissolved counties are clipped
    to the landward side of mean high water so estuary / sea remnants are not
    retained in the study boundary (#78).
    
    Args:
        counties_file: Path to the UK counties GeoJSON file (optional; live/manual fallback)
        county_names: List of county names (None for Yorkshire)
        target_crs: Target CRS for the output boundary
        simplify_tolerance: Simplification tolerance in meters (None for no simplification)
        live_download: When True, fetch ONS boundaries from the Open Geography Portal if needed
        clip_to_land: Clip the dissolved boundary to land using MHW
        coastline_path: Optional explicit MHW / coastline path
        coastline_cache_dir: Optional Boundary-Line cache directory
        
    Returns:
        GeoDataFrame with the study boundary
    """
    if county_names is None:
        county_names = list(DEFAULT_YORKSHIRE_COUNTY_NAMES)

    resolved_counties_file = resolve_counties_file(
        counties_file,
        live_download=live_download,
        county_names=county_names,
    )
    
    logger.info(f"Loading counties data from: {resolved_counties_file}")
    counties_gdf = gpd.read_file(resolved_counties_file)
    name_column = county_name_column(counties_gdf)
    
    # Filter to requested counties
    logger.info(f"Filtering to {len(county_names)} counties: {county_names}")
    study_area = counties_gdf[counties_gdf[name_column].isin(county_names)].copy()
    
    if study_area.empty:
        raise ValueError(f"No counties found matching: {county_names}")
    
    # Reproject to target CRS
    logger.info(f"Reprojecting to {target_crs}")
    study_area = study_area.to_crs(target_crs)
    
    # Simplify if requested
    if simplify_tolerance and simplify_tolerance > 0:
        logger.info(f"Simplifying geometries with tolerance {simplify_tolerance}m")
        study_area['geometry'] = study_area.geometry.simplify(simplify_tolerance)
    
    # Merge all geometries into one
    logger.info("Dissolving all geometries into single boundary")
    dissolved = study_area.dissolve()
    dissolved = dissolved[["geometry"]]

    if clip_to_land:
        from sdm.data.land_clip import clip_boundary_to_land, load_mhw_for_boundary

        logger.info("Clipping study boundary to land (mean high water)")
        mhw = load_mhw_for_boundary(
            dissolved,
            coastline_path=coastline_path,
            cache_dir=coastline_cache_dir,
            live_download=live_download,
        )
        dissolved = clip_boundary_to_land(dissolved, mhw, land_gdf=dissolved)
    
    logger.info(f"Created boundary with {len(dissolved)} feature(s)")
    return dissolved


def create_study_boundary(
    counties_filepath: Path,
    target_crs: str,
    simplify_tolerance: Optional[float] = 100.0,
) -> gpd.GeoDataFrame:
    """
    Legacy function for Yorkshire boundary creation.
    Use create_boundary() instead.
    """
    return create_boundary(
        counties_file=counties_filepath,
        county_names=None,  # Yorkshire default
        target_crs=target_crs,
        simplify_tolerance=simplify_tolerance
    )
