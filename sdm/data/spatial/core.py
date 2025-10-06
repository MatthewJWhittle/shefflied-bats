"""
Core spatial data processing functionality.
"""

import logging
from pathlib import Path
from typing import Optional, List

import geopandas as gpd

logger = logging.getLogger(__name__)

def create_boundary(
    counties_file: Path,
    county_names: Optional[List[str]] = None,
    target_crs: str = "EPSG:27700",
    simplify_tolerance: Optional[float] = 100.0
) -> gpd.GeoDataFrame:
    """
    Create a boundary from UK counties. If no counties specified, creates Yorkshire boundary.
    
    Args:
        counties_file: Path to the UK counties GeoJSON file
        county_names: List of county names (None for Yorkshire)
        target_crs: Target CRS for the output boundary
        simplify_tolerance: Simplification tolerance in meters (None for no simplification)
        
    Returns:
        GeoDataFrame with the study boundary
    """
    if not counties_file.exists():
        raise FileNotFoundError(f"Counties file not found: {counties_file}")
    
    logger.info(f"Loading counties data from: {counties_file}")
    counties_gdf = gpd.read_file(counties_file)
    
    # Default to Yorkshire if no counties specified
    if county_names is None:
        county_names = [
            "Barnsley", "Doncaster", "Rotherham", "Sheffield",
            "Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield",
            "North Yorkshire", "York",
            "East Riding of Yorkshire", "Kingston upon Hull, City of"
        ]
    
    # Filter to requested counties
    logger.info(f"Filtering to {len(county_names)} counties: {county_names}")
    study_area = counties_gdf[counties_gdf["CTYUA23NM"].isin(county_names)].copy()
    
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
    
    logger.info(f"Created boundary with 1 feature")
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
