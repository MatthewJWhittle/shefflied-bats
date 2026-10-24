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
    buffer_distance: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Create a boundary from UK counties. If no counties specified, creates Yorkshire boundary.

    When ``buffer_distance`` > 0, the dissolved counties are buffered then clipped
    to land using the full ONS BFC (coastline-clipped) counties layer in the same
    source file. That keeps inland edge buffers into neighbouring authorities while
    dropping seaward / estuary lobes (#78). Apply the buffer here at setup so EV
    layer builders can load the saved boundary without an extra buffer.

    Args:
        counties_file: Path to the UK counties GeoJSON file (optional; live/manual fallback)
        county_names: List of county names (None for Yorkshire)
        target_crs: Target CRS for the output boundary
        simplify_tolerance: Simplification tolerance in meters (None for no simplification)
        live_download: When True, fetch ONS boundaries from the Open Geography Portal if needed
        buffer_distance: Modelling buffer in metres (0 skips buffer and land clip)

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
    land_gdf = counties_gdf.to_crs(target_crs)

    # Simplify if requested
    if simplify_tolerance and simplify_tolerance > 0:
        logger.info(f"Simplifying geometries with tolerance {simplify_tolerance}m")
        study_area["geometry"] = study_area.geometry.simplify(simplify_tolerance)

    # Merge all geometries into one
    logger.info("Dissolving all geometries into single boundary")
    dissolved = study_area.dissolve()
    dissolved = dissolved[["geometry"]]

    if buffer_distance > 0:
        logger.info(
            "Buffering study boundary by %sm then clipping to ONS BFC land",
            buffer_distance,
        )
        dissolved = dissolved.copy()
        dissolved["geometry"] = dissolved.geometry.buffer(buffer_distance)
        # ONS BFC polygons are already coastline-clipped; their union is land.
        dissolved = gpd.clip(dissolved, land_gdf[["geometry"]])
        dissolved = dissolved[~dissolved.geometry.is_empty].copy()
        if dissolved.empty:
            raise ValueError(
                "Land clip removed the entire buffered boundary; check counties "
                "source coverage for the study area and buffer distance."
            )

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
        simplify_tolerance=simplify_tolerance,
    )
