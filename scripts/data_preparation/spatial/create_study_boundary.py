import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

from sdm.data.spatial import create_study_boundary
from sdm.utils.logging_utils import setup_logging

def create_study_boundary_wrapper(
    raw_counties_file: Path = Path("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"),
    output_geojson: Path = Path("data/processed/boundary.geojson"),
    target_crs: str = "EPSG:27700",
    simplify_tolerance: Optional[float] = 100.0,
    verbose: bool = False
) -> Path:
    """
    Creates the study area boundary GeoJSON file by loading UK counties, 
    filtering for a defined Yorkshire region, reprojecting, simplifying, 
    and dissolving.

    Args:
        raw_counties_file: Path to the raw UK counties GeoJSON file.
        output_geojson: Path to save the processed study area boundary GeoJSON.
        target_crs: Target CRS for the output boundary.
        simplify_tolerance: Simplification tolerance for TopoJSON (meters). Set to 0 or negative for no simplification.
        verbose: Enable verbose logging.

    Returns:
        Path to the saved boundary file.

    Raises:
        FileNotFoundError: If input files are not found.
        ValueError: If there are errors processing the study area.
    """
    setup_logging(verbose=verbose)
    logging.info(f"Creating study area boundary from: {raw_counties_file}")

    actual_simplify_tolerance = simplify_tolerance if simplify_tolerance and simplify_tolerance > 0 else None

    try:
        study_area_gdf = create_study_boundary(
            counties_filepath=raw_counties_file,
            target_crs=target_crs,
            simplify_tolerance=actual_simplify_tolerance
        )
    except FileNotFoundError as e:
        logging.error(f"Error loading counties data: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error processing study area: {e}")
        raise

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    study_area_gdf.to_file(output_geojson, driver="GeoJSON")
    logging.info(f"Study area boundary saved to: {output_geojson}")
    logging.info(f"Boundary details: {len(study_area_gdf)} feature(s), CRS: {study_area_gdf.crs}")
    
    return output_geojson 