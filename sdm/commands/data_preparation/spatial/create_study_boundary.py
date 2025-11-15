import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

from sdm.data.spatial import create_boundary
from sdm.utils.logging_utils import setup_logging

def create_study_boundary_wrapper(
    raw_counties_file: Path = Path("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"),
    output_geojson: Path = Path("data/processed/boundary.geojson"),
    target_crs: str = "EPSG:27700",
    simplify_tolerance: Optional[float] = 100.0,
    verbose: bool = False
) -> Path:
    """
    Creates the study area boundary GeoJSON file.
    """
    setup_logging(verbose=verbose)
    
    study_area_gdf = create_boundary(
        counties_file=raw_counties_file,
        county_names=None,  # Default to Yorkshire
        target_crs=target_crs,
        simplify_tolerance=simplify_tolerance
    )

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    study_area_gdf.to_file(output_geojson, driver="GeoJSON")
    logging.info(f"Study area boundary saved to: {output_geojson}")
    
    return output_geojson
