import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd

from sdm.data.spatial import create_boundary
from sdm.utils.logging_utils import setup_logging

def create_study_boundary_wrapper(
    raw_counties_file: Optional[Path] = None,
    output_geojson: Path = Path("data/processed/boundary.geojson"),
    target_crs: str = "EPSG:27700",
    simplify_tolerance: Optional[float] = 100.0,
    live_download: bool = True,
    verbose: bool = False,
    *,
    clip_to_land: bool = True,
    coastline_path: Optional[Path] = None,
) -> Path:
    """
    Creates the study area boundary GeoJSON file.

    By default the dissolved counties are clipped to land (mean high water)
    so sea / Humber cells are not retained (#78).
    """
    setup_logging(verbose=verbose)
    
    study_area_gdf = create_boundary(
        counties_file=raw_counties_file,
        county_names=None,  # Default to Yorkshire
        target_crs=target_crs,
        simplify_tolerance=simplify_tolerance,
        live_download=live_download,
        clip_to_land=clip_to_land,
        coastline_path=coastline_path,
    )

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    study_area_gdf.to_file(output_geojson, driver="GeoJSON")
    logging.info(f"Study area boundary saved to: {output_geojson}")
    
    return output_geojson
