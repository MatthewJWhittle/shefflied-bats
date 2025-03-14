"""
Main function for running the data ingest and modelling pipeline.
"""

import logging
from pathlib import Path

from src.ingestion.get_climate_data import main as get_climate_data
from src.ingestion.get_terrain_data import main as get_terrain_data
from src.ingestion.get_vom_data import main as get_vom_data
from src.ingestion.get_ceh_data import main as get_ceh_data
from src.ingestion.process_os_data import main as process_os_data
from src.processing.terrain_stats import main as process_terrain_stats

from src.utils.config import setup_logging


STUDY_AREA_PATH = "data/processed/boundary.geojson"


def main(boundary_path: str, output_dir: str, debug: bool = False):
    """
    Main function for running the data ingest and modelling pipeline.
    """

    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)

    # ETL data
    # Climate data
    climate_stats_path = get_climate_data(output_dir, boundary_path)

    # EA LiDAR data
    terrain_path = get_terrain_data(output_dir, boundary_path)

    # VOM data
    vom_path = get_vom_data(output_dir, boundary_path)
    # OS Data
    os_cover_path, os_distance_path = process_os_data(boundary_path, output_dir)

    # CEH data
    ceh_path = get_ceh_data(output_dir, boundary_path)

    ## Transform data
    terrain_stats_path = process_terrain_stats(terrain_path, dem_band=1, Path(output_dir) / "terrain_stats.tif")
