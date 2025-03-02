"""
Main function for running the data ingest and modelling pipeline.
"""
import logging

from src.ingestion.get_climate_data import main as get_climate_data
from src.ingestion.get_terrain_data import main as get_terrain_data
from src.ingestion.get_vom_data import main as get_vom_data
from src.ingestion.process_os_data import main as process_os_data

from src.utils.config import setup_logging


STUDY_AREA_PATH = "data/processed/boundary.geojson"


def main(
        boundary_path: str,
        output_dir: str,
        debug: bool = False
):
    """
    Main function for running the data ingest and modelling pipeline.
    """

    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)
    
    # ETL data
    # Climate data
    climate_stats_path = get_climate_data(output_dir, boundary_path)
    
    # EA LiDAR data
    terrain_paths = get_terrain_data(output_dir, boundary_path)
    dtm_path = terrain_paths["dtm"]
    dsm_path = terrain_paths["dsm"]

    # VOM data
    vom_path = get_vom_data(output_dir, boundary_path)
    # OS Data
    os_cover_path, os_distance_path = process_os_data(boundary_path, output_dir)