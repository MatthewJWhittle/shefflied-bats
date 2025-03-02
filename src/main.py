"""
Main function for running the data ingest and modelling pipeline.
"""

from src.ingestion.get_climate_data import main as get_climate_data
from src.ingestion.get_terrain_data import main as get_terrain_data
from src.ingestion.get_vom_data import main as get_vom_data


STUDY_AREA_PATH = "data/processed/boundary.geojson"


def main(
        boundary_path: str,
        output_dir: str,

):
    """
    Main function for running the data ingest and modelling pipeline.
    """
    climate_stats_path = get_climate_data(output_dir, boundary_path)
    terrain_paths = get_terrain_data(output_dir, boundary_path)
    vom_path = get_vom_data(output_dir, boundary_path)
    