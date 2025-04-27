"""
Main function for running the data ingest and modelling pipeline.
"""

import logging
from pathlib import Path

from data_prep.generate_evs.ingestion.get_climate_data import main as get_climate_data
from data_prep.generate_evs.ingestion.get_terrain_data import main as get_terrain_data
from data_prep.generate_evs.ingestion.get_vom_data import main as get_vom_data
from data_prep.generate_evs.ingestion.get_ceh_data import main as get_ceh_data
from data_prep.generate_evs.ingestion.process_os_data import main as process_os_data
from data_prep.generate_evs.ingestion.get_coastal_distance import main as get_coastal_distance
from data_prep.generate_evs.processing.terrain_stats import main as process_terrain_stats
from data_prep.generate_evs.processing.merge_datasets import main as merge_datasets

from data_prep.utils.config import setup_logging


STUDY_AREA_PATH = "data/processed/boundary.geojson"


def main(boundary_path: str, output_dir: str, debug: bool = False):
    """
    Main function for running the data ingest and modelling pipeline.
    """

    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)

    # ETL data
    # Climate data
    climate_stats_paths = get_climate_data(output_dir, boundary_path)

    # EA LiDAR data
    terrain_path = get_terrain_data(output_dir, boundary_path)

    # VOM data
    vom_path = get_vom_data(output_dir, boundary_path)
    # OS Data
    os_cover_path, os_distance_path = process_os_data(boundary_path, output_dir)

    # CEH data
    ceh_path = get_ceh_data(output_dir, boundary_path)

    ## Transform data
    terrain_stats_path = process_terrain_stats(terrain_path, dem_band=1, output_path=Path(output_dir) / "terrain_stats.tif")

    # Coastal distance
    coastal_distance_path = get_coastal_distance(output_dir, boundary_path)

    dataset_paths = {
        "ceh_landcover" : ceh_path,
        "vom" : vom_path,
        "terrain_stats" : terrain_stats_path,
        "terrain" : terrain_path,
        "os_cover" : os_cover_path,
        "os-distance" : os_distance_path,
        "climate_stats" : climate_stats_paths["climate_stats"],
        "climate_bioclim" : climate_stats_paths["bioclim"],
        "bgs-coast": coastal_distance_path,
        }

    # add the climate stats to the dataset mapping
    dataset_paths.update(
        climate_stats_paths
    )

    # Merge all datasets
    merge_datasets(
    datasets=dataset_paths,
        output_path=Path(output_dir) / "all-evs.tif",
    )
