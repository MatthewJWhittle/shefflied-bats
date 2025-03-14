from pathlib import Path
from typing import Union
import logging
import asyncio

import numpy as np
import rioxarray as rxr
import xarray as xr

from src.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)
from src.utils.config import setup_logging
from src.ingestion.geo_utils import reproject_data
from src.ingestion.ogc import WCSDownloader


def init_wcs_downloaders() -> dict[str, WCSDownloader]:
    """Create the WCS downloaders for each layer."""
    tile_size = (1024, 1024)
    specification = {
        "dtm": {
            "endpoint": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
            "coverage_id": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
            "fill_value": np.nan,
        },
        "dsm": {
            "endpoint": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
            "coverage_id": "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m",
            "fill_value": np.nan,
        },
    }
    return {
        layer: WCSDownloader(
            **spec, request_tile_pixels=tile_size, use_temp_storage=True
        )
        for layer, spec in specification.items()
    }


async def get_data(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
) -> dict[str, Path]:
    """
    Main function to process climate data based on a given boundary.
    Parameters:
    boundary_path (Union[str, Path]): Path to the boundary GeoJSON file. Default is "data/processed/boundary.geojson".
    output_dir (Union[str, Path]): Directory where the output data will be saved. Default is "data/evs".
    This function performs the following steps:
    1. Loads the boundary from the specified path.
    2. Loads spatial configuration settings.
    3. Transforms the boundary to the specified coordinate reference system (CRS).
    4. Constructs a model transform based on the boundary's total bounds and spatial resolution.
    5. Retrieves climate data (bioclimatic variables, temperature average, precipitation, and wind) for the boundary.
    6. Reprojects all datasets to the specified CRS and transform.
    7. Assigns variable names to the datasets.
    8. Writes the processed data to the specified output directory.
    9. Calculates and saves climate statistics based on the processed data.
    """
    setup_logging()

    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs="EPSG:27700"
    )
    spatial_config = load_spatial_config()
    boundary.to_crs(spatial_config["crs"], inplace=True)
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), spatial_config["resolution"]
    )

    wcs_downloaders = init_wcs_downloaders()

    results = []
    for layer, wcs_downloader in wcs_downloaders.items():
        logging.info("Downloading %s data...", layer)
        data = await wcs_downloader.get_coverage(
            bbox=bounds, resolution=10, max_concurrent=100
        )
        data = reproject_data(
            data, spatial_config["crs"], model_transform, spatial_config["resolution"]
        )
        results.append(data)

    results = xr.merge(results)
    # rename the variables to dtm and dsm
    name_mapping = {
        downloader.coverage_id: name for name, downloader in wcs_downloaders.items()
    }
    results = results.rename(name_mapping)

    # Write intermediate data to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name in results.data_vars:
        resolution_int = int(round(spatial_config["resolution"]))
        path = output_dir / f"{name}_{resolution_int}m.tif"
        results[name].rio.to_raster(path)
        paths[name] = path

    logging.info("Data saved to %s", output_dir)
    return paths


def main(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
) -> dict[str, Path]:
    return asyncio.run(get_data(output_dir, boundary_path, buffer_distance))


if __name__ == "__main__":
    main(
        boundary_path="data/processed/boundary.geojson",
    )
