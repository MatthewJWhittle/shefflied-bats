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
from src.ingestion.geo_utils import reproject_data, squeeze_dataset
from src.ingestion.ogc import WCSDownloader


def summarise_vegetation_height(
    data: xr.DataArray, target_resolution: int, name: str = "vegetation_height"
) -> xr.Dataset:
    """
    Summarise the vegetation height by calculating the mean, min, max, and standard deviation.

    Args:
        data (xr.Dataset): The vegetation height data.
        target_resolution (int): The target resolution to use in coarsening the data.
    """

    current_res = int(abs(data.rio.resolution()[0]))
    scale_factor = target_resolution / current_res
    if scale_factor <= 1:
        raise ValueError(
            f"The target resolution {target_resolution} must be less than the current resolution {current_res}."
        )

    scale_factor = int(scale_factor)
    data_coarse = data.coarsen(x=scale_factor, y=scale_factor, boundary="pad")
    data_mean = data_coarse.mean()
    data_min = data_coarse.min()
    data_max = data_coarse.max()
    data_std = data_coarse.std()

    summary = xr.Dataset(
        {
            f"{name}_mean": data_mean,
            f"{name}_min": data_min,
            f"{name}_max": data_max,
            f"{name}_std": data_std,
        }
    )
    return summary


async def get_data(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
):
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
    resolution = 10
    setup_logging()

    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs="EPSG:27700"
    )
    spatial_config = load_spatial_config()
    boundary.to_crs(spatial_config["crs"], inplace=True)
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), resolution
    )

    wcs_downloader = WCSDownloader(
        "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs",
        "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022",
        request_tile_pixels=(1024, 1024),
        use_temp_storage=True,
    )

    logging.info("Downloading vom data...")
    results = await wcs_downloader.get_coverage(
        bbox=bounds, resolution=10, max_concurrent=100
    )

    # rename the variables to dtm and dsm
    var_name = "vegetation_height"
    name_mapping = {wcs_downloader.coverage_id: var_name}
    results = results.rename(name_mapping)

    result_summary = summarise_vegetation_height(
        results.to_array().squeeze(), 
        target_resolution=spatial_config["resolution"]
    )

    result_summary = reproject_data(
        result_summary, spatial_config["crs"], model_transform, resolution
    )
    # Write intermediate data to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution_int = int(spatial_config["resolution"])
    path = output_dir / f"vom_{resolution_int}m.tif"
    result_summary = squeeze_dataset(result_summary)
    result_summary.rio.to_raster(path)

    logging.info("Data saved to %s", output_dir)

    return path


def main(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
):
    return asyncio.run(get_data(output_dir, boundary_path, buffer_distance))


if __name__ == "__main__":
    main(
        boundary_path="data/processed/boundary.geojson",
    )
