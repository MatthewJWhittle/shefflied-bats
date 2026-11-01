import asyncio
import logging
from pathlib import Path

import xarray as xr

from sdm.data.wcs_tilearray import (
    EA_DSM_COVERAGE_ID,
    EA_DTM_COVERAGE_ID,
    fetch_wcs_dataset,
)
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging

_TERRAIN_LAYERS: tuple[tuple[str, str, str], ...] = (
    (
        "dtm",
        "https://environment.data.gov.uk/spatialdata/"
        "lidar-composite-digital-terrain-model-dtm-1m/wcs",
        EA_DTM_COVERAGE_ID,
    ),
    (
        "dsm",
        "https://environment.data.gov.uk/spatialdata/"
        "lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
        EA_DSM_COVERAGE_ID,
    ),
)


async def fetch_and_process_terrain_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance: float,
    wcs_tile_pixels: tuple[int, int],
    wcs_temp_storage: bool,
    target_resolution_m: int,
    max_concurrent_downloads: int,
) -> Path:
    """Core async logic for fetching and processing terrain data."""
    del wcs_temp_storage  # tilearray manages chunk fetch; kept for CLI compat

    boundary_gdf, model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path, buffer_distance=buffer_distance)
    )
    model_crs = boundary_gdf.crs
    model_resolution = spatial_config["resolution"]

    processed_layers = []
    for layer_name, endpoint, coverage_id in _TERRAIN_LAYERS:
        logging.info(
            "Downloading %s via tilearray at %sm...",
            layer_name.upper(),
            target_resolution_m,
        )
        raw_layer_data = await asyncio.to_thread(
            fetch_wcs_dataset,
            endpoint=endpoint,
            coverage_id=coverage_id,
            bbox=grid_bounds,
            resolution_m=float(target_resolution_m),
            chunk_size=wcs_tile_pixels,
            max_concurrent=max_concurrent_downloads,
            compute=True,
        )

        logging.info(
            "Reprojecting %s to model grid (CRS: %s, Resolution: %sm)...",
            layer_name.upper(),
            model_crs,
            model_resolution,
        )
        reprojected_layer_data = reproject_data(
            raw_layer_data,
            crs=model_crs,
            transform=model_transform,
            resolution=model_resolution,
            bounds=grid_bounds,
        )
        processed_layers.append(
            reprojected_layer_data.rename({coverage_id: layer_name})
        )

    logging.info("Merging DTM and DSM datasets...")
    merged_terrain_data = xr.merge(processed_layers)
    merged_terrain_data = squeeze_dataset(merged_terrain_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dtm_dsm_{int(model_resolution)}m.tif"
    logging.info("Saving merged terrain data to %s...", output_path)
    merged_terrain_data.rio.to_raster(output_path)
    logging.info("Terrain data processing complete.")
    return output_path


def generate_terrain_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance_m: float = 0,
    wcs_tile_width_px: int = 1024,
    wcs_tile_height_px: int = 1024,
    wcs_temp_storage: bool = True,
    wcs_download_resolution_m: int = 10,
    max_concurrent_downloads: int = 10,
    verbose: bool = False,
) -> Path:
    """Download and process DTM/DSM from EA WCS via tilearray.

    1. Load study boundary and spatial configuration.
    2. Fetch DTM and DSM mosaics with tilearray (Dask + AIMD/403 breaker).
    3. Reproject each layer to the model CRS/resolution.
    4. Merge and write a multi-band GeoTIFF.
    """
    setup_logging(verbose=verbose)
    logging.info("Starting terrain data generation workflow...")

    result = asyncio.run(
        fetch_and_process_terrain_data(
            output_dir=output_dir,
            boundary_path=boundary_path,
            buffer_distance=buffer_distance_m,
            wcs_tile_pixels=(wcs_tile_width_px, wcs_tile_height_px),
            wcs_temp_storage=wcs_temp_storage,
            target_resolution_m=wcs_download_resolution_m,
            max_concurrent_downloads=max_concurrent_downloads,
        )
    )
    logging.info("Terrain data generation workflow finished.")
    return result
