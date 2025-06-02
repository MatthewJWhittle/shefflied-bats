import asyncio
import logging
from pathlib import Path

import xarray as xr

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.raster.processing import summarise_raster_metrics
from sdm.data import WCSDownloader

async def fetch_and_process_vom_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance: float,
    wcs_tile_pixels: tuple[int,int],
    wcs_temp_storage: bool,
    wcs_download_resolution_m: int,
    max_concurrent_downloads: int,
    summary_target_resolution_m: int
) -> Path:
    """Core async logic for fetching and processing Vegetation Object Model (VOM) data."""
    
    boundary_gdf, model_transform, grid_bounds, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance
    )
    model_crs = boundary_gdf.crs
    # model_resolution is the project's reference EV resolution, from spatial_config["resolution"]
    # summary_target_resolution_m is the resolution to which VOM stats are coarsened.

    # VOM WCS Service Details
    vom_wcs_endpoint = "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs"
    vom_coverage_id = "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
    
    wcs_downloader = WCSDownloader(
        endpoint=vom_wcs_endpoint,
        coverage_id=vom_coverage_id,
        request_tile_pixels=wcs_tile_pixels,
        use_temp_storage=wcs_temp_storage,
    )

    logging.info(f"Downloading VOM data at {wcs_download_resolution_m}m resolution...")
    # raw_vom_data will be an xr.Dataset, with the variable named after coverage_id
    raw_vom_data = await wcs_downloader.get_coverage(
        bbox=grid_bounds, 
        resolution=float(wcs_download_resolution_m),
        max_concurrent=max_concurrent_downloads
    )
    
    # Extract the DataArray (assuming single variable in the Dataset from WCSDownloader)
    if not raw_vom_data.data_vars:
        logging.error("VOM WCS download returned no data variables.")
        raise ValueError("VOM WCS download failed to return data.")
    # The variable name is set to coverage_id by WCSDownloader
    vom_data_array = raw_vom_data[vom_coverage_id]
    
    logging.info(f"Summarising VOM data to {summary_target_resolution_m}m resolution metrics...")
    # summarise_raster_metrics expects a DataArray
    vom_summary_ds = summarise_raster_metrics(
        data_array=vom_data_array, 
        target_resolution=summary_target_resolution_m,
        var_name_prefix="vom_height" # Results in vom_height_mean_XXm etc.
    )

    logging.info(f"Reprojecting VOM summary data to model grid (CRS: {model_crs}, Resolution: {spatial_config['resolution']}m)...")
    # Reproject the summary dataset to the main model grid
    reprojected_vom_summary = reproject_data(
        vom_summary_ds, 
        crs=model_crs, 
        transform=model_transform, 
        resolution=spatial_config["resolution"]
    )

    logging.info("Squeezing final VOM summary dataset...")
    final_vom_ds = squeeze_dataset(reprojected_vom_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Filename includes the summary_target_resolution_m as it reflects the summarized data resolution
    output_filename = f"vom_summary_metrics_{summary_target_resolution_m}m.tif"
    output_path = output_dir / output_filename
    
    logging.info(f"Saving VOM summary data to {output_path}...")
    final_vom_ds.rio.to_raster(output_path)
    logging.info("VOM data processing complete.")
    return output_path

def generate_vom_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance_m: float = 7000,
    wcs_tile_width_px: int = 1024,
    wcs_tile_height_px: int = 1024,
    wcs_temp_storage: bool = True,
    wcs_download_resolution_m: int = 10,
    max_concurrent_downloads: int = 5,
    summary_target_resolution_m: int = 100,
    verbose: bool = False
) -> Path:
    """
    Downloads Vegetation Object Model (VOM) data, summarises it to various metrics
    (mean, min, max, std) at a specified resolution, and saves as a multi-band GeoTIFF.

    Args:
        output_dir: Directory to save the output VOM summary TIFF file.
        boundary_path: Path to the boundary file for defining processing extent.
        buffer_distance_m: Buffer distance in meters for the boundary.
        wcs_tile_width_px: Width of WCS request tiles in pixels.
        wcs_tile_height_px: Height of WCS request tiles in pixels.
        wcs_temp_storage: Use temporary disk storage for WCS tiles.
        wcs_download_resolution_m: Target resolution (meters) for initial VOM WCS GetCoverage download.
        max_concurrent_downloads: Maximum concurrent WCS download requests.
        summary_target_resolution_m: Target resolution (meters) for VOM summary metrics.
        verbose: Enable verbose logging.

    Returns:
        Path to the generated VOM data file.
    """
    setup_logging(verbose=verbose)
    logging.info("Starting VOM data generation workflow...")

    result = asyncio.run(fetch_and_process_vom_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        buffer_distance=buffer_distance_m,
        wcs_tile_pixels=(wcs_tile_width_px, wcs_tile_height_px),
        wcs_temp_storage=wcs_temp_storage,
        wcs_download_resolution_m=wcs_download_resolution_m,
        max_concurrent_downloads=max_concurrent_downloads,
        summary_target_resolution_m=summary_target_resolution_m
    ))
    logging.info("VOM data generation workflow finished.")
    return result 