import asyncio
import logging
from pathlib import Path
from typing import Union # Not strictly needed if only Path is used in main args

import typer
import xarray as xr

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.data.terrain import create_terrain_wcs_downloaders

app = typer.Typer()

async def fetch_and_process_terrain_data(
    output_dir: Path,
    boundary_path: Path,
    buffer_distance: float,
    wcs_tile_pixels: tuple[int,int],
    wcs_temp_storage: bool,
    target_resolution_m: int, # Target resolution for WCS GetCoverage call
    max_concurrent_downloads: int
) -> Path:
    """Core async logic for fetching and processing terrain data."""
    
    # Load boundary and spatial configuration
    # load_boundary_and_transform gives boundary_gdf, model_transform, grid_bounds, spatial_config
    # spatial_config contains the model's reference resolution and CRS
    boundary_gdf, model_transform, grid_bounds, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance
    )
    model_crs = boundary_gdf.crs
    model_resolution = spatial_config["resolution"] # This is the project's target resolution for EVs

    # Initialize WCS downloaders for DTM and DSM
    wcs_downloaders = create_terrain_wcs_downloaders(
        tile_pixels=wcs_tile_pixels, 
        use_temp_storage=wcs_temp_storage
    )

    processed_layers = []
    layer_names_in_order = [] # To ensure consistent renaming later

    for layer_name, downloader in wcs_downloaders.items():
        logging.info(f"Downloading {layer_name.upper()} data at {target_resolution_m}m resolution initially...")
        # GetCoverage is called with `target_resolution_m` for the initial download.
        # This might be different from `model_resolution` if downsampling is desired during WCS call.
        raw_layer_data = await downloader.get_coverage(
            bbox=grid_bounds, 
            resolution=float(target_resolution_m), # WCS resolution parameter
            max_concurrent=max_concurrent_downloads
        )
        
        logging.info(f"Reprojecting {layer_name.upper()} data to model grid (CRS: {model_crs}, Resolution: {model_resolution}m)...")
        # Reproject to the consistent model CRS and resolution
        reprojected_layer_data = reproject_data(
            raw_layer_data, 
            crs=model_crs, 
            transform=model_transform, 
            resolution=model_resolution
        )
        # Store the original coverage_id (which is the variable name in reprojected_layer_data)
        # and the desired final name (dtm/dsm)
        # WCSDownloader should name the var in Dataset with its coverage_id
        original_var_name = downloader.coverage_id 
        processed_layers.append(reprojected_layer_data.rename({original_var_name: layer_name}))
        layer_names_in_order.append(layer_name)

    logging.info("Merging DTM and DSM datasets...")
    # Merge datasets. If variable names were already renamed, this should be fine.
    # If they still have original coverage_id names, we rename after merge.
    merged_terrain_data = xr.merge(processed_layers)
    
    # Ensure variables are named dtm, dsm as per layer_names_in_order if not already
    # This might be redundant if the rename during append worked as expected.
    current_vars = list(merged_terrain_data.data_vars.keys())
    expected_vars_map = {wcs_downloaders[lname].coverage_id: lname for lname in layer_names_in_order}
    rename_map = {}
    for cv in current_vars:
        if cv in expected_vars_map: # if current var is a coverage_id
            rename_map[cv] = expected_vars_map[cv]
    if rename_map:
        merged_terrain_data = merged_terrain_data.rename(rename_map)

    logging.info("Squeezing final terrain dataset...")
    merged_terrain_data = squeeze_dataset(merged_terrain_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Filename includes the *model* resolution, as that's the final state of the EV.
    output_filename = f"terrain_dtm_dsm_{int(model_resolution)}m.tif"
    output_path = output_dir / output_filename
    
    logging.info(f"Saving merged terrain data to {output_path}...")
    merged_terrain_data.rio.to_raster(output_path)
    logging.info("Terrain data processing complete.")
    return output_path

@app.command()
def main(
    output_dir: Path = typer.Option(
        "data/evs/terrain", 
        help="Directory to save the output terrain TIFF file.",
        writable=True, resolve_path=True
    ),
    boundary_path: Path = typer.Option(
        "data/processed/boundary.geojson", 
        help="Path to the boundary file for defining processing extent.",
        exists=True, readable=True, resolve_path=True
    ),
    buffer_distance_m: float = typer.Option(7000, help="Buffer distance in meters for the boundary."),
    wcs_tile_width_px: int = typer.Option(1024, help="Width of WCS request tiles in pixels."),
    wcs_tile_height_px: int = typer.Option(1024, help="Height of WCS request tiles in pixels."),
    wcs_temp_storage: bool = typer.Option(True, help="Use temporary disk storage for WCS tiles."),
    wcs_download_resolution_m: int = typer.Option(10, help="Target resolution in meters for initial WCS GetCoverage download."),
    max_concurrent_downloads: int = typer.Option(5, help="Maximum concurrent WCS download requests."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Downloads and processes DTM and DSM terrain data from EA WCS services.
    
    The process involves:
    1. Loading the study boundary and spatial configuration.
    2. Initializing WCS downloaders for DTM and DSM.
    3. Fetching data for DTM and DSM layers via WCS GetCoverage requests (optionally tiled).
    4. Reprojecting each layer to the model's reference CRS and resolution.
    5. Merging the DTM and DSM layers into a single multi-band GeoTIFF.
    6. Saving the final raster.
    """
    setup_logging(verbose=verbose)
    logging.info("Starting terrain data generation workflow...")

    asyncio.run(fetch_and_process_terrain_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        buffer_distance=buffer_distance_m,
        wcs_tile_pixels=(wcs_tile_width_px, wcs_tile_height_px),
        wcs_temp_storage=wcs_temp_storage,
        target_resolution_m=wcs_download_resolution_m,
        max_concurrent_downloads=max_concurrent_downloads
    ))
    logging.info("Terrain data generation workflow finished.")

if __name__ == "__main__":
    app() 