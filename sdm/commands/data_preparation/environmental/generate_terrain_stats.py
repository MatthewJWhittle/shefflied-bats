import logging
from pathlib import Path

from sdm.utils.logging_utils import setup_logging
from sdm.raster.terrain import process_dem_to_terrain_attributes, save_terrain_dataset

def generate_terrain_stats(
    input_dem_path: Path,
    output_path: Path,
    dem_band_index: int = 0,
    slope_window_size: int = 3,
    tpi_window_size: int = 3,
    output_slope_units: str = "radians",
    drop_dem_from_output: bool = True,
    verbose: bool = False
) -> Path:
    """
    Calculates various terrain statistics from an input Digital Elevation Model (DEM)
    and saves them as a multi-band GeoTIFF file.

    Args:
        input_dem_path: Path to the input Digital Elevation Model (DEM) GeoTIFF file.
        output_path: Path to save the output multi-band terrain statistics GeoTIFF.
        dem_band_index: 0-indexed band number in the DEM file to process (e.g., 0 for first band).
        slope_window_size: Window size (pixels) for calculating roughness (std dev of slope).
        tpi_window_size: Window size (pixels) for calculating Topographic Position Index (TPI).
        output_slope_units: Units for the output slope layer ('degrees', 'percent', or 'radians').
        drop_dem_from_output: Whether to drop the original DEM layer from the output terrain statistics file.
        verbose: Enable verbose logging.

    Returns:
        Path to the saved terrain statistics file.

    Raises:
        FileNotFoundError: If the input DEM file is not found.
        Exception: If there is an error processing the DEM or saving the terrain statistics.
    """
    setup_logging(verbose=verbose)
    logging.info(f"Starting terrain statistics generation for DEM: {input_dem_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    terrain_attributes_ds = process_dem_to_terrain_attributes(
        dem_path=input_dem_path,
        dem_band_index=dem_band_index,
        slope_window_size=slope_window_size,
        tpi_window_size=tpi_window_size,
        output_slope_units=output_slope_units
    )

    saved_path = save_terrain_dataset(
        terrain_ds=terrain_attributes_ds,
        output_path=output_path,
        drop_dem_variable=drop_dem_from_output
    )
    logging.info(f"Terrain statistics successfully saved to: {saved_path}")
    logging.info("Terrain statistics generation finished.")
    
    return saved_path 