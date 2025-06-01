import logging
from pathlib import Path

import typer

from sdm.utils.logging_utils import setup_logging
from sdm.raster.terrain import process_dem_to_terrain_attributes, save_terrain_dataset

app = typer.Typer()

@app.command()
def main(
    input_dem_path: Path = typer.Option(
        ..., # Required
        help="Path to the input Digital Elevation Model (DEM) GeoTIFF file.",
        exists=True, readable=True, resolve_path=True
    ),
    output_path: Path = typer.Option(
        ..., # Required
        help="Path to save the output multi-band terrain statistics GeoTIFF.",
        writable=True, resolve_path=True
    ),
    dem_band_index: int = typer.Option(
        0, help="0-indexed band number in the DEM file to process (e.g., 0 for first band)."
    ),
    slope_window_size: int = typer.Option(
        3, help="Window size (pixels) for calculating roughness (std dev of slope)."
    ),
    tpi_window_size: int = typer.Option(
        3, help="Window size (pixels) for calculating Topographic Position Index (TPI)."
    ),
    output_slope_units: str = typer.Option(
        "radians", help="Units for the output slope layer ('degrees', 'percent', or 'radians')."
    ),
    drop_dem_from_output: bool = typer.Option(
        True, help="Whether to drop the original DEM layer from the output terrain statistics file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Calculates various terrain statistics from an input Digital Elevation Model (DEM)
    and saves them as a multi-band GeoTIFF file.

    Terrain attributes calculated include:
    - Slope
    - Aspect (Eastness, Northness)
    - Topographic Wetness Index (TWI)
    - Planform Curvature
    - Roughness (std dev of slope)
    - Topographic Position Index (TPI)
    - Slope-weighted aspect components
    """
    setup_logging(verbose=verbose)
    logging.info(f"Starting terrain statistics generation for DEM: {input_dem_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        terrain_attributes_ds = process_dem_to_terrain_attributes(
            dem_path=input_dem_path,
            dem_band_index=dem_band_index,
            slope_window_size=slope_window_size,
            tpi_window_size=tpi_window_size,
            output_slope_units=output_slope_units
        )
    except FileNotFoundError:
        logging.error(f"Input DEM file not found: {input_dem_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error(f"Error processing DEM to terrain attributes: {e}")
        raise typer.Exit(code=1)

    try:
        saved_path = save_terrain_dataset(
            terrain_ds=terrain_attributes_ds,
            output_path=output_path,
            drop_dem_variable=drop_dem_from_output
        )
        logging.info(f"Terrain statistics successfully saved to: {saved_path}")
    except Exception as e:
        logging.error(f"Error saving terrain statistics dataset: {e}")
        raise typer.Exit(code=1)
    
    logging.info("Terrain statistics generation finished.")


if __name__ == "__main__":
    # Example CLI usage:
    # python scripts/generate_terrain_stats.py --input-dem-path data/evs/terrain/terrain_dtm_dsm_100m.tif \
    #                                       --output-path data/evs/terrain/terrain_derivatives_100m.tif \
    #                                       --dem-band-index 0 \
    #                                       --output-slope-units degrees -v
    app() 