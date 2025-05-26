import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated # For newer Typer features if needed, or use typing.Literal
from typing import Literal

from sdm.utils.logging_utils import setup_logging
from sdm.data.processing import generate_background_points

app = typer.Typer()

@app.command()
def main(
    occurrence_data_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to occurrence data (GeoJSON, GPKG or Parquet).", 
            exists=True, 
            readable=True, 
            resolve_path=True
        )
    ],
    boundary_path: Annotated[
        Path, 
        typer.Option(
            help="Path to boundary data (e.g., GeoJSON).", 
            exists=True, 
            readable=True, 
            resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"), # Default from original
    output_dir: Annotated[
        Path, 
        typer.Option(
            help="Base directory to save outputs (density raster and background points).", 
            writable=True, 
            resolve_path=True, 
            file_okay=False, 
            dir_okay=True
        )
    ] = Path("data/processed/background_generation"),
    n_background_points: Annotated[int, typer.Option(help="Number of background points to generate.")] = 10000,
    background_method: Annotated[
        Literal["contrast", "percentile", "scale", "fixed", "binary"],
        typer.Option(case_sensitive=False, help="Method for setting minimum background probability.")
    ] = "contrast",
    background_value: Annotated[float, typer.Option(help="Value for background_method (e.g., contrast ratio, percentile).")] = 0.3,
    grid_resolution: Annotated[Optional[int], typer.Option(help="Resolution of the model grid in CRS units (e.g., meters). Defaults to project setting.")] = None,
    transform_method: Annotated[
        Literal["log", "sqrt", "presence", "cap", "rank"],
        typer.Option(case_sensitive=False, help="Method to transform occurrence counts for density estimation.")
    ] = "log",
    cap_percentile: Annotated[float, typer.Option(help="Percentile for 'cap' transform_method (0-100).")] = 90.0,
    sigma: Annotated[float, typer.Option(help="Sigma value for Gaussian smoothing of occurrence density.")] = 1.5,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False
) -> None:
    """
    Generates background points for species distribution modeling based on 
    density-smoothed occurrence data.
    Saves an intermediate occurrence density raster and the final background points (Parquet).
    """
    setup_logging(verbose=verbose)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # Specific subdir for the density raster, can be same as output_dir if preferred
    density_raster_output_dir = output_dir 

    logging.info(f"Starting background point generation. Outputs will be in: {output_dir}")

    bg_points_gdf = generate_background_points(
        occurrence_data_path=occurrence_data_path,
        boundary_path=boundary_path,
        output_dir_for_density_raster=density_raster_output_dir, 
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        grid_resolution=grid_resolution,
        transform_method=transform_method,
        cap_percentile=cap_percentile,
    )

    if not bg_points_gdf.empty:
        # Construct a descriptive output filename for the points
        bg_value_str = str(background_value).replace('.', 'p')
        output_filename = f"background_points_{n_background_points}_{background_method}_{bg_value_str}_sigma{str(sigma).replace('.', 'p')}.parquet"
        output_path = output_dir / output_filename
        
        bg_points_gdf.to_parquet(output_path)
        logging.info(f"Saved {len(bg_points_gdf)} background points to: {output_path}")
    else:
        logging.warning("No background points were generated. Skipping save.")

if __name__ == "__main__":
    app() 