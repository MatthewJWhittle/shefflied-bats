import logging
from pathlib import Path
from typing import Optional, Literal, Tuple

from sdm.utils.logging_utils import setup_logging
from sdm.occurrence.sampling import generate_background_points, TransformMethod

def generate_background_points_wrapper(
    occurrence_data_path: Path,
    boundary_path: Path = Path("data/processed/boundary.geojson"),
    output_dir: Path = Path("data/processed/background_generation"),
    n_background_points: int = 10000,
    background_method: Literal["contrast", "percentile", "scale", "fixed", "binary"] = "contrast",
    background_value: float = 0.3,
    grid_resolution: Optional[int] = None,
    transform_method: TransformMethod = TransformMethod.LOG,
    cap_percentile: float = 90.0,
    sigma: float = 1.5,
    verbose: bool = False
) -> Tuple[Path, Path]:
    """
    Core function to generate background points for species distribution modeling.
    Can be called from other scripts or notebooks.

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON, GPKG or Parquet)
        boundary_path: Path to boundary data (e.g., GeoJSON)
        output_dir: Base directory to save outputs (density raster and background points)
        n_background_points: Number of background points to generate
        background_method: Method for setting minimum background probability
        background_value: Value for background_method (e.g., contrast ratio, percentile)
        grid_resolution: Resolution of the model grid in CRS units (e.g., meters)
        transform_method: Method to transform occurrence counts for density estimation
        cap_percentile: Percentile for 'cap' transform_method (0-100)
        sigma: Sigma value for Gaussian smoothing of occurrence density
        verbose: Enable verbose logging

    Returns:
        Path to the saved background points file, or None if no points were generated
    """
    setup_logging(verbose=verbose)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # Specific subdir for the density raster, can be same as output_dir if preferred
    density_raster_output_dir = output_dir 

    logging.info(f"Starting background point generation. Outputs will be in: {output_dir}")

    bg_points_path, density_raster_path = generate_background_points(
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

    return bg_points_path, density_raster_path