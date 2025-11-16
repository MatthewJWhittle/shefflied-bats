import logging
from pathlib import Path
from typing import Optional, Literal, Tuple
import geopandas as gpd

from sdm.utils.logging_utils import setup_logging
from sdm.occurrence.sampling import generate_background_points, TransformMethod, BackgroundMethod

def generate_background_points_wrapper(
    occurrence_data_path: Path,
    boundary_path: Path = Path("data/processed/boundary.geojson"),
    output_dir: Path = Path("data/processed/background_generation"),
    background_points_output_path: Optional[Path] = None,  # If provided, write directly here instead of output_dir
    n_background_points: int = 2000,  # Reduced from 10000 to match notebook
    background_method: Literal["contrast", "percentile", "scale", "fixed", "binary"] = "percentile",  # Changed from "contrast" to "percentile"
    background_value: float = 0.01,  # Changed from 0.3 to 0.01 to match notebook
    grid_resolution: Optional[int] = None,
    transform_method: TransformMethod = TransformMethod.PRESENCE,  # Changed from LOG to PRESENCE to match notebook
    cap_percentile: float = 90.0,
    sigma: float = 1.0,  # Reduced from 1.5 to 1.0 to match notebook
    region_weighting_factor: Optional[float] = None,  # Optional region weighting (not used in notebook)
    verbose: bool = False
) -> Tuple[Path, Path]:
    """
    Core function to generate background points for species distribution modeling.
    Can be called from other scripts or notebooks.

    Args:
        occurrence_data_path: Path to occurrence data (GeoJSON, GPKG or Parquet)
        boundary_path: Path to boundary data (e.g., GeoJSON)
        output_dir: Base directory to save outputs (density raster and background points)
        background_points_output_path: Optional path to write background points directly (if None, uses output_dir)
        n_background_points: Number of background points to generate
        background_method: Method for setting minimum background probability
        background_value: Value for background_method (e.g., contrast ratio, percentile)
        grid_resolution: Resolution of the model grid in CRS units (e.g., meters)
        transform_method: Method to transform occurrence counts for density estimation
        cap_percentile: Percentile for 'cap' transform_method (0-100)
        sigma: Sigma value for Gaussian smoothing of occurrence density
        region_weighting_factor: Factor to weight density by regions (None = no weighting)
        verbose: Enable verbose logging

    Returns:
        Path to the saved background points file, or None if no points were generated
    """
    setup_logging(verbose=verbose)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # Specific subdir for the density raster, can be same as output_dir if preferred
    density_raster_output_dir = output_dir 

    logging.info(f"Starting background point generation. Outputs will be in: {output_dir}")

    # Convert string background_method to BackgroundMethod enum
    bg_method = BackgroundMethod(background_method)
    
    bg_points_path, density_raster_path = generate_background_points(
        occurrence_data_path=occurrence_data_path,
        boundary_path=boundary_path,
        output_dir_for_density_raster=density_raster_output_dir, 
        n_background_points=n_background_points,
        background_method=bg_method,
        background_value=background_value,
        sigma=sigma,
        grid_resolution=grid_resolution if grid_resolution is not None else 100,  # Default to 100m if None
        transform_method=transform_method,
        cap_percentile=cap_percentile,
        region_weighting_factor=region_weighting_factor if region_weighting_factor is not None else 1.0,
    )

    # If a specific output path is provided, write directly there instead of copying
    if background_points_output_path is not None:
        background_points_output_path.parent.mkdir(parents=True, exist_ok=True)
        bg_points_gdf = gpd.read_file(bg_points_path)
        bg_points_gdf.to_file(background_points_output_path, driver="GeoJSON")
        logging.info(f"Background points written directly to {background_points_output_path}")
        return background_points_output_path, density_raster_path

    return bg_points_path, density_raster_path