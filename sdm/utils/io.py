import os
from pyhere import here
import yaml
import logging

from pathlib import Path
import json
from typing import Union, Tuple, Dict, Any
import geopandas as gpd
from affine import Affine
import pickle
import pandas as pd
from sdm.raster.utils import construct_transform_shift_bounds


def set_project_wd(verbose=True):
    # Navigate to your project directory and create a '.here' file if it doesn't exist
    project_dir = here(".")
    os.chdir(project_dir)

    # Verify that the working directory has been changed
    if verbose:
        logging.info("Current Working Directory:", os.getcwd())
    
    return None

CONFIG_PATH = Path(here(".")) / "config" / "default.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_boundary(
    filepath : Union[str, Path],
    buffer_distance: Union[float, int] = 0,
    target_crs: Union[str, int, dict] = "EPSG:27700",
) -> gpd.GeoDataFrame:
    """
    Loads a boundary from a file, optionally reprojects and applies a buffer.

    Parameters:
    filepath (str): The path to the file containing the boundary data.
    buffer_distance (float): The buffer distance to apply to the boundary geometry (in units of target_crs).
                             If 0, no buffer is applied.
    target_crs (str): The target coordinate reference system (CRS) to reproject the boundary to.

    Returns:
    GeoDataFrame: A GeoDataFrame containing the boundary.
    """
    boundary = gpd.read_file(filepath)
    if boundary.crs != target_crs:
        boundary = boundary.to_crs(target_crs)
    if buffer_distance > 0:
        boundary["geometry"] = boundary.buffer(buffer_distance)
    return boundary

def load_spatial_config() -> Dict:
    # TODO: Refactor to use the main config file (default.yaml)
    # For now, keeps loading from config/spatial.json if it exists
    # or falls back to a section in the main config.
    spatial_json_path = Path(here(".")) / "config" / "spatial.json"
    if spatial_json_path.exists():
        with open(spatial_json_path) as f:
            spatial_config = json.load(f)
    else:
        main_config = load_config()
        if "spatial" not in main_config:
            raise FileNotFoundError(
                f"spatial.json not found and no 'spatial' section in {CONFIG_PATH}"
            )
        spatial_config = main_config["spatial"]
    
    assert isinstance(spatial_config.get("resolution"), int), "Resolution must be an integer."
    assert "crs" in spatial_config, "CRS missing from spatial config."

    return spatial_config

def load_boundary_and_transform(
        boundary_path: Union[str, Path],
        buffer_distance: Union[float, int] = 7000,
) -> Tuple[gpd.GeoDataFrame, Affine, tuple, Dict]:
    """
    Load the boundary and construct the model transform.
    Returns: Tuple[gpd.GeoDataFrame, Affine, tuple, Dict]
    """
    spatial_config = load_spatial_config()
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs=spatial_config["crs"]
    )
    
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), spatial_config["resolution"]
    )
    return boundary, model_transform, bounds, spatial_config

def bbox_filter(bounds:Tuple[float, float, float, float], bounds_vars = ("minx", "miny", "maxx", "maxy")) -> list:
    """Generate a filter list for pd.read_parquet based on bounding box.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy).
        bounds_vars: Tuple of column names in the parquet file for bounds.

    Returns:
        List of filters for pd.read_parquet.
    """
    return [
        (bounds_vars[0], ">=", bounds[0]),
        (bounds_vars[1], ">=", bounds[1]),
        (bounds_vars[2], "<=", bounds[2]),
        (bounds_vars[3], "<=", bounds[3]),
    ]

def load_model_run_summary(summary_csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads the SDM run summary CSV file."""
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        # It's better to raise an error that can be caught by the caller
        raise FileNotFoundError(f"Model run summary not found at {summary_path}")
    try:
        return pd.read_csv(summary_path)
    except Exception as e:
        # Log or print the error before re-raising or raising a custom error
        # logger.error(f"Error reading summary CSV {summary_path}: {e}", exc_info=True)
        raise ValueError(f"Error reading summary CSV {summary_path}: {e}")

def load_pickled_model(model_path_str: Union[str, Path]) -> Any:
    """Loads a pickled model object from a given path string."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        # logger.error(f"Model file not found: {model_path}")
        # Raise an error or return None, depending on desired handling by caller
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        # logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        # Raise a custom error or return None
        raise IOError(f"Error loading model from {model_path}: {e}")

def csv_to_parquet(input_file: Union[str, Path], output_file: Union[str, Path]):
    """Converts a CSV file to a Parquet file."""
    # logger.info(f"Converting {input_file} to Parquet format at {output_file}...") # Add logger if io.py has one
    df = pd.read_csv(input_file)
    try:
        df.to_parquet(output_file)
        # logger.info("Conversion successful.")
    except Exception as e:
        # logger.error(f"Failed to convert CSV to Parquet: {e}", exc_info=True)
        raise
def update_config(config_path: Path, updates: Dict) -> None:
    """Update the config file with new values."""
    import yaml
    import logging
    
    # Load existing config
    config = load_config(config_path)
    
    # Update with new values
    for key, value in updates.items():
        if isinstance(key, str) and '.' in key:
            # Handle nested keys like "paths.boundary"
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = str(value)
        else:
            config[key] = str(value)
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
def validate_boundary_file(filepath: Path) -> None:
    """Validate that a boundary file exists and has the expected structure."""
    if not filepath.exists():
        raise FileNotFoundError(f"Boundary file not found: {filepath}")
    
    # Try to load the file
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        raise ValueError(f"Could not read boundary file {filepath}: {e}")
    
    # Check for geometry column
    if 'geometry' not in gdf.columns:
        raise ValueError(f"Boundary file {filepath} must contain a 'geometry' column")
    
    # Check that geometries are valid
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"Boundary file {filepath} contains invalid geometries")
    
    # Check that we have at least one feature
    if len(gdf) == 0:
        raise ValueError(f"Boundary file {filepath} is empty")
    
    logging.info(f"Boundary file validated: {filepath} ({len(gdf)} features)")

def validate_occurrence_file(filepath: Path) -> None:
    """Validate that an occurrence file exists and has the expected structure."""
    if not filepath.exists():
        raise FileNotFoundError(f"Occurrence file not found: {filepath}")
    
    # Try to load the file
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        raise ValueError(f"Could not read occurrence file {filepath}: {e}")
    
    # Expected columns for bat occurrence data
    expected_columns = ['geometry', 'latin_name', 'activity_type']
    missing_columns = [col for col in expected_columns if col not in gdf.columns]
    
    if missing_columns:
        raise ValueError(f"Occurrence file {filepath} missing required columns: {missing_columns}")
    
    # Check for geometry column
    if 'geometry' not in gdf.columns:
        raise ValueError(f"Occurrence file {filepath} must contain a 'geometry' column")
    
    # Check that geometries are valid
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"Occurrence file {filepath} contains invalid geometries")
    
    # Check that we have at least one occurrence
    if len(gdf) == 0:
        raise ValueError(f"Occurrence file {filepath} is empty")
    
    # Check that we have the expected data
    if gdf['latin_name'].isna().all():
        raise ValueError(f"Occurrence file {filepath} has no valid species names")
    
    if gdf['activity_type'].isna().all():
        raise ValueError(f"Occurrence file {filepath} has no valid activity types")
    
    # Log some statistics
    n_species = gdf['latin_name'].nunique()
    n_activities = gdf['activity_type'].nunique()
    logging.info(f"Occurrence file validated: {filepath} ({len(gdf)} records, {n_species} species, {n_activities} activity types)")
