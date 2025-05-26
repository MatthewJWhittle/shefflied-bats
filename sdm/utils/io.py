import os
from pyhere import here
import yaml
from pathlib import Path
import json
from typing import Union, Tuple, Dict, Any, List
import geopandas as gpd
from affine import Affine
import pickle
import pandas as pd

def set_project_wd(verbose=True):
    # Navigate to your project directory and create a '.here' file if it doesn't exist
    project_dir = here(".")
    os.chdir(project_dir)

    # Verify that the working directory has been changed
    if verbose:
        print("Current Working Directory:", os.getcwd())
    
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
) -> tuple:
    """
    Load the boundary and construct the model transform.
    Returns: Tuple[gpd.GeoDataFrame, Affine, tuple, Dict]
    """
    spatial_config = load_spatial_config()
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs=spatial_config["crs"]
    )
    from sdm.raster.utils import construct_transform_shift_bounds 
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