import os
from pyhere import here
import yaml
import logging

from pathlib import Path
import json
from typing import Union, Tuple, Dict, Any, Optional, List
import geopandas as gpd
from affine import Affine
import pickle
import pandas as pd

from sdm.raster.utils import construct_transform_shift_bounds
from pydantic import ValidationError

from sdm.types import ProjectConfig, ModelConfig, VariablesConfig
from sdm.utils.fs import join_uri_path, uri_exists
from sdm.utils.sdm_config import model_config_file_paths, resolve_model_config_dir

import fsspec


def set_project_wd(verbose=True):
    # Navigate to your project directory and create a '.here' file if it doesn't exist
    project_dir = here(".")
    os.chdir(project_dir)

    # Verify that the working directory has been changed
    if verbose:
        logging.info("Current Working Directory:", os.getcwd())
    
    return None

CONFIG_PATH = Path(here(".")) / "config.yml"
MODEL_CONFIG_PATH = Path(here(".")) / "model_config.yml"
VARIABLES_CONFIG_PATH = Path(here(".")) / "variables_config.yml"

def load_config(config_path: Union[str, Path] = CONFIG_PATH) -> Dict:
    """Loads the YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError((
            f"Config file not found at {config_path}. \n "
            "Create a config.yml file in the root of the project."
            ))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_project_config(config_path: Union[str, Path] = CONFIG_PATH) -> ProjectConfig:
    """Load project-level configuration as a Pydantic model."""
    raw = load_config(config_path)
    return ProjectConfig(**raw)

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
    """Load spatial configuration from the main config.yml file."""
    main_config = load_config()
    if "spatial" not in main_config:
        raise KeyError(f"No 'spatial' section found in {CONFIG_PATH}")
    
    spatial_config = main_config["spatial"]
    assert isinstance(spatial_config.get("resolution"), int), "Resolution must be an integer."
    assert "crs" in spatial_config, "CRS missing from spatial config."

    return spatial_config

def load_input_variables() -> list:
    """Deprecated helper; input variables moved to variables_config.yml."""
    raise RuntimeError(
        "Input variables are now stored in variables_config.yml. "
        "Use load_variables_config() instead."
    )

def _model_config_from_section(
    model_section: dict[str, Any],
    config_ref: str,
    *,
    fallback_seed: int | None = None,
) -> ModelConfig:
    section = dict(model_section)
    if "seed" not in section:
        if fallback_seed is not None:
            section["seed"] = fallback_seed
        else:
            raise ValueError(
                f"Missing required 'seed' in the model section of {config_ref}. "
                "Add an integer seed under model: in model_config.yml."
            )
    try:
        return ModelConfig.model_validate(section)
    except ValidationError as exc:
        if any(error.get("loc") == ("seed",) for error in exc.errors()):
            raise ValueError(
                f"Missing required 'seed' in the model section of {config_ref}. "
                "Add an integer seed under model: in model_config.yml."
            ) from exc
        raise


def load_model_config(
    config_path: Union[str, Path] = MODEL_CONFIG_PATH,
    *,
    fallback_seed: int | None = None,
) -> ModelConfig:
    """Load model configuration from model_config.yml as a Pydantic model."""
    config_ref = str(config_path)
    if not uri_exists(config_ref):
        raise FileNotFoundError(
            f"Model config file not found at {config_ref}. "
            "Create a model_config.yml file in the root of the project."
        )

    with fsspec.open(config_ref, "r") as f:
        raw = yaml.safe_load(f) or {}

    model_section = raw.get("model")
    if model_section is None:
        raise KeyError(f"No 'model' section found in {config_path}")

    return _model_config_from_section(
        model_section, config_ref, fallback_seed=fallback_seed
    )


def load_variables_config(
    config_path: Union[str, Path] = VARIABLES_CONFIG_PATH,
) -> VariablesConfig:
    """Load variables configuration for tuning and feature selection."""
    config_ref = str(config_path)
    if not uri_exists(config_ref):
        raise FileNotFoundError(
            f"Variables config file not found at {config_ref}. "
            "Create variables_config.yml or provide a custom path."
        )

    with fsspec.open(config_ref, "r") as f:
        raw = yaml.safe_load(f) or {}
    
    return VariablesConfig.model_validate(raw)


def get_tuning_config_path(
    base_dir: Union[str, Path],
    model_id: str,
    *,
    must_exist: bool = False,
) -> str:
    """Get the path to a species-activity config directory.

    Uses case-insensitive folder matching when reading existing configs.
    Relative ``base_dir`` values resolve from the project root, not
    ``INPUT_DATA_ROOT``.
    """
    return resolve_model_config_dir(base_dir, model_id, must_exist=must_exist)


def load_tuning_variables_config(
    config_path: Union[str, Path],
) -> List[str]:
    """Load variables config from tuning directory (simple list format).
    
    Args:
        config_path: Path to variables_config.yml file
        
    Returns:
        List of variable names
        
    Raises:
        FileNotFoundError: If config file is not found
        KeyError: If 'variables' key is not found
    """
    config_ref = str(config_path)
    if not uri_exists(config_ref):
        raise FileNotFoundError(
            f"Variables config file not found at {config_ref}"
        )

    with fsspec.open(config_ref, "r") as f:
        raw = yaml.safe_load(f) or {}

    variables = raw.get("variables")
    if variables is None:
        raise KeyError(f"No 'variables' key found in {config_ref}")
    
    # Handle both list format and dict format (for backward compatibility)
    if isinstance(variables, list):
        return variables
    elif isinstance(variables, dict):
        # If it's a dict, try to extract from activity_feature_sets
        # This handles old format with activity_feature_sets
        if "activity_feature_sets" in variables:
            # Return first activity's features (or could raise error)
            activity_sets = variables["activity_feature_sets"]
            if activity_sets:
                return list(activity_sets.values())[0]
            raise ValueError(f"Unexpected variables config format in {config_ref}")
    else:
        raise ValueError(f"Variables must be a list, got {type(variables)}")


def load_tuning_model_config(
    config_path: Union[str, Path],
    *,
    fallback_seed: int | None = None,
) -> ModelConfig:
    """Load model config from tuning directory (only maxent section).
    
    If base_model_config is provided, merges the tuning config with it.
    Otherwise, expects a full model config.
    
    Args:
        config_path: Path to model_config.yml file
        base_model_config: Optional base config to merge with
        
    Returns:
        ModelConfig object
    """
    config_ref = str(config_path)
    if not uri_exists(config_ref):
        raise FileNotFoundError(
            f"Model config file not found at {config_ref}"
        )

    with fsspec.open(config_ref, "r") as f:
        raw = yaml.safe_load(f) or {}

    model_section = raw.get("model")
    if model_section is None:
        raise KeyError(f"No 'model' section found in {config_ref}")
    
    return _model_config_from_section(
        model_section, config_ref, fallback_seed=fallback_seed
    )


def load_tuning_configs(
    tuning_dir: Union[str, Path],
    model_id: str,
) -> Tuple[ModelConfig, List[str]]:
    """Load species-activity specific configs from a tuning directory.
    
    Args:
        tuning_dir: Base directory containing tuning results
        model_id: Model identifier (e.g., "Myotis_mystacinus_In_flight")
        
    Returns:
        Tuple of (ModelConfig, List[str]) where List[str] is the selected features
        
    Raises:
        FileNotFoundError: If config files are not found
    """
    config_dir = resolve_model_config_dir(tuning_dir, model_id, must_exist=True)
    model_config_path, variables_config_path = model_config_file_paths(config_dir)

    if not uri_exists(model_config_path):
        raise FileNotFoundError(
            f"Model config not found at {model_config_path}. "
            f"Expected tuning configs in {config_dir}"
        )

    if not uri_exists(variables_config_path):
        raise FileNotFoundError(
            f"Variables config not found at {variables_config_path}. "
            f"Expected tuning configs in {config_dir}"
        )

    base_seed = load_model_config().seed
    model_config = load_tuning_model_config(
        model_config_path, fallback_seed=base_seed
    )
    selected_features = load_tuning_variables_config(variables_config_path)
    
    return model_config, selected_features

def load_boundary_and_transform(
        boundary_path: Union[str, Path],
        buffer_distance: Union[float, int] = 0,
) -> Tuple[gpd.GeoDataFrame, Affine, tuple, Dict]:
    """
    Load the boundary and construct the model transform.

    Default ``buffer_distance`` is 0: ``sdm setup`` writes the modelling extent
    (study_area_buffer ∩ ONS BFC land) into ``paths.boundary``, so EV builders
    should not buffer again.

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

def resolve_trained_model_path(model_path: Union[str, Path]) -> Path:
    """Resolve ``model_path`` from ``model_results.csv`` to the pipeline pickle file.

    Supports:

    - **Package layout** (current training): ``…/{model_id}/model.pkl`` (path may be
      relative or absolute).
    - **Package directory only**: ``…/{model_id}/`` → loads ``model.pkl`` inside it
      (e.g. ``model_package_dir`` column).

    Legacy single-file pickles at ``…/<identifier>.pkl`` still work when that file exists.
    """
    p = Path(model_path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        inner = p / "model.pkl"
        if inner.is_file():
            return inner.resolve()
    p_res = p.resolve()
    if p_res.is_file():
        return p_res
    if p_res.is_dir():
        inner = p_res / "model.pkl"
        if inner.is_file():
            return inner
    raise FileNotFoundError(
        f"Trained model not found at {model_path!s}. "
        "Expected a pickle file, or a package directory containing model.pkl."
    )


def load_pickled_model(model_path_str: Union[str, Path]) -> Any:
    """Load a pickled sklearn / MaxEnt pipeline (handles packaged ``model.pkl`` paths)."""
    model_path = resolve_trained_model_path(model_path_str)
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Error loading model from {model_path}: {e}") from e

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
