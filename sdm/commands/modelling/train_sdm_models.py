"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, and evaluation.
Uses a modular approach where each function does one thing and can be easily composed.
"""

import logging
import os
import pickle
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, cast
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
from mlflow.sklearn import log_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import elapid as ela
from shapely import distance
import topojson as tp

from sdm.data.loaders.vector import load_bat_data
from sdm.models.maxent.maxent_model import (
    ActivityType,
    DefaultMaxentConfig,
    create_maxent_pipeline,
    evaluate_and_train_maxent_model,
)
from sdm.occurrence.sampling import (
    generate_background_points_from_data,
    BackgroundMethod,
    TransformMethod,
)
from sdm.raster.io import load_environmental_variables
from sdm.utils.io import (
    load_project_config,
    load_model_config,
    load_variables_config,
    load_boundary,
    load_tuning_configs,
    CONFIG_PATH,
    MODEL_CONFIG_PATH,
)
from sdm.utils.logging_utils import setup_logging
from sdm.types import TrainingData, TrainingResults, ProjectConfig, VariablesConfig, ModelConfig, BackgroundConfig
from sdm.commands.modelling.utils import get_model_id
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=pd.DataFrame)


# ============================================================================
# Modular data preparation functions
# ============================================================================


def convert_evs_to_geodataframe(evs_dataset: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Convert environmental variables xarray Dataset to GeoDataFrame.
    
    Args:
        evs_dataset: xarray Dataset with environmental variables
        
    Returns:
        GeoDataFrame with environmental variables as columns and grid points as rows
    """
    ev_df = evs_dataset.to_dataframe()
    ev_df.reset_index(inplace=True)
    ev_gdf = gpd.GeoDataFrame(ev_df, geometry=gpd.points_from_xy(ev_df.x, ev_df.y))
    ev_gdf.crs = evs_dataset.rio.crs
    ev_gdf = ev_gdf.drop(columns=["spatial_ref"])
    ev_gdf.rename(columns={"x": "grid_x", "y": "grid_y"}, inplace=True)
    
    # Drop rows with missing values
    ev_gdf.dropna(inplace=True)
    
    return ev_gdf


def annotate_points_with_evs(
    points: gpd.GeoDataFrame,
    ev_gdf: gpd.GeoDataFrame,
    tolerance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Annotate points with environmental variables using spatial join.
    
    Args:
        points: GeoDataFrame of points to annotate
        ev_gdf: GeoDataFrame with environmental variables
        tolerance: Maximum distance for nearest neighbor join (defaults to raster resolution)
        
    Returns:
        Annotated GeoDataFrame
    """
    if tolerance is None:
        resolution = ev_gdf.rio.resolution()
        tolerance = abs(resolution[0])
    
    annotated = gpd.sjoin_nearest(
        points, ev_gdf, how="inner", max_distance=tolerance
    )
    annotated.drop(columns=["index_right"], inplace=True)
    
    return annotated


def simplify_boundary(boundary: gpd.GeoDataFrame, tolerance: float = 100) -> gpd.GeoDataFrame:
    """
    Simplify boundary geometry using TopoJSON simplification.
    
    Args:
        boundary: GeoDataFrame with boundary geometry
        tolerance: Simplification tolerance
        
    Returns:
        Simplified boundary GeoDataFrame
    """
    boundary_tp = tp.Topology(boundary, prequantize=True)
    boundary_tp.toposimplify(tolerance, inplace=True)
    simplified = boundary_tp.to_gdf()
    simplified["geometry"] = simplified.geometry.make_valid()
    simplified["geometry"] = simplified.geometry.buffer(0)
    return simplified


def prepare_training_data(
    occurrence_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    evs_to_model: xr.Dataset,
    n_background_points: int,
    background_method: BackgroundMethod,
    background_value: float,
    sigma: float,
    transform_method: TransformMethod,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, List[str]]:
    """
    Prepare study-level training data shared across all species-activity combinations.
    
    This takes in-memory objects (occurrence records, boundary, EV dataset) and returns:
    - presence_with_evs_gdf: all presence points annotated with environmental variables
    - background_with_evs_gdf: all background points annotated with environmental variables
    - ev_columns: list of environmental variable column names
    """
    logger.info("Preparing shared training data (presence and background with EVs)")

    # Derive available feature names from EV dataset
    ev_columns: List[str] = list(evs_to_model.data_vars.keys())

    # Convert EV dataset to GeoDataFrame once
    ev_gdf = convert_evs_to_geodataframe(evs_to_model)

    # Prepare presence data for EV annotation
    split_keys = ["latin_name", "activity_type"]
    occurrence_for_annotation_gdf = occurrence_gdf[split_keys + ["geometry"]].copy()
    occurrence_for_annotation_gdf["class"] = 1

    # Annotate presence points with EVs
    resolution = evs_to_model.rio.resolution()
    tolerance = abs(resolution[0])
    presence_with_evs_gdf = annotate_points_with_evs(
        occurrence_for_annotation_gdf,
        ev_gdf,
        tolerance=tolerance,
    )

    # Generate background points from all occurrence data
    background_points_gdf, _ = generate_background_points_from_data(
        occurrence_data=occurrence_gdf,
        boundary=boundary,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        transform_method=transform_method,
        clip_to_boundary=True,
    )

    # Annotate background points with EVs
    background_with_evs_gdf = annotate_points_with_evs(
        background_points_gdf,
        ev_gdf,
        tolerance=tolerance,
    )
    background_with_evs_gdf["class"] = 0

    logger.info(
        "Shared training data prepared: %d presence points, %d background points",
        len(presence_with_evs_gdf),
        len(background_with_evs_gdf),
    )

    return presence_with_evs_gdf, background_with_evs_gdf, ev_columns


def generate_background_points_for_activity(
    occurrence_data: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    activity_type: str,
    n_background_points: int = 4000,
    background_method: BackgroundMethod = BackgroundMethod.CONTRAST,
    background_value: float = 0.00,
    sigma: float = 6.5,
    transform_method: TransformMethod = TransformMethod.PRESENCE,
    clip_to_boundary: bool = True,
) -> Tuple[gpd.GeoDataFrame, xr.DataArray]:
    """
    Generate background points for a specific activity type.
    
    Args:
        occurrence_data: All occurrence data
        boundary: Study area boundary
        activity_type: Activity type to filter by
        n_background_points: Number of background points to generate
        background_method: Method for background point generation
        background_value: Value for background method
        sigma: Gaussian smoothing sigma
        transform_method: Transform method for density
        clip_to_boundary: Whether to clip points to boundary
        
    Returns:
        Tuple of (background points GeoDataFrame, density array)
    """
    activity_occurrences = occurrence_data[
        occurrence_data.activity_type == activity_type
    ]
    
    background_points, density_array = generate_background_points_from_data(
        occurrence_data=activity_occurrences,
        boundary=boundary,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        transform_method=transform_method,
        clip_to_boundary=clip_to_boundary,
    )
    
    background_points.rename(columns={"presence": "class"}, inplace=True)
    
    return background_points, density_array


# ============================================================================
# Species-specific data processing functions
# ============================================================================


def get_species_presence_data(
    presence: gpd.GeoDataFrame,
    latin_name: str,
    activity_type: str,
) -> gpd.GeoDataFrame:
    """
    Filter presence data for a specific species and activity type.
    
    Args:
        presence: All presence data
        latin_name: Species latin name
        activity_type: Activity type
        
    Returns:
        Filtered presence data
    """
    assert "latin_name" in presence.columns, "latin_name column not found in presence data"
    assert "activity_type" in presence.columns, "activity_type column not found in presence data"
    return presence[
        (presence.activity_type == activity_type) &
        (presence.latin_name == latin_name)
    ]


def drop_duplicate_grid_points(
    gdf: gpd.GeoDataFrame,
    index_cols: List[str] = ["grid_x", "grid_y"],
) -> gpd.GeoDataFrame:
    """
    Drop duplicate points based on grid coordinates.
    
    Args:
        gdf: GeoDataFrame to deduplicate
        index_cols: Columns to use for deduplication
        
    Returns:
        Deduplicated GeoDataFrame
    """
    return gdf.drop_duplicates(subset=index_cols)


def get_distance_matrix(
    a: gpd.GeoSeries,
    b: gpd.GeoSeries,
) -> np.ndarray:
    """
    Calculate pairwise distance matrix between two geometry series.
    
    Args:
        a: First geometry series
        b: Second geometry series
        
    Returns:
        Distance matrix (len(a) x len(b))
    """
    A = a.array.to_numpy()
    B = b.array.to_numpy()
    dist_mat = distance(A[:, None], B[None, :])
    return dist_mat


def filter_background_by_distance(
    background_data: gpd.GeoDataFrame,
    presence_data: gpd.GeoDataFrame,
    d_min: float = 500,
    d_max: float = np.inf,
) -> gpd.GeoDataFrame:
    """
    Filter background points by distance from presence points.
    
    Args:
        background_data: Background points
        presence_data: Presence points
        d_min: Minimum distance from presence (meters)
        d_max: Maximum distance from presence (meters)
        
    Returns:
        Filtered background points
    """
    if len(presence_data) == 0:
        return background_data
    
    p_dist = get_distance_matrix(
        presence_data.geometry,
        background_data.geometry,
    ).min(axis=0)
    
    filtered = background_data[
        (p_dist >= d_min) & (p_dist <= d_max)
    ]
    
    return filtered


def grid_sample_points(
    gdf: gpd.GeoDataFrame,
    stratify_cols: Optional[List[str]] = None,
    grid_size_x: float = 100,
    grid_size_y: float = 100,
) -> gpd.GeoDataFrame:
    """
    Sample one point per grid cell, optionally stratified by columns.
    
    Args:
        gdf: GeoDataFrame to sample
        stratify_cols: Columns to stratify by (one point per cell per combination)
        grid_size_x: Grid cell size in x direction
        grid_size_y: Grid cell size in y direction
        
    Returns:
        Sampled GeoDataFrame
    """
    gdf = gdf.copy()
    centroids = gdf.geometry.centroid
    
    x_bins = (centroids.x // grid_size_x).astype(int)
    y_bins = (centroids.y // grid_size_y).astype(int)
    
    gdf["_x_bin"] = x_bins
    gdf["_y_bin"] = y_bins
    
    if stratify_cols is None:
        stratify_cols = []
    
    sampled = gdf.groupby(stratify_cols + ["_x_bin", "_y_bin"], group_keys=False).apply(
        lambda x: x.sample(1)
    )
    sampled.drop(columns=["_x_bin", "_y_bin"], inplace=True)
    
    return sampled


def apply_sample_weights(
    gdf: gpd.GeoDataFrame,
    n_neighbors: int = 10,
) -> gpd.GeoDataFrame:
    """
    Apply distance-based sample weights to points.
    
    Args:
        gdf: GeoDataFrame with geometry
        n_neighbors: Number of neighbors for distance weighting
        
    Returns:
        GeoDataFrame with sample_weight column added
    """
    gdf = gdf.copy()
    gdf["sample_weight"] = ela.distance_weights(
        gdf.geometry,
        n_neighbors=n_neighbors,
    )
    return gdf


def prepare_species_training_data(
    presence_data: gpd.GeoDataFrame,
    background_data: gpd.GeoDataFrame,
    latin_name: str,
    activity_type: str,
    ev_columns: List[str],
    grid_size_m: float = 500,
    d_min: float = 500,
    d_max: float = np.inf,
    n_max_background: Optional[int] = None,
    sort_density: bool = False,
    sample_weight_n_neighbors: int = 10,
) -> gpd.GeoDataFrame:
    """
    Prepare training data for a single species-activity combination.
    
    This function composes multiple smaller functions to:
    1. Filter presence data for species/activity
    2. Drop duplicates
    3. Filter background by distance
    4. Grid sample both presence and background
    5. Apply sample weights
    6. Combine into final training dataset
    
    Args:
        presence_data: All annotated presence data
        background_data: All annotated background data
        latin_name: Species latin name
        activity_type: Activity type
        ev_columns: List of environmental variable column names
        grid_size_m: Grid cell size for sampling (meters)
        d_min: Minimum distance from presence for background (meters)
        d_max: Maximum distance from presence for background (meters)
        n_max_background: Maximum number of background points (defaults to 10x presence)
        sort_density: Whether to sort background by density before sampling
        sample_weight_n_neighbors: Number of neighbors for sample weighting
        
    Returns:
        Combined training data GeoDataFrame with class, sample_weight, and EV columns
    """
    # Filter presence data
    species_presence = get_species_presence_data(
        presence_data, latin_name, activity_type
    )
    
    if len(species_presence) == 0:
        logger.warning(f"No presence data for {latin_name} - {activity_type}")
        return gpd.GeoDataFrame()
    
    # Drop duplicates
    species_presence = drop_duplicate_grid_points(species_presence)
    
    # Filter background by distance
    species_absence = filter_background_by_distance(
        background_data, species_presence, d_min=d_min, d_max=d_max
    )
    species_absence = drop_duplicate_grid_points(species_absence)
    
    # Grid sample
    species_presence = grid_sample_points(
        species_presence, grid_size_x=grid_size_m, grid_size_y=grid_size_m
    )
    species_absence = grid_sample_points(
        species_absence, grid_size_x=grid_size_m, grid_size_y=grid_size_m
    )
    
    # Limit background points
    if n_max_background is None:
        n_max_background = len(species_presence) * 10
    
    if sort_density and "density" in species_absence.columns:
        species_absence = species_absence.sort_values(
            by="density", ascending=False
        ).head(n_max_background)
    else:
        species_absence = species_absence.sample(
            n=min(len(species_absence), n_max_background), replace=False
        )
    
    # Apply sample weights
    species_presence = apply_sample_weights(
        species_presence, n_neighbors=sample_weight_n_neighbors
    )
    species_absence = apply_sample_weights(
        species_absence, n_neighbors=sample_weight_n_neighbors
    )
    
    # Combine and select columns
    species_data = pd.concat([species_presence, species_absence])
    columns = ["class", "sample_weight", "geometry"] + ev_columns
    species_data = species_data[columns]

    species_data = species_data.dropna()
    
    return species_data


# ============================================================================
# Model training functions
# ============================================================================


def _summarize_cv_scores(
    cv_scores: Optional[np.ndarray],
) -> Tuple[float, float, int, int]:
    """Return (mean, std, n_valid, n_total) for a CV score array, safely handling NaNs."""
    if cv_scores is None or len(cv_scores) == 0:
        return np.nan, np.nan, 0, 0
    
    valid_scores = cv_scores[~np.isnan(cv_scores)]
    n_total = len(cv_scores)
    n_valid = len(valid_scores)
    
    if n_valid == 0:
        return np.nan, np.nan, 0, n_total
    
    return float(valid_scores.mean()), float(valid_scores.std()), n_valid, n_total


def _configure_mlflow_from_config(project_config: ProjectConfig) -> None:
    """Configure MLflow tracking URI and experiment from project configuration."""
    tracking_uri = project_config.mlflow.tracking_uri
    experiment_name = project_config.mlflow.experiment_name
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.debug("MLflow tracking URI: %s", tracking_uri)
    logger.debug("MLflow experiment: %s", experiment_name)


def train_single_model(
    data: TrainingData,
    max_threads_per_model: int,
) -> TrainingResults:
    """Train a single MaxEnt model for a given set of training data."""
    try:
        activity_type = ActivityType(data.activity_type)
        latin_name = data.latin_name
        logger.debug(f"Training model for {latin_name} ({activity_type.value})...")
        
        model_config = data.maxent_config
        model_features = data.model_features
        
        # Verify features list is not empty
        if len(model_features) == 0:
            raise ValueError(
                f"model_features list is empty for {latin_name} - {activity_type}. "
                f"At least one feature must be provided."
            )
        
        # Verify all required features are in the data
        missing_features = list(set(model_features) - set(data.occurrence.columns))
        if missing_features:
            raise ValueError(
                f"Missing features {missing_features} in occurrence data for {latin_name} - {activity_type}. "
                f"Available columns: {list(data.occurrence.columns)}"
            )
        
        # Create model with appropriate thread count
        model = create_maxent_pipeline(
            feature_names=model_features,
            maxent_n_jobs=max_threads_per_model,
            model_config=model_config,
        )
        logger.info(f"Model features: {model_features}")
        final_model, cv_models, cv_scores = evaluate_and_train_maxent_model(
            model=model,
            occurrence_gdf=data.occurrence,
            n_cv_folds=3,
            metric_fn=roc_auc_score,
            feature_columns=model_features,  # Explicitly pass feature columns to avoid using extra columns as features
        )
        
        if final_model is None:
            raise ValueError("Model training failed - final_model is None")
        
        cv_mean, cv_std, n_valid, n_total = _summarize_cv_scores(cv_scores)
        if n_valid > 0:
            logger.debug(
                "✓ %s - %s: CV AUC = %.4f ± %.4f (%d/%d folds valid)",
                latin_name,
                activity_type.value,
                cv_mean,
                cv_std,
                n_valid,
                n_total,
            )
        else:
            logger.warning(
                "✗ %s - %s: No valid CV scores", latin_name, activity_type.value
            )
        
        return TrainingResults(
            latin_name=latin_name,
            activity_type=activity_type.value,
            final_model=final_model,
            cv_models=cv_models if cv_models is not None else None,
            cv_scores=cv_scores,
            success=True,
            error=None,
        )
    except Exception as e:
        logger.error(
            f"Error training model for {data.latin_name} - {data.activity_type}: {e}"
        )
        return TrainingResults(
            latin_name=data.latin_name,
            activity_type=data.activity_type,
            final_model=None,
            cv_models=None,
            cv_scores=None,
            success=False,
            error=str(e),
        )


def train_models_parallel(
    training_data: List[TrainingData],
    max_threads_per_model: int = 2,
    n_jobs: Optional[int] = None,
) -> List[TrainingResults]:
    """Train MaxEnt models in parallel for each set of training data.
    
    Args:
        training_data: List of TrainingData objects, one per species-activity combination.
            Each TrainingData must include maxent_config and model_features.
        max_threads_per_model: Maximum threads per model
        n_jobs: Number of parallel jobs (None = auto)
        
    Returns:
        List of TrainingResults
    """
    # Calculate optimal number of jobs if not specified
    if n_jobs is None:
        total_cpus = os.cpu_count()
        if total_cpus is None:
            total_cpus = 1.0
        # Use 80% of available CPUs by default
        n_jobs = max(1, int(total_cpus * 0.8) // max_threads_per_model)
    
    logger.debug(
        f"Training with {n_jobs} parallel jobs, {max_threads_per_model} threads per model"
    )
    
    # Execute training in parallel using ProcessPoolExecutor
    results: List[TrainingResults] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Create a list of futures
        futures = []
        for data in training_data:
            identifier = data.identifier()
            
            futures.append(
                executor.submit(
                    train_single_model,
                    data,
                    max_threads_per_model,
                )
            )
        
        # Collect results as they complete
        for future in tqdm(futures, total=len(futures), desc="Training models"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel training: {e}")
                # Create a failed result
                results.append(
                    TrainingResults(
                        latin_name="unknown",
                        activity_type="unknown",
                        final_model=None,
                        cv_models=None,
                        cv_scores=None,
                        success=False,
                        error=str(e),
                    )
                )
    
    # Extract successful results
    successful_results = [r for r in results if r.success]
    
    # Report failed models
    failed_models = [
        f"{r.latin_name} - {r.activity_type}: {r.error}"
        for r in results
        if not r.success
    ]
    if failed_models:
        logger.warning("Failed models:")
        for failure in failed_models:
            logger.warning(f"  {failure}")
    
    logger.info(
        f"Successfully trained {len(successful_results)} models out of {len(training_data)} attempts"
    )
    
    # Print summary of model accuracies
    if successful_results:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        for result in successful_results:
            cv_mean, cv_std, n_valid, n_total = _summarize_cv_scores(result.cv_scores)
            if n_valid > 0:
                logger.info(
                    "  %s - %s: AUC = %.4f ± %.4f (%d/%d folds valid)",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                    cv_mean,
                    cv_std,
                    n_valid,
                    n_total,
                )
            else:
                logger.info(
                    "  %s - %s: No valid scores",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                )
        logger.info("=" * 80 + "\n")
    
    return successful_results


def prepare_results_dataframe(
    models: List[TrainingResults],
    training_data: List[TrainingData],
) -> pd.DataFrame:
    """Prepare a DataFrame with model results."""
    results = []
    for model, data in zip(models, training_data):
        mean_cv, std_cv, _, _ = _summarize_cv_scores(model.cv_scores)
        results.append(
            {
                "identifier": model.identifier(),
                "latin_name": model.latin_name,
                "activity_type": model.activity_type,
                "mean_cv_score": mean_cv,
                "std_cv_score": std_cv,
                "n_presence": len(data.occurrence[data.occurrence["class"] == 1]),
                "n_background": len(data.occurrence[data.occurrence["class"] == 0]),
            }
        )
    return pd.DataFrame(results)


def save_training_data(
    training_data: List[TrainingData],
    output_dir: Path,
) -> Path:
    """
    Combine the training data into a single parquet file with an identifier column.
    
    Args:
        training_data: List of TrainingData objects
        output_dir: Path to output directory
        
    Returns:
        Path to the saved training data file
    """
    training_data_path = output_dir / "training_data.parquet"
    
    # Add identifier to each DataFrame and combine
    dfs = []
    for data in training_data:
        df = data.occurrence.copy()
        df["identifier"] = data.identifier()
        dfs.append(df)
    
    training_data_df = pd.concat(dfs, ignore_index=True)
    training_data_df.to_parquet(training_data_path)
    return training_data_path


def save_models(
    models: List[TrainingResults],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save trained models to disk.
    
    Args:
        models: List of TrainingResults
        output_dir: Path to output directory
        
    Returns:
        Dictionary of model identifier to path to saved model
    """
    model_paths: Dict[str, Path] = {}
    for model in models:
        model_path = output_dir / f"{model.identifier()}.pkl"
        model_paths[model.identifier()] = model_path
        # Save final model
        with open(model_path, "wb") as f:
            pickle.dump(model.final_model, f)
    
    return model_paths


def save_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save results DataFrame to disk.
    
    Args:
        results_df: DataFrame containing model results
        output_dir: Path to output directory
        
    Returns:
        None
    """
    results_df.to_csv(output_dir / "model_results.csv", index=False)


def log_models_to_mlflow(
    models: List[TrainingResults],
    training_data: List[TrainingData],
    results_df: pd.DataFrame,
) -> None:
    """
    Log trained models and results to MLflow.
    
    Args:
        models: List of TrainingResults
        results_df: DataFrame containing model results
        
    Returns:
        None
    """
    # Check if a run is already active (e.g., from tuning script)
    active_run = mlflow.active_run()
    
    if active_run is None:
        # Start parent run for this training session
        parent_run_name = f"SDM_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_context = mlflow.start_run(run_name=parent_run_name)
        logger.debug(f"Started MLflow parent run: {parent_run_name}")
    else:
        # Use existing run (e.g., from tuning)
        logger.debug(f"Using existing MLflow run: {active_run.info.run_id}")
        # Create a context manager that does nothing (we're already in a run)
        run_context = nullcontext()
    
    with run_context:
        # Log results DataFrame
        mlflow.log_table(data=results_df, artifact_file="model_results.parquet")
        logger.debug(f"Logging {len(models)} models to MLflow...")
        
        # Log individual models
        for model in models:
            data = training_data[models.index(model)]
            model_identifier = model.identifier()
            with mlflow.start_run(nested=True, run_name=model_identifier):
                logger.debug(f"Logging model: {model_identifier}")
                
                # Log basic model parameters
                mlflow.log_params(
                    {
                        "latin_name": model.latin_name,
                        "activity_type": model.activity_type,
                        "n_presence": len(data.occurrence[data.occurrence["class"] == 1]),
                        "n_background": len(data.occurrence[data.occurrence["class"] == 0]),
                    }
                )
                # log model parameters
                if model.final_model is not None:
                    model_params = model.final_model.get_params()
                    mlflow.log_params(model_params)
                else:
                    logger.error(f"Model is None for {model_identifier}")
                
                ## Log model tags
                mlflow.set_tag("latin_name", model.latin_name)
                genus = model.latin_name.split(" ")[0]
                species = model.latin_name.split(" ")[1]
                species_code = genus[:3] + "_" + species[:3]
                mlflow.set_tag("species_code", species_code)
                mlflow.set_tag("activity_type", model.activity_type)
                
                # Log model metrics
                cv_mean, cv_std, n_valid, _ = _summarize_cv_scores(model.cv_scores)
                if n_valid > 0:
                    mlflow.log_metric("mean_auc", cv_mean)
                    mlflow.log_metric("std_auc", cv_std)
                else:
                    logger.info(
                        "Skipping MLflow metric logging for %s (no valid CV scores)",
                        model_identifier,
                    )
                
                # Log model artifact
                if model.final_model is not None:
                    occurrence = data.occurrence
                    x_sample = occurrence.drop(columns=["geometry", "class", "sample_weight"])
                    x_sample = x_sample.iloc[0]
                    input_example = pd.DataFrame(x_sample).T
                    artifact_path = f"{model_identifier}_final_model"
                    try:
                        model_info = log_model(
                            model.final_model,
                            name=artifact_path,
                            input_example=input_example,
                        )
                        logger.debug(f"✓ Model logged to MLflow: {model_info.model_uri}")
                    except Exception as e:
                        if "UNIQUE constraint" in str(e) or "duplicate" in str(e).lower():
                            logger.warning(
                                f"MLflow metric conflict for {model_identifier}, but model was saved. Error: {e}"
                            )
                        else:
                            logger.error(
                                f"Failed to log model {model_identifier} to MLflow: {e}"
                            )
                            raise
                else:
                    logger.error(f"Model is None for {model_identifier}")
        
        # Log summary information at the end of the parent run
        active_run_after = mlflow.active_run()
        parent_run_id = (
            active_run_after.info.run_id if active_run_after else "N/A"
        )
        logger.debug(
            f"MLflow logging complete - Run ID: {parent_run_id}, {len(models)} models logged"
        )


# ============================================================================
# Main training function
# ============================================================================


def train_sdm_models(
    project_config_path: Path = CONFIG_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    variables_config_path: Optional[Path] = None,
    tuning_dir: Optional[Path] = None,
    bats_file: Optional[Path] = None,
    ev_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_presence: Optional[int] = None,
    n_jobs: Optional[int] = None,
    max_threads_per_model: int = 2,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False,
    # Species-specific processing parameters
    grid_size_m: float = 2000,
    d_min: float = 500,
    d_max: float = np.inf,
    sample_weight_n_neighbors: int = 10,
) -> pd.DataFrame:
    """Run the MaxEnt model training pipeline using the new modular approach.
    
    This function uses a modular approach where:
    - Background points are generated once and shared across all species-activity combinations
    - Environmental variables are converted to GeoDataFrame and joined using spatial joins
    - Each species-activity combination is processed independently
    - Per-species-activity configs are loaded from tuning_dir if provided, with fallback to base configs
    
    Args:
        project_config_path: Path to project-level config (paths, spatial, etc.)
        model_config_path: Path to base model hyperparameter config (used as fallback)
        variables_config_path: Path to base variables config (used as fallback)
        tuning_dir: Optional directory containing per-species-activity tuning configs
            (if provided, configs are loaded from {tuning_dir}/{latin_name}_{activity_type}/)
        bats_file: Path to bat data file
        ev_file: Path to environmental variables file
        output_dir: Output directory for models and results
        min_presence: Minimum number of presence records required
        n_jobs: Number of parallel jobs
        max_threads_per_model: Maximum threads per model
        species: List of species to model
        activity_types: List of activity types to model
        verbose: Enable verbose logging
        grid_size_m: Grid cell size for spatial sampling (meters)
        d_min: Minimum distance from presence for background (meters)
        d_max: Maximum distance from presence for background (meters)
        sample_weight_n_neighbors: Number of neighbors for sample weighting
        
    Note:
        Background point generation parameters are loaded from model_config.yml
        under the 'background' key. See BackgroundConfig for available options.
        
    Returns:
        DataFrame containing model results
        
    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If no valid models can be trained
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("=== Starting SDM Model Training Pipeline (Modular Approach) ===")
    
    # Load configs
    project_config = load_project_config(project_config_path)
    base_model_cfg = load_model_config(model_config_path)
    
    # Get background config from model config (with defaults if not present)
    background_config = base_model_cfg.background or BackgroundConfig()
    # BackgroundConfig has default enum values, but add assertions for type checker
    assert background_config.background_method is not None
    assert background_config.transform_method is not None
    logger.info(
        f"Background point config: n={background_config.n_background_points}, "
        f"method={background_config.background_method.value}, "
        f"value={background_config.background_value}, sigma={background_config.sigma}, "
        f"transform={background_config.transform_method.value}"
    )
    
    # Configure MLflow
    _configure_mlflow_from_config(project_config)
    
    # Load data
    logger.info("Loading input data...")
    
    bats_path = bats_file or Path(project_config.paths.occurence_data)
    ev_path = ev_file or Path(project_config.paths.ev_tiff)
    boundary_path = Path(project_config.paths.boundary)
    models_output_dir = output_dir or Path(project_config.paths.models)
    
    models_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug("Loading bat occurrence data...")
    bats_ant = load_bat_data(bats_path)
    bats_ant = bats_ant[bats_ant.accuracy <= 100]  # Filter by accuracy
    
    logger.debug("Loading boundary...")
    boundary = load_boundary(
        filepath=boundary_path, buffer_distance=0, target_crs=27700
    )
    boundary = simplify_boundary(boundary, tolerance=100)
    
    logger.debug("Loading environmental variables...")
    evs_to_model, _ = load_environmental_variables(ev_path)
    
    # Filter species and activity types if specified
    if species is not None:
        logger.debug(f"Filtering to species: {', '.join(species)}")
        bats_ant = bats_ant[bats_ant.latin_name.isin(species)]
    
    if activity_types is not None:
        logger.debug(f"Filtering to activity types: {', '.join(activity_types)}")
        bats_ant = bats_ant[bats_ant.activity_type.isin(activity_types)]
    
    # Get unique combinations (after any filtering)
    latin_names = cast(List[str], bats_ant.latin_name.unique().tolist())
    activity_types_list = cast(List[str], bats_ant.activity_type.unique().tolist())
    logger.info(
        f"Training models for {len(latin_names)} species × {len(activity_types_list)} activity types = {len(latin_names) * len(activity_types_list)} combinations"
    )
    
    # Prepare shared training data (presence/background annotated with EVs)
    # Unpack background config for prepare_training_data (keeps interface simple for testing)
    assert background_config.background_method is not None
    assert background_config.transform_method is not None
    (
        presence_with_evs_gdf,
        background_with_evs_gdf,
        ev_columns,
    ) = prepare_training_data(
        occurrence_gdf=bats_ant,
        boundary=boundary,
        evs_to_model=evs_to_model,
        n_background_points=background_config.n_background_points,
        background_method=background_config.background_method,
        background_value=background_config.background_value,
        sigma=background_config.sigma,
        transform_method=background_config.transform_method,
    )
    
    # Create default model config from base config (used as fallback)
    default_model_config = DefaultMaxentConfig.from_config(
        config=base_model_cfg
    )
    
    # Generate training data for each species-activity combination
    logger.info("Generating training data for each species-activity combination...")
    training_data: List[TrainingData] = []
    filter_combinations = list(product(latin_names, activity_types_list))
    
    effective_min_presence = (
        min_presence if min_presence is not None else base_model_cfg.sampling.min_presence
    )
    
    for latin_name, activity_type in tqdm(
        filter_combinations, desc="Preparing training data"
    ):
        # Prepare training data (using shared presence/background data)
        species_data = prepare_species_training_data(
            presence_data=presence_with_evs_gdf,
            background_data=background_with_evs_gdf,
            latin_name=latin_name,
            activity_type=activity_type,
            ev_columns=ev_columns,
            grid_size_m=grid_size_m,
            d_min=d_min,
            d_max=d_max,
            sample_weight_n_neighbors=sample_weight_n_neighbors,
        )
        
        # Check if training data was generated
        if len(species_data) == 0:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: No training data generated"
            )
            continue
        
        # Check if "class" column exists (should always exist if data was generated)
        if "class" not in species_data.columns:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Missing 'class' column in training data"
            )
            continue
        
        # Check minimum presence requirement
        n_presence = len(species_data[species_data["class"] == 1])
        if n_presence < effective_min_presence:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Only {n_presence} presence records (minimum {effective_min_presence} required)"
            )
            continue
        
        # Load per-species-activity configs from tuning_dir if provided
        identifier = get_model_id([latin_name, activity_type])

        # this can be optional as we will just use all available features if not provided
        model_features_config : VariablesConfig = VariablesConfig(variables=ev_columns)
        maxent_model_config : DefaultMaxentConfig = default_model_config
        if tuning_dir is not None:
            tuning_dir_path = Path(tuning_dir)
            if tuning_dir_path.exists():
                try:
                    model_config_path = tuning_dir_path / identifier / "model_config.yml"
                    model_config = load_model_config(model_config_path)
                    # Convert ModelConfig to DefaultMaxentConfig
                    maxent_model_config = DefaultMaxentConfig.from_config(
                        config=model_config
                    )
                    logger.debug(f"Loaded model config for {identifier}: {maxent_model_config}")
                except FileNotFoundError:
                    logger.debug(
                        f"No model config found for {identifier} at {model_config_path}. "
                        f"Using default configs"
                    )
                except Exception as e:
                    raise Exception(f"Error loading model config for {identifier}: {e}")
                
                try:
                    variables_config_path = tuning_dir_path / identifier / "variables_config.yml"
                    variables_config = load_variables_config(variables_config_path)
                    # This will raise an error if the variables config contains features that are not in the data
                    model_features_config = variables_config.validate_features(ev_columns)
                    logger.debug(f"Loaded variables config for {identifier}: {model_features_config.variables}")

                except FileNotFoundError:
                    logger.debug(
                        f"No variables config found for {identifier} at {variables_config_path}. "
                        f"Using default configs"
                    )
                except Exception as e:
                    raise Exception(f"Error loading variables config for {identifier}: {e}")
            else:
                raise FileNotFoundError(f"Tuning directory {tuning_dir_path} does not exist")
        else:
            logger.debug(f"No tuning_dir provided, using default configs for {identifier}")
        
        # Create TrainingData with config and features included
        training_data.append(
            TrainingData(
                latin_name=latin_name,
                activity_type=activity_type,
                occurrence=species_data,
                maxent_config=maxent_model_config,
                model_features=model_features_config.variables,
            )
        )
    
    logger.info(
        f"Generated training data for {len(training_data)} species-activity combinations"
    )
    
    # Train models
    logger.info(f"Training {len(training_data)} models...")
    
    models = train_models_parallel(
        training_data=training_data,
        max_threads_per_model=max_threads_per_model,
        n_jobs=n_jobs,
    )
    
    # Prepare and save results
    logger.info("Saving results...")
    results_df = prepare_results_dataframe(models, training_data)
    model_paths = save_models(models, models_output_dir)
    
    # Add model paths to results
    results_df["model_path"] = [
        str(model_paths[identifier]) for identifier in results_df["identifier"]
    ]
    
    # Save results and training data
    save_results(results_df, models_output_dir)
    save_training_data(training_data, models_output_dir)
    
    # Log to MLflow
    logger.info("Logging to MLflow...")
    log_models_to_mlflow(models, training_data, results_df)
    
    logger.info("✓ Training pipeline complete")

    # print a summary of the results (model id, mean cv score, std cv score) in a table format
    logger.info("Summary of results:")
    logger.info(results_df[["identifier", "mean_cv_score", "std_cv_score"]].to_string(index=False))
    return results_df
