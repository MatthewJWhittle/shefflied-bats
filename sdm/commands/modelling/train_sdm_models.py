"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, and evaluation.
"""

import logging
import os
import pickle
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, cast

import geopandas as gpd
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
from elapid.models import MaxentConfig
from mlflow.sklearn import log_model
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator
from tqdm import tqdm

from sdm.data.loaders.vector import load_background_points, load_bat_data
from sdm.data.processing import annotate_points
from sdm.models.maxent.maxent_model import (
    ActivityType,
    DefaultMaxentConfig,
    create_maxent_pipeline,
    evaluate_and_train_maxent_model,
    get_feature_config,
)
from sdm.models.utils import prepare_occurrence_data
from sdm.occurrence import filter_bats_data
from sdm.raster.io import load_environmental_variables
from sdm.utils.io import load_config
from sdm.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=pd.DataFrame)

project_config = load_config()

class SDMModel(BaseModel):
    latin_name: str
    activity_type: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def identifier(self) -> str:
        return f"{self.latin_name}_{self.activity_type}"

class TrainingData(SDMModel):
    occurrence: gpd.GeoDataFrame




class TrainingResults(SDMModel):
    """Results from training a single model."""
    final_model: Optional[BaseEstimator] = None
    cv_models: Optional[List[BaseEstimator]] = None  # List of valid models (None values filtered out)
    cv_scores: Optional[np.ndarray] = None
    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _summarize_cv_scores(
    cv_scores: Optional[np.ndarray],
) -> tuple[float, float, int, int]:
    """Return (mean, std, n_valid, n_total) for a CV score array, safely handling NaNs."""

    if cv_scores is None or len(cv_scores) == 0:
        return np.nan, np.nan, 0, 0

    valid_scores = cv_scores[~np.isnan(cv_scores)]
    n_total = len(cv_scores)
    n_valid = len(valid_scores)

    if n_valid == 0:
        return np.nan, np.nan, 0, n_total

    return float(valid_scores.mean()), float(valid_scores.std()), n_valid, n_total


def _configure_mlflow_from_config() -> None:
    """Configure MLflow tracking URI and experiment from project configuration."""

    tracking_uri = project_config["mlflow"]["tracking_uri"]
    experiment_name = project_config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment: %s", experiment_name)


def extract_grid_points(
    xr_dataset: xr.Dataset,
) -> gpd.GeoDataFrame:
    """
    Extract grid points from a raster array.

    Args:
        array: xarray.DataArray containing the raster data

    Returns:
        GeoDataFrame containing the grid points
    """

    null_cells = xr_dataset.to_array().isnull().any(dim="variable")
    null_cells_df = null_cells.to_dataframe(name="is_null").reset_index()

    valid_cells_df = null_cells_df[~null_cells_df.is_null]

    # Create a GeoDataFrame from the valid cells
    valid_cells_gdf = gpd.GeoDataFrame(
        valid_cells_df,
        geometry=gpd.points_from_xy(valid_cells_df.x, valid_cells_df.y), # type: ignore
        crs=xr_dataset.rio.crs,
        index=valid_cells_df.index,
    )

    return valid_cells_gdf




def generate_training_data(
    bats_ant: gpd.GeoDataFrame,
    background_points_gdf: gpd.GeoDataFrame,
    background_density_series: pd.Series,
    grid_points: gpd.GeoDataFrame,
    latin_names: List[str],
    activity_types: List[str],
    ev_columns: List[str],
    min_presence: int = 15,
    subset: Optional[int] = None,
) -> List[TrainingData]:
    """Generate training data for all valid combinations of species and activity types."""
    training_data = []
    filter_combinations = list(product(latin_names, activity_types))
    logger.info(f"Generating training data for {len(filter_combinations)} species-activity combinations")

    for latin_name, activity_type in tqdm(filter_combinations, desc="Preparing training data"):
        presence = filter_bats_data(
            bats_ant, latin_name=latin_name, activity_type=activity_type
        )
        count_1_input = len(presence)
        count_0_input = len(background_points_gdf)

        if len(presence) < min_presence:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Only {len(presence)} presence records (minimum {min_presence} required)"
            )
            continue

        if subset is not None:
            n_presence = len(presence)
            presence = cast(gpd.GeoDataFrame, presence.sample(
                n=min(subset, n_presence), random_state=42
            ))

            n_background = len(background_points_gdf)
            background_points_gdf = cast(gpd.GeoDataFrame, background_points_gdf.sample(
                n=min(subset, n_background), random_state=42
            ))
            background_density_series = background_density_series.loc[
                background_points_gdf.index
            ]

        occurrence = prepare_occurrence_data(
            presence_gdf=cast(gpd.GeoDataFrame, presence),
            background_gdf=background_points_gdf,
            background_density=background_density_series,
            grid_gdf=grid_points,
            input_vars=ev_columns,
            filter_to_grid=True,
            sample_weight_n_neighbors=5,
            subset_background=True,  # Subset background points proportionally to presence count
            order_by_density_for_subset=True,  # Prioritize high-density background points
        )
        count_1_output = len(occurrence[occurrence["class"] == 1])
        count_0_output = len(occurrence[occurrence["class"] == 0])

        logger.info(
            f"Generated training data for {latin_name} - {activity_type}: using {count_1_output}/{count_1_input} presence and {count_0_output}/{count_0_input} background points"
        )

        training_data.append(
            TrainingData(
                latin_name=latin_name,
                activity_type=activity_type,
                occurrence=occurrence,
            )
        )

    logger.info(f"Successfully generated training data for {len(training_data)} species-activity combinations")
    return training_data



def train_single_model(
    data: TrainingData,
    feature_selection: Dict[str, List[str]],
    max_threads_per_model: int,
    model_config: MaxentConfig = DefaultMaxentConfig(),
) -> TrainingResults:
    """Train a single MaxEnt model for a given set of training data."""
    try:
        activity_type = ActivityType(data.activity_type)
        latin_name = data.latin_name
        logger.info(f"Training model for {latin_name} ({activity_type.value})...")

        model_features = feature_selection[activity_type]
        logger.debug(f"Using features: {model_features}")

        # Create model with appropriate thread count
        model = create_maxent_pipeline(
            feature_names=model_features,
            maxent_n_jobs=max_threads_per_model,
            model_config=model_config,
        )

        logger.info(f"Starting cross-validation for {latin_name} ({activity_type.value})...")
        final_model, cv_models, cv_scores = evaluate_and_train_maxent_model(
            model=model,
            occurrence_gdf=data.occurrence,
            n_cv_folds=3,
            feature_columns=model_features,
        )

        if final_model is None:
            raise ValueError("Model training failed - final_model is None")

        cv_mean, cv_std, n_valid, n_total = _summarize_cv_scores(cv_scores)
        if n_valid > 0:
            logger.info(
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
    feature_selection: Dict[str, List[str]],
    max_threads_per_model: int = 2,
    n_jobs: Optional[int] = None,
    model_config: MaxentConfig = DefaultMaxentConfig(),
) -> List[TrainingResults]:
    """Train MaxEnt models in parallel for each set of training data."""
    # Calculate optimal number of jobs if not specified
    if n_jobs is None:
        total_cpus = os.cpu_count()
        if total_cpus is None:
            total_cpus = 1.0
        # Use 80% of available CPUs by default
        n_jobs = max(1, int(total_cpus * 0.8) // max_threads_per_model)

    logger.info(
        f"Training with {n_jobs} parallel jobs, {max_threads_per_model} threads per model"
    )

    

    # Execute training in parallel using ProcessPoolExecutor
    results: List[TrainingResults] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Create a list of futures
        futures = [
            executor.submit(
                train_single_model,
                data,
                feature_selection,
                max_threads_per_model,
                model_config
            )
            for data in training_data
        ]
        
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
                        error=str(e)
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
        df['identifier'] = data.identifier()
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
    model_paths : Dict[str, Path] = {}
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

    Raises:
        ValueError: If the model training fails - received None values
    """
    # Start parent run for this training session
    parent_run_name = f"SDM_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=parent_run_name):
        logger.info(f"Started MLflow parent run: {parent_run_name}")
        # Log results DataFrame
        mlflow.log_table(data=results_df, artifact_file="model_results.parquet")
        logger.info(f"Logged {len(models)} models to MLflow in nested runs")

        # Log individual models
        for model in models:
            data = training_data[models.index(model)]
            model_identifier = model.identifier()
            with mlflow.start_run(nested=True, run_name=model_identifier):
                logger.info(f"Logging model to MLflow: {model_identifier}")
                
                # Log basic model parameters
                mlflow.log_params(
                    {
                        "latin_name": model.latin_name,
                        "activity_type": model.activity_type,
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
                    mlflow.log_metric("mean_cv_score", cv_mean)
                    mlflow.log_metric("std_cv_score", cv_std)
                else:
                    logger.info(
                        "Skipping MLflow metric logging for %s (no valid CV scores)",
                        model_identifier,
                    )

                # Log model artifact
                if model.final_model is not None:
                    occurrence = data.occurrence
                    X = occurrence.drop(columns=["geometry", "class", "sample_weight"])
                    X = X.iloc[0]
                    input_example = pd.DataFrame(X).T
                    artifact_path = f"{model_identifier}_final_model"
                    try:
                        model_info = log_model(
                            model.final_model, 
                            artifact_path=artifact_path,  # Keep for compatibility
                            input_example=input_example
                        )
                        logger.info(f"✓ Model logged to MLflow: {model_info.model_uri}")
                        logger.info(f"  Artifact path: {artifact_path}")
                        logger.info(f"  Run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}")
                    except Exception as e:
                        if "UNIQUE constraint" in str(e) or "duplicate" in str(e).lower():
                            logger.warning(f"MLflow metric conflict for {model_identifier}, but model was saved. Error: {e}")
                        else:
                            logger.error(f"Failed to log model {model_identifier} to MLflow: {e}")
                            raise
                else:
                    logger.error(f"Model is None for {model_identifier}")
        
        # Log summary information at the end of the parent run
        parent_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "N/A"
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"\n{'='*80}")
        logger.info(f"MLflow Logging Complete")
        logger.info(f"{'='*80}")
        logger.info(f"Parent run: {parent_run_name} (ID: {parent_run_id})")
        logger.info(f"Total models logged: {len(models)}")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"{'='*80}\n")


def train_sdm_models(
    bats_file: Path = Path(project_config["paths"]["occurence_data"]),
    background_file: Path = Path(project_config["paths"]["background_points"]),
    ev_file: Path = Path(project_config["paths"]["ev_tiff"]),
    grid_points_file: Optional[Path] = Path(project_config["paths"]["grid_points"]),
    output_dir: Path = Path(project_config["paths"]["models"]),
    min_presence: int = 15,
    n_jobs: Optional[int] = None,
    max_threads_per_model: int = 2,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    subset_occurrence: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Run the MaxEnt model training pipeline.

    Args:
        bats_file: Path to bat data file
        background_file: Path to background points file
        ev_file: Path to environmental variables file
        grid_points_file: Path to grid points file (for training data generation)
        output_dir: Output directory for models and results
        min_presence: Minimum number of presence records required
        n_jobs: Number of parallel jobs
        max_threads_per_model: Maximum threads per model
        species: List of species to model
        activity_types: List of activity types to model
        subset_occurrence: If provided, randomly sample this many presence records
        verbose: Enable verbose logging

    Returns:
        DataFrame containing model results

    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If no valid models can be trained
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("=== Starting SDM Model Training Pipeline ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow from project config
    logger.info("Configuring MLflow tracking...")
    _configure_mlflow_from_config()

    # Load data
    logger.info("=== Loading Input Data ===")
    logger.info("Loading bat occurrence data...")
    bats_ant = load_bat_data(bats_file)
    
    logger.info("Loading background points...")
    background, background_density = load_background_points(background_file)
    
    logger.info("Loading environmental variables...")
    ev_data, ev_raster_path = load_environmental_variables(ev_file)
    ev_columns = list(ev_data.data_vars.keys())
    logger.info(f"Found {len(ev_columns)} environmental variables")

    # Load grid points
    logger.info("Loading grid points...")
    if grid_points_file is None:
        grid_points = extract_grid_points(ev_data)
    else:
        grid_points = gpd.read_parquet(grid_points_file)

    # Annotate points with environmental variables
    logger.info("=== Annotating Points with Environmental Variables ===")
    annotated_bats_gdf, annotated_background_gdf = annotate_points(
        bats_ant, background, ev_raster_path, ev_columns
    )

    # Filter species and activity types if specified
    if species is not None:
        logger.info(f"Filtering to species: {', '.join(species)}")
        annotated_bats_gdf = annotated_bats_gdf[annotated_bats_gdf.latin_name.isin(species)]
    
    if activity_types is not None:
        logger.info(f"Filtering to activity types: {', '.join(activity_types)}")
        annotated_bats_gdf = annotated_bats_gdf[annotated_bats_gdf.activity_type.isin(activity_types)]

    latin_names = cast(List[str], annotated_bats_gdf.latin_name.unique().tolist())
    activity_types = cast(List[str], annotated_bats_gdf.activity_type.unique().tolist())
    logger.info(f"Found {len(latin_names)} species and {len(activity_types)} activity types")

    # Configure model parameters
    logger.info("=== Configuring Model Parameters ===")
    model_config = DefaultMaxentConfig(
        feature_types=["linear", "hinge"],  # Removed "product" and "quadratic" to reduce overfitting
        beta_multiplier=3.0,  # Increased from 2.5 to 3.0 for stronger regularization
        beta_lqp=1.0,
        beta_hinge=1.0,
        beta_threshold=1.0,
        beta_categorical=1.0,
        n_hinge_features=10,
        n_threshold_features=10,
        transform="cloglog",
        clamp=False,  # Disabled clamping to prevent extrapolation
        tau=0.5,
        convergence_tolerance=1e-5,
        use_lambdas="best",
        n_lambdas=100,
        class_weights="balanced",
    )

    # Generate training data
    logger.info("=== Generating Training Data ===")
    training_data = generate_training_data(
        bats_ant=cast(gpd.GeoDataFrame, annotated_bats_gdf),
        background_points_gdf=cast(gpd.GeoDataFrame, annotated_background_gdf),
        background_density_series=background_density,
        grid_points=grid_points,
        latin_names=latin_names,
        activity_types=activity_types,
        ev_columns=ev_columns,
        min_presence=min_presence,
        subset=subset_occurrence,
    )

    # Train models
    logger.info("=== Training Models ===")
    feature_selection = {str(k): v for k, v in get_feature_config().items()}
    models = train_models_parallel(
        training_data, 
        feature_selection,
        max_threads_per_model=max_threads_per_model, 
        n_jobs=n_jobs,
        model_config=model_config,
    )

    # Prepare and save results
    logger.info("=== Saving Results ===")
    results_df = prepare_results_dataframe(models, training_data)
    model_paths = save_models(models, output_dir)
    
    # Add model paths to results
    results_df["model_path"] = [str(model_paths[identifier]) for identifier in results_df["identifier"]]
    
    # Save results and training data
    save_results(results_df, output_dir)
    save_training_data(training_data, output_dir)

    # Log to MLflow
    logger.info("=== Logging to MLflow ===")
    log_models_to_mlflow(models, training_data, results_df)

    logger.info("=== SDM Model Training Pipeline Complete ===")
    return results_df
