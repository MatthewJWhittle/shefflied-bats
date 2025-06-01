"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, and evaluation.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
from itertools import product
import pickle
from dataclasses import dataclass

import typer
import pandas as pd
import geopandas as gpd
import numpy as np
import mlflow
from mlflow.sklearn import log_model
from joblib import Parallel, delayed
import xarray as xr
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator
from elapid.models import MaxentConfig
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sdm.utils.logging_utils import setup_logging
from sdm.data.loaders.vector import (
    load_bat_data,
    load_background_points,
    load_environmental_variables,
)
from sdm.data.processing import annotate_points
from sdm.models.maxent.maxent_model import (
    create_maxent_pipeline,
    get_feature_config,
    ActivityType,
    evaluate_and_train_maxent_model,
    DefaultMaxentConfig,
)
from sdm.models.utils import prepare_occurrence_data
from sdm.occurrence import filter_bats_data

app = typer.Typer()
logger = logging.getLogger(__name__)

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
    cv_models: Optional[List[BaseEstimator]] = None
    cv_scores: Optional[np.ndarray] = None
    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    background_points: gpd.GeoDataFrame,
    background_density: pd.Series,
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
        count_0_input = len(background_points)

        if len(presence) < min_presence:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Only {len(presence)} presence records (minimum {min_presence} required)"
            )
            continue

        if subset is not None:
            n_presence = len(presence)
            presence: gpd.GeoDataFrame = presence.sample(
                n=min(subset, n_presence), random_state=42
            )  # type: ignore

            n_background = len(background_points)
            background_points: gpd.GeoDataFrame = background_points.sample(
                n=min(subset, n_background), random_state=42
            )  # type: ignore
            background_density: pd.Series = background_density.loc[
                background_points.index
            ]  # type: ignore

        occurrence = prepare_occurrence_data(
            presence_gdf=presence,
            background_gdf=background_points,
            background_density=background_density,
            grid_gdf=grid_points,
            input_vars=ev_columns,
            filter_to_grid=True,
            sample_weight_n_neighbors=5,
            subset_background=False,
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
            random_state_kfold=42,
        )

        if final_model is None or cv_models is None or cv_scores is None:
            raise ValueError("Model training failed - received None values")

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        logger.info(f"Model training complete for {latin_name} ({activity_type.value}): CV AUC = {cv_mean:.3f} ± {cv_std:.3f}")

        return TrainingResults(
            latin_name=latin_name,
            activity_type=activity_type.value,
            final_model=final_model,
            cv_models=cv_models,
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
    return successful_results


def prepare_results_dataframe(
    models: List[TrainingResults],
    training_data: List[TrainingData],
) -> pd.DataFrame:
    """Prepare a DataFrame with model results."""
    results = []
    for model, data in zip(models, training_data):
        results.append(
            {
                "identifier": model.identifier(),
                "latin_name": model.latin_name,
                "activity_type": model.activity_type,
                "mean_cv_score": model.cv_scores.mean() if model.cv_scores is not None else None,
                "std_cv_score": model.cv_scores.std() if model.cv_scores is not None else None,
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
    with mlflow.start_run():
        # Log results DataFrame
        mlflow.log_table(data=results_df, artifact_file="model_results.parquet")

        # Log individual models
        for model in models:
            data = training_data[models.index(model)]
            with mlflow.start_run(nested=True):
                ## Log model parameters
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
                    logger.error(f"Model is None for {model.identifier()}")

                ## Log model tags
                mlflow.set_tag("latin_name", model.latin_name)
                genus = model.latin_name.split(" ")[0]
                species = model.latin_name.split(" ")[1]
                species_code = genus[:3] + "_" + species[:3]
                mlflow.set_tag("species_code", species_code)
                mlflow.set_tag("activity_type", model.activity_type)

                ## Log model metrics
                cv_mean = model.cv_scores.mean() if model.cv_scores is not None else None # type: ignore
                cv_std = model.cv_scores.std() if model.cv_scores is not None else None # type: ignore
                if cv_mean is not None:
                    mlflow.log_metric(
                        "mean_cv_score", cv_mean
                    )
                if cv_std is not None:
                    mlflow.log_metric(
                        "std_cv_score", cv_std
                    )

                ## Log model artifact
                if model.final_model is not None:
                    # Create an input example from the first row of the training data
                    # Get the feature names from the model's feature selection step
                    occurrence = data.occurrence
                    X = occurrence.drop(columns=["geometry", "class", "sample_weight"])
                    X = X.iloc[0]
                    input_example = pd.DataFrame(X).T
                    log_model(
                        model.final_model, 
                        f"{model.identifier()}_final_model",
                        input_example=input_example
                    )
                else:
                    logger.error(f"Model is None for {model.identifier()}")


@app.command()
def main(
    bats_file: Path = typer.Argument(
        default="data/processed/bats-tidy.geojson",
        help="Path to bat data file"
    ),
    background_file: Path = typer.Argument(
        default="data/processed/background-points.geojson",
        help="Path to background points file"
    ),
    ev_file: Path = typer.Argument(
        default="data/evs/evs-to-model.tif", 
        help="Path to environmental variables file"),
    grid_points_file: Optional[Path] = typer.Option(
        "data/evs/grid-points.parquet",
        help="Path to grid points file (for training data generation). If not provided, grid points will be extracted from the environmental variables file.",
    ),
    output_dir: Path = typer.Argument(
        default="data/sdm_models",
        help="Output directory for models and results"
    ),
    min_presence: int = typer.Option(
        15, help="Minimum number of presence records required"
    ),
    n_jobs: Optional[int] = typer.Option(None, help="Number of parallel jobs"),
    max_threads_per_model: int = typer.Option(2, help="Maximum threads per model"),
    species: Optional[List[str]] = typer.Option(
        None,
        help="List of species to model. If not provided, all species will be used."
    ),
    activity_types: Optional[List[str]] = typer.Option(
        None,
        help="List of activity types to model. If not provided, all activity types will be used."
    ),
    subset_occurrence: Optional[int] = typer.Option(
        None,
        help="If provided, randomly sample this many presence records for each species-activity type combination."
    ),
) -> None:
    """Run the MaxEnt model training pipeline."""
    setup_logging()
    logger.info("=== Starting SDM Model Training Pipeline ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow
    logger.info("Configuring MLflow tracking...")
    mlflow_db_path = output_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("bat_sdm_models")

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

    latin_names = annotated_bats_gdf.latin_name.unique().tolist()
    activity_types = annotated_bats_gdf.activity_type.unique().tolist()
    logger.info(f"Found {len(latin_names)} species and {len(activity_types)} activity types")

    # Configure model parameters
    logger.info("=== Configuring Model Parameters ===")
    model_config = DefaultMaxentConfig(
        feature_types=["linear", "quadratic", "hinge", "product"],
        beta_multiplier=2.5,
        beta_lqp=1,
        beta_hinge=1,
        beta_threshold=1,
        beta_categorical=1,
        n_hinge_features=10,
        n_threshold_features=10,
        transform="cloglog",
        clamp=True,
        tau=0.5,
        convergence_tolerance=1e-5,
        use_lambdas="best",
        n_lambdas=100,
        class_weights="balanced",
    )

    # Generate training data
    logger.info("=== Generating Training Data ===")
    training_data = generate_training_data(
        bats_ant=annotated_bats_gdf,
        background_points=annotated_background_gdf,
        background_density=background_density,
        grid_points=grid_points,
        latin_names=latin_names,
        activity_types=activity_types,
        ev_columns=ev_columns,
        min_presence=min_presence,
        subset=subset_occurrence,
    )

    # Train models
    logger.info("=== Training Models ===")
    feature_selection = get_feature_config()
    #feature_selection = {str(activity_type): ev_columns for activity_type in [ActivityType.ROOST, ActivityType.IN_FLIGHT]}
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
    results_df["model_path"] = [str(path) for path in results_df["identifier"].map(model_paths)]
    
    # Save results and training data
    save_results(results_df, output_dir)
    save_training_data(training_data, output_dir)

    # Log to MLflow
    logger.info("=== Logging to MLflow ===")
    log_models_to_mlflow(models, training_data, results_df)

    logger.info("=== SDM Model Training Pipeline Complete ===")


if __name__ == "__main__":
    app()
