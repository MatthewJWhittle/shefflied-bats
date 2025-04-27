"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, and evaluation.
Prediction functionality has been moved to model_inference.py.
"""

import os
from typing import Optional, Union
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import elapid as ela
from pathlib import Path
from itertools import product
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import pickle
import multiprocessing
from joblib import Parallel, delayed
import logging

from pyhere import here
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts, log_artifact


from elapid import MaxentModel

from modelling.maxent_utils import prepare_occurence_data, filter_bats, eval_train_model

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MaxEnt Species Distribution Modelling for bat species."
    )

    parser.add_argument(
        "--ev-path",
        type=str,
        default="data/evs/evs-to-model.tif",
        help="Path to environmental variables raster.",
    )

    parser.add_argument(
        "--bats-path",
        type=str,
        default="data/processed/bats-tidy.geojson",
        help="Path to bat occurrence data.",
    )

    parser.add_argument(
        "--background-path",
        type=str,
        default="data/processed/background-points.geojson",
        help="Path to background points data.",
    )

    parser.add_argument(
        "--grid-points-path",
        type=str,
        default="data/evs/grid-points.parquet",
        help="Path to grid points data.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sdm_models",
        help="Directory for output model files.",
    )

    parser.add_argument(
        "--accuracy-threshold",
        type=int,
        default=100,
        help="Maximum accuracy threshold for bat records (in meters).",
    )

    parser.add_argument(
        "--min-presence",
        type=int,
        default=15,
        help="Minimum number of presence records required.",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="Number of CPUs to use. Default is 80%% of available CPUs.",
    )

    parser.add_argument(
        "--max-threads-per-model",
        type=int,
        default=2,
        help="Maximum number of threads per model. Default is 2.",
    )

    return parser.parse_args()


def load_bat_data(bats_path, accuracy_threshold=100):
    """Load and filter the bat occurrence data."""
    bats = gpd.read_file(bats_path)
    bats = bats[bats.accuracy <= accuracy_threshold]
    logger.info(
        f"Loaded {len(bats)} bat records: {bats.latin_name.value_counts().to_dict()}"
    )
    return bats


def load_background_points(background_path) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Load background points for modelling."""
    background = gpd.read_file(background_path)
    density = background.density
    background = background[["geometry"]]
    return background, density


def load_environmental_variables(ev_path):
    """Load environmental variables for modelling."""
    ev_raster = Path(ev_path)

    evs_to_model = rxr.open_rasterio(
        ev_raster, masked=True, band_as_variable=True
    ).squeeze()
    # rename the variables by their long name
    for var in evs_to_model.data_vars:
        evs_to_model = evs_to_model.rename({var: evs_to_model[var].attrs["long_name"]})

    return evs_to_model, ev_raster


def load_grid_points(grid_points_path):
    """Load the grid points for spatial reference."""
    return gpd.read_parquet(grid_points_path)


def annotate_points(bats, background, ev_raster, ev_columns):
    """Annotate bat and background points with environmental variables."""
    bats_ant = ela.annotate(bats, str(ev_raster), labels=ev_columns)
    background = ela.annotate(background, str(ev_raster), labels=ev_columns)
    return bats_ant, background


def create_maxent_model(n_jobs=1):
    """Create a MaxEnt model pipeline with standard scaling."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "maxent",
                MaxentModel(
                    feature_types=["linear", "quadratic", "hinge", "product"],
                    beta_multiplier=2.5,
                    beta_lqp=-1,
                    beta_hinge=-1,
                    beta_threshold=-1,
                    beta_categorical=-1,
                    n_hinge_features=15,
                    n_threshold_features=10,
                    transform="cloglog",
                    clamp=True,
                    tau=0.5,
                    convergence_tolerance=1e-5,
                    use_lambdas="best",
                    n_lambdas=100,
                    class_weights="balanced",
                    n_cpus=n_jobs,
                ),
            ),
        ]
    )
    return model


def generate_training_data(
    bats_ant,
    background,
    background_density,
    grid_points,
    latin_name,
    activity_type,
    ev_columns,
    min_presence=15,
    subset: Optional[int] = None,
):
    """Generate training data for all valid combinations of species and activity types."""
    training_data = []
    filter_combinations = list(product(latin_name, activity_type))

    for latin_name, activity_type in filter_combinations:
        presence = filter_bats(
            bats_ant, latin_name=latin_name, activity_type=activity_type
        )
        count_1_input = len(presence)
        count_0_input = len(background)

        if len(presence) < min_presence:
            print(
                f"Skipping {latin_name} - {activity_type}: Only {len(presence)} presence records (minimum {min_presence} required)"
            )
            continue

        if subset is not None:
            n_presence = len(presence)
            presence = presence.sample(n=min(subset, n_presence), random_state=42)

            n_background = len(background)
            background = background.sample(n=min(subset, n_background), random_state=42)
            # filter the density by the background index
            background_density = background_density.loc[background.index]

        occurrence = prepare_occurence_data(
            presence,
            background,
            background_density,
            grid_points,
            input_vars=ev_columns,
            filter_to_grid=True,
        )
        count_1_output = len(occurrence[occurrence["class"] == 1])
        count_0_output = len(occurrence[occurrence["class"] == 0])

        print(
            f"Generated training data for {latin_name} - {activity_type}: using {count_1_output}/{count_1_input} presence and {count_0_output}/{count_0_input} background points"
        )

        training_data.append(
            {
                "latin_name": latin_name,
                "activity_type": activity_type,
                "occurrence": occurrence,
            }
        )

    return training_data


def train_models_parallel(training_data, max_threads_per_model=2, n_jobs=None):
    """Train MaxEnt models in parallel for each set of training data.

    Args:
        training_data: List of training data dicts
        max_threads_per_model: Maximum number of threads per model
        n_jobs: Number of parallel jobs (models trained simultaneously)
    """
    # Calculate optimal number of jobs if not specified
    if n_jobs is None:
        total_cpus = os.cpu_count()
        # Use 80% of available CPUs by default
        n_jobs = max(1, int(total_cpus * 0.8) // max_threads_per_model)

    print(
        f"Training with {n_jobs} parallel jobs, {max_threads_per_model} threads per model"
    )

    # Define the function to train a single model
    def train_single_model(data):
        try:
            # Create model with appropriate thread count
            model = create_maxent_model(n_jobs=max_threads_per_model)
            result = eval_train_model(data["occurrence"], model)
            return {
                "success": True,
                "result": result,
                "latin_name": data["latin_name"],
                "activity_type": data["activity_type"],
            }
        except Exception as e:
            print(
                f"Error processing {data['latin_name']} - {data['activity_type']}: {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "latin_name": data["latin_name"],
                "activity_type": data["activity_type"],
            }

    # Execute training in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_model)(data) for data in training_data
    )

    # Extract successful results
    successful_results = [r["result"] for r in results if r["success"]]

    # Report failed models
    failed_models = [
        f"{r['latin_name']} - {r['activity_type']}: {r['error']}"
        for r in results
        if not r["success"]
    ]
    if failed_models:
        print("Failed models:")
        for failure in failed_models:
            print(f"  {failure}")

    print(
        f"Successfully trained {len(successful_results)} models out of {len(training_data)} attempts"
    )
    return successful_results


def get_len(x: Union[list, float, int]) -> int:
    """Get the length of a list or return 1 for non-list inputs."""
    if isinstance(x, list):
        return len(x)
    return 1


def prepare_results_dataframe(results, training_data):
    """Prepare a results dataframe with model information and metrics."""
    # Convert the inputs and outputs to a dataframe
    modelling_df = pd.DataFrame(
        [
            {
                "final_model": final_model,
                "cv_models": cv_models,
                "cv_scores": np.array(cv_scores),
            }
            for final_model, cv_models, cv_scores in results
        ]
    )
    inputs_df = pd.DataFrame(training_data)
    # Combine them
    results_df = pd.concat([inputs_df, modelling_df], axis=1)

    # Mutate some columns
    def count_presence(occurrence):
        return (occurrence["class"] == 1).sum()

    def count_background(occurrence):
        return (occurrence["class"] == 0).sum()

    results_df["n_presence"] = results_df.occurrence.apply(count_presence)
    results_df["n_background"] = results_df.occurrence.apply(count_background)

    results_df["mean_cv_score"] = results_df.cv_scores.apply(np.mean)
    results_df["mean_cv_score"] = results_df["mean_cv_score"].round(3)
    results_df["std_cv_score"] = results_df.cv_scores.apply(np.std)
    results_df["std_cv_score"] = results_df["std_cv_score"].round(3)

    results_df["folds"] = results_df.cv_scores.apply(get_len)

    results_df["activity_type"] = results_df.activity_type.fillna("All")

    return results_df


def save_models(results_df, output_dir, ev_columns):
    """Save models and metadata to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the environmental variable names for later inference
    with open(output_dir / "ev_columns.json", "w") as f:
        json.dump(ev_columns, f)

    # Save each model with its metadata
    for idx, row in results_df.iterrows():
        # Create unique filename for the model
        model_filename = f"{row['latin_name']}_{row['activity_type']}_model.pkl"
        model_path = output_dir / model_filename

        # Save the actual model
        with open(model_path, "wb") as f:
            pickle.dump(row["final_model"], f)

        # Create metadata for this model
        metadata = {
            "latin_name": row["latin_name"],
            "activity_type": row["activity_type"],
            "n_presence": int(row["n_presence"]),
            "n_background": int(row["n_background"]),
            "mean_cv_score": float(row["mean_cv_score"]),
            "std_cv_score": float(row["std_cv_score"]),
            "folds": int(row["folds"]),
            "model_file": model_filename,
        }

        # Save metadata to JSON
        metadata_path = (
            output_dir / f"{row['latin_name']}_{row['activity_type']}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    # Save model index for easy lookup
    model_index = results_df[
        ["latin_name", "activity_type", "mean_cv_score", "n_presence"]
    ].copy()
    model_index["model_file"] = model_index.apply(
        lambda row: f"{row['latin_name']}_{row['activity_type']}_model.pkl", axis=1
    )
    model_index.to_csv(output_dir / "model_index.csv", index=False)

    return model_index


def log_models_to_mlflow(results_df, ev_columns, output_dir):
    """Log models, parameters, and metrics to MLFlow."""

    # Ensure models directory exists
    models_dir = Path(output_dir) / "models"
    models_dir.mkdir(exist_ok=True)

    input_var_json_path = models_dir / "input_variables.json"

    with open(input_var_json_path, "w") as f:
        json.dump(ev_columns, f)

    # Set experiment name
    mlflow.set_experiment("Sheffield Bat Group - SDM - Maxent")

    # Iterate over the results dataframe to log models, parameters and metrics
    for _, row in tqdm(results_df.iterrows()):
        try:
            with mlflow.start_run(
                run_name=f"Model_{row['latin_name']}_{row['activity_type']}"
            ):
                mlflow.set_tag("model", "Maxent")
                # Generate a species code from the first 3 letters of the genus and species
                # This makes it easier to identify the species in mlflow
                genus = row["latin_name"].split(" ")[0]
                species = row["latin_name"].split(" ")[1]
                species_code = genus[:3] + "_" + species[:3]
                mlflow.set_tag("species_code", species_code)

                mlflow.set_tag("latin_name", row["latin_name"])
                mlflow.set_tag("activity_type", row["activity_type"])
                # Log the metrics
                log_metric("mean_cv_score", row["mean_cv_score"])
                log_metric("std_cv_score", row["std_cv_score"])
                # Log the parameters
                log_params(row[["n_presence", "n_background", "folds"]].to_dict())
                # Log model parameters
                try:
                    log_params(row["final_model"].get_params())
                except Exception as e:
                    print(f"Error logging model parameters: {e}")

                # Log the input variables which exceed the param length limit
                log_artifact(input_var_json_path, "input_variables")

                # Log the training data
                with NamedTemporaryFile(suffix=".parquet") as f:
                    occurence_gdf = row["occurrence"]
                    occurence_gdf.to_parquet(f.name)
                    log_artifact(f.name, "training_data")


                # Log the model
                mlflow.sklearn.log_model(row["final_model"], "model")
        except Exception as e:
            print(
                f"Error logging model for {row['latin_name']} - {row['activity_type']}: {e}"
            )
            # Continue with next model even if this one fails
            continue


def save_results(results_df, output_dir):
    """Save results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "results.csv", index=False)

    # Pickle the results dataframe
    results_df.to_pickle(output_dir / "results.pkl")

    # Unnest the occurrence dataframe and save as a parquet file
    def extract_occurrence_df(row):
        row_occurrence = row["occurrence"]
        row_occurrence["latin_name"] = row["latin_name"]
        row_occurrence["activity_type"] = row["activity_type"]
        return row_occurrence

    occurrence_gdf = pd.concat(
        [extract_occurrence_df(row) for _, row in results_df.iterrows()]
    )
    occurrence_gdf.to_parquet(output_dir / "training-occurrence-data.parquet")


def main(
    ev_path: str = "data/evs/evs-to-model.tif",
    bats_path: str = "data/processed/bats-tidy.geojson",
    background_path: str = "data/processed/background-points.geojson",
    grid_points_path: str = "data/evs/grid-points.parquet",
    output_dir: str = "data/sdm_models",
    accuracy_threshold: int = 100,
    min_presence: int = 15,
    cpus: Optional[int] = None,
    max_threads_per_model: int = 2,
    subset: Optional[int] = None,
):
    """Run the MaxEnt model training pipeline."""

    logger.setLevel(logging.INFO)
    # Load data
    print("Loading data...")
    bats = load_bat_data(bats_path, accuracy_threshold)
    background, density = load_background_points(background_path)
    evs_to_model, ev_raster = load_environmental_variables(ev_path)
    grid_points = load_grid_points(grid_points_path)

    # Calculate optimal CPU usage if not provided
    if cpus is None:
        total_cpus = os.cpu_count()
        cpus = max(1, int(total_cpus * 0.8))
        print(f"Automatically using {cpus} CPUs out of {total_cpus} available")
    else:
        print(f"Using {cpus} CPUs as specified")

    # Calculate number of parallel jobs based on threads per model
    n_parallel_jobs = max(1, cpus // max_threads_per_model)

    # Extract unique species and activity types
    latin_name = bats.latin_name.unique().tolist()
    activity_type = bats.activity_type.unique().tolist()
    print(
        f"Training models for {len(latin_name)} species and {len(activity_type)} activity types..."
    )

    # Get environmental variable names
    ev_columns = list(evs_to_model.data_vars.keys())

    # Annotate points with environmental data
    print("Annotating points with environmental data...")
    bats_ant, background = annotate_points(bats, background, ev_raster, ev_columns)

    # Generate training data
    print("Generating training data...")
    training_data = generate_training_data(
        bats_ant,
        background,
        density,
        grid_points,
        latin_name,
        activity_type,
        ev_columns,
        min_presence=min_presence,
        subset=subset,
    )

    print(f"Generated {len(training_data)} valid training datasets")

    # Train models in parallel
    print("Training models...")
    results = train_models_parallel(
        training_data,
        max_threads_per_model=max_threads_per_model,
        n_jobs=n_parallel_jobs,
    )

    # Prepare results dataframe
    print("Preparing results...")
    results_df = prepare_results_dataframe(results, training_data)

    # Save models to disk
    print("Saving models...")
    save_models(results_df, output_dir, ev_columns)

    # Log models to MLFlow
    print("Logging models to MLFlow...")
    log_models_to_mlflow(results_df, ev_columns, output_dir)

    # Save results
    print("Saving results...")
    save_results(results_df, output_dir)

    print("Model training complete!")
    return results_df


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    main(
        ev_path=args.ev_path,
        bats_path=args.bats_path,
        background_path=args.background_path,
        grid_points_path=args.grid_points_path,
        output_dir=args.output_dir,
        accuracy_threshold=args.accuracy_threshold,
        min_presence=args.min_presence,
        cpus=args.cpus,
        max_threads_per_model=args.max_threads_per_model,
    )
