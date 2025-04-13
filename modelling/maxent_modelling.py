"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, evaluation,
and prediction generation.
"""

import os
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

from pyhere import here
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from elapid import MaxentModel

from sdm.maxent import (
    prepare_occurence_data,
    filter_bats, 
    eval_train_model
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MaxEnt Species Distribution Modelling for bat species."
    )
    
    parser.add_argument(
        "--ev-path", 
        type=str,
        default="data/evs/evs-to-model.tif",
        help="Path to environmental variables raster."
    )
    
    parser.add_argument(
        "--bats-path", 
        type=str,
        default="data/processed/bats-tidy.geojson",
        help="Path to bat occurrence data."
    )
    
    parser.add_argument(
        "--background-path", 
        type=str,
        default="data/processed/background-points.geojson",
        help="Path to background points data."
    )
    
    parser.add_argument(
        "--grid-points-path", 
        type=str,
        default="data/evs/grid-points.parquet",
        help="Path to grid points data."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="data/sdm_predictions",
        help="Directory for output files."
    )
    
    parser.add_argument(
        "--accuracy-threshold", 
        type=int,
        default=100,
        help="Maximum accuracy threshold for bat records (in meters)."
    )
    
    parser.add_argument(
        "--min-presence", 
        type=int,
        default=15,
        help="Minimum number of presence records required."
    )
    
    return parser.parse_args()


def load_bat_data(bats_path, accuracy_threshold=100):
    """Load and filter the bat occurrence data."""
    bats = gpd.read_file(bats_path)
    bats = bats[bats.accuracy <= accuracy_threshold]
    return bats


def load_background_points(background_path):
    """Load background points for modelling."""
    background = gpd.read_file(background_path)
    background = background[["geometry"]]
    return background


def load_environmental_variables(ev_path):
    """Load environmental variables for modelling."""
    ev_raster = Path(ev_path)
    
    evs_to_model = rxr.open_rasterio(ev_raster, masked=True, band_as_variable=True).squeeze()
    # rename the variables by their long name
    for var in evs_to_model.data_vars:
        evs_to_model = evs_to_model.rename({var: evs_to_model[var].attrs["long_name"]})
    
    return evs_to_model, ev_raster


def load_grid_points(grid_points_path):
    """Load the grid points for spatial reference."""
    return gpd.read_parquet(grid_points_path)


def annotate_points(bats, background, ev_raster, ev_columns):
    """Annotate bat and background points with environmental variables."""
    bats_ant = ela.annotate(
        bats, 
        str(ev_raster), 
        labels=ev_columns
    )
    background = ela.annotate(
        background, 
        str(ev_raster), 
        labels=ev_columns
    )
    return bats_ant, background


def create_maxent_model(n_jobs=1):
    """Create a MaxEnt model pipeline with standard scaling."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "maxent",
                MaxentModel(
                    feature_types=["linear", "hinge", "product"],
                    beta_multiplier=6,
                    n_cpus=n_jobs,
                    class_weights="balanced",
                ),
            ),
        ]
    )
    return model


def generate_training_data(bats_ant, background, grid_points, latin_name, activity_type, ev_columns, min_presence=15):
    """Generate training data for all valid combinations of species and activity types."""
    training_data = []
    filter_combinations = list(product(latin_name, activity_type))
    
    for latin_name, activity_type in filter_combinations:
        presence = filter_bats(bats_ant, latin_name=latin_name, activity_type=activity_type)

        if len(presence) < min_presence:
            print(f"Skipping {latin_name} - {activity_type}: Only {len(presence)} presence records (minimum {min_presence} required)")
            continue

        occurrence = prepare_occurence_data(
            presence, background, grid_points, input_vars=ev_columns
        )
        training_data.append({
            "latin_name": latin_name,
            "activity_type": activity_type,
            "occurrence": occurrence,
        })
    
    return training_data


def train_models(training_data, n_jobs):
    """Train MaxEnt models for each set of training data."""
    results = []
    for data in tqdm(training_data):
        try:
            result = eval_train_model(data["occurrence"], create_maxent_model(n_jobs=n_jobs))
            results.append(result)
        except Exception as e:
            print(f"Error processing {data['latin_name']} - {data['activity_type']}: {e}")
            continue
    
    return results


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
    
    results_df["folds"] = results_df.cv_scores.apply(len)
    
    results_df["activity_type"] = results_df.activity_type.fillna("All")
    
    return results_df


def make_predictions(results_df, ev_raster, output_dir):
    """Apply trained models to make predictions across the study area."""
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_paths = []
    # Define the arguments for each task
    tasks = []
    for _, row in results_df.iterrows():
        latin_name = row.latin_name
        activity_type = row.activity_type
        model = row.final_model
        path_predict = output_dir / f"{latin_name}_{activity_type}.tif"
        prediction_paths.append(path_predict)
        tasks.append({
            "model": model,
            "raster_paths": [ev_raster],
            "output_path": path_predict,
            "latin_name": latin_name,
            "activity_type": activity_type,
        })
    
    # Iterate through the tasks and apply the model to the raster
    for task in tqdm(tasks):
        try:
            output_path = ela.apply_model_to_rasters(**task)
        except Exception as e:
            print(f"Error applying model to raster: {e}")
            continue
    
    results_df["prediction_path"] = prediction_paths
    return results_df


def log_models_to_mlflow(results_df, ev_columns, output_dir):
    """Log models, parameters, and metrics to MLFlow."""
    import mlflow
    from mlflow import log_metric, log_param, log_params, log_artifacts, log_artifact
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    input_var_json_path = models_dir / 'input_variables.json'
    
    with open(input_var_json_path, 'w') as f:
        json.dump(ev_columns, f)
    
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Sheffield Bat Group - SDM - Maxent")
    
    # Iterate over the results dataframe to log models, parameters and metrics
    for _, row in tqdm(results_df.iterrows()):
        with mlflow.start_run(run_name=f"Model_{row['latin_name']}_{row['activity_type']}"):
            mlflow.set_tag("model", "Maxent")
            # Generate a species code from the first 3 letters of the genus and species
            # This makes it easier to identify the species in mlflow
            genus = row["latin_name"].split(" ")[0]
            species = row["latin_name"].split(" ")[1]
            species_code = genus[:3] + "_" + species[:3]
            mlflow.set_tag("species_code", species_code)
            
            mlflow.set_tag("latin_name", row["latin_name"])
            mlflow.set_tag("activity_type", row["activity_type"])
            # Log the parameters
            log_params(row[["n_presence", "n_background", "folds"]].to_dict())
            # Log model parameters
            log_params(row["final_model"].get_params())
            
            # Log the input variables which exceed the param length limit
            log_artifact(input_var_json_path, "input_variables")
            
            # Log the training data
            with NamedTemporaryFile(suffix=".parquet") as f:
                occurence_gdf = row["occurrence"]
                occurence_gdf.to_parquet(f.name)
                log_artifact(f.name, "training_data")
            
            # Log the metrics
            log_metric("mean_cv_score", row["mean_cv_score"])
            log_metric("std_cv_score", row["std_cv_score"])
            
            # Log the predictions as an artifact
            log_artifact(row["prediction_path"], "predictions_raster")
            
            # Log the model
            mlflow.sklearn.log_model(row["final_model"], "model")


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
    
    occurrence_gdf = pd.concat([extract_occurrence_df(row) for _, row in results_df.iterrows()])
    occurrence_gdf.to_parquet(output_dir / "training-occurrence-data.parquet")


def main(
    ev_path : str = "data/evs/evs-to-model.tif",
    bats_path : str = "data/processed/bats-tidy.geojson",
    background_path : str = "data/processed/background-points.geojson",
    grid_points_path : str = "data/evs/grid-points.parquet",
    output_dir : str = "data/sdm_predictions",
    accuracy_threshold : int = 100,
    min_presence : int = 15
):
    """Run the full MaxEnt modelling pipeline."""
    
    # Load data
    print("Loading data...")
    bats = load_bat_data(bats_path, accuracy_threshold)
    background = load_background_points(background_path)
    evs_to_model, ev_raster = load_environmental_variables(ev_path)
    grid_points = load_grid_points(grid_points_path)
    
    # Extract unique species and activity types
    latin_name = bats.latin_name.unique().tolist()
    activity_type = bats.activity_type.unique().tolist()
    print(f"Training models for {len(latin_name)} species and {len(activity_type)} activity types...")
    
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
        grid_points, 
        latin_name, 
        activity_type, 
        ev_columns,
        min_presence=min_presence
    )
    
    print(f"Generated {len(training_data)} valid training datasets")
    
    # Train models
    print("Training models...")
    num_cpus = os.cpu_count()
    results = train_models(training_data, num_cpus)
    
    # Prepare results dataframe
    print("Preparing results...")
    results_df = prepare_results_dataframe(results, training_data)
    
    # Make predictions
    print("Making predictions...")
    results_df = make_predictions(results_df, ev_raster, output_dir)
    
    # Log models to MLFlow
    print("Logging models to MLFlow...")
    log_models_to_mlflow(results_df, ev_columns, output_dir)
    
    # Save results
    print("Saving results...")
    save_results(results_df, output_dir)
    
    print("Modelling complete!")
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
        min_presence=args.min_presence
    )
