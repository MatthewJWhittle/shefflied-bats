import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any # Added List, Dict, Tuple, Any
from enum import StrEnum
import tempfile # Added tempfile

import typer
import pandas as pd
import geopandas as gpd
import numpy as np
import mlflow # Uncommented mlflow
# import elapid as ela # Will be used by functions moved here

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_config # May need this for paths or params
from sdm.data.loaders import (
    load_bat_data,
    load_background_points,
    load_environmental_variables,
    load_grid_points
)
from sdm.data.extraction import annotate_points
from sdm.models.maxent_model import (
    create_maxent_pipeline, 
    get_feature_config, 
    ActivityType, 
    evaluate_and_train_maxent_model, 
    predict_rasters_with_elapid_model
)
from sdm.models.utils import prepare_occurrence_data, calculate_background_points
from sdm.occurrence.processing import filter_bats_data # Renamed from filter_bats

app = typer.Typer()

logger = logging.getLogger(__name__)

# --- Helper function for saving model --- (Adapted from save_models in original)
def save_trained_model(
    model_object: Any, 
    species_name: str, 
    activity_type_value: str, 
    feature_names: List[str],
    output_dir: Path
) -> Path:
    """Saves a trained model object and its feature list."""
    import pickle # Local import for this function
    import json   # Local import for this function

    model_sub_dir = output_dir / "models" / f"{species_name}_{activity_type_value}"
    model_sub_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"{species_name}_{activity_type_value}_model.pkl"
    model_path = model_sub_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(model_object, f)
    logger.info(f"Saved model object to: {model_path}")

    features_path = model_sub_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Saved model features to: {features_path}")
    
    return model_path # Return path to the pickled model

# --- Helper function for MLflow logging --- (Adapted from log_models_to_mlflow)
def log_model_run_to_mlflow(
    model_run_series: pd.Series, # Expects a row from the results DataFrame
    ev_column_names: List[str],
    # mlflow_experiment_name: str, # Experiment is set once in main
    run_name_prefix: str,
    output_dir_for_artifacts: Path # Base output directory for locating artifacts if needed
):
    """Logs a single model run's details, parameters, metrics, and model to MLflow."""
    import pickle # For loading pickled model
    import json # For saving feature list artifact

    # Check if model training was successful and a model path exists
    if pd.isna(model_run_series.get("model_path")) or model_run_series.get("status") != "Success":
        logger.info(f"Skipping MLflow logging for {model_run_series['species']} - {model_run_series['activity']} due to status: {model_run_series.get('status')}")
        return

    try:
        with mlflow.start_run(
            run_name=f"{run_name_prefix}_{model_run_series['species']}_{model_run_series['activity']}",
            # experiment_id=mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id # Ensure experiment is set before calling main loop
            description=f"MaxEnt model for {model_run_series['species']} ({model_run_series['activity']})"
        ) as run:
            mlflow.set_tag("species_latin_name", model_run_series["species"])
            mlflow.set_tag("activity_type", model_run_series["activity"])
            mlflow.set_tag("model_type", "MaxEnt_Elapid")
            mlflow.set_tag("status", model_run_series["status"])

            # Log parameters
            params_to_log = {
                "n_presence_records": model_run_series["n_presence"],
                "n_background_records": model_run_series["n_background"],
                # Add other relevant params from model_run_series if available (e.g., CV folds, model hyperparams)
            }
            mlflow.log_params(params_to_log)

            # Log metrics
            metrics_to_log = {
                "mean_cv_auc": model_run_series["mean_cv_score"],
                "std_cv_auc": model_run_series["std_cv_score"],
            }
            mlflow.log_metrics(metrics_to_log)

            # Log feature names as an artifact
            feature_list_path = output_dir_for_artifacts / "temp_feature_list.json"
            with open(feature_list_path, "w") as f:
                json.dump(model_run_series["feature_names_used"], f)
            mlflow.log_artifact(str(feature_list_path), "model_artifacts/features") # Convert to str
            feature_list_path.unlink() # Clean up temp file
            
            # Log list of all EV columns (context for features used)
            ev_list_path = output_dir_for_artifacts / "temp_ev_column_list.json"
            with open(ev_list_path, "w") as f:
                json.dump(ev_column_names, f)
            mlflow.log_artifact(str(ev_list_path), "model_artifacts/environment_context") # Convert to str
            ev_list_path.unlink()

            # Load and log the scikit-learn model
            model_path = Path(model_run_series["model_path"])
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_object = pickle.load(f)
                mlflow.sklearn.log_model(model_object, "model") # Logs the sklearn pipeline
            else:
                logger.warning(f"Model file not found at {model_path}, cannot log model to MLflow.")

            # Log other artifacts if needed (e.g., prediction raster path, plots)
            # For example, if prediction raster is generated per model:
            # prediction_raster_path = model_run_series.get("prediction_raster_path")
            # if prediction_raster_path and Path(prediction_raster_path).exists():
            #    mlflow.log_artifact(prediction_raster_path, "predictions")
            
            logger.info(f"Successfully logged run to MLflow: {run.info.run_name}")

    except Exception as e:
        logger.error(f"Error logging model to MLflow for {model_run_series['species']} - {model_run_series['activity']}: {e}", exc_info=True)

@app.command()
def main(
    ev_path: Path = typer.Option(
        "data/evs/evs-to-model.tif", 
        help="Path to environmental variables raster (GeoTIFF)."),
    bats_path: Path = typer.Option(
        "data/processed/bats-tidy.geojson", 
        help="Path to bat occurrence data (GeoJSON)."),
    background_path: Path = typer.Option(
        "data/processed/background-points.geojson", 
        help="Path to background points data (GeoJSON)."),
    grid_points_path: Path = typer.Option(
        "data/evs/grid-points.parquet", 
        help="Path to grid points data (Parquet for spatial reference of grid cells)."),
    output_dir: Path = typer.Option(
        "outputs/sdm_runs", 
        help="Directory to save model outputs.", 
        writable=True, resolve_path=True, file_okay=False, dir_okay=True),
    accuracy_threshold: int = typer.Option(
        100, 
        help="Maximum accuracy threshold for bat records (in meters)."),
    min_presence: int = typer.Option(
        15, 
        help="Minimum number of presence records required for a species/activity to be modelled."),
    cpus: Optional[int] = typer.Option(
        None, 
        help="Number of CPUs to use for parallel model training. Default is 80% of available CPUs."),
    max_threads_per_model: int = typer.Option(
        2, 
        help="Maximum number of threads for each individual Maxent model. Used to calculate parallel jobs."),
    # subset_param: Optional[int] = typer.Option(None, "--subset", help="Optional: Number of records to subset for testing pipeline."), # Renamed from 'subset' to avoid conflict if used as var
    mlflow_tracking_uri: Optional[str] = typer.Option(None, help="MLflow tracking URI."),
    mlflow_experiment_name: str = typer.Option("Species_SDM_Maxent_Runs", help="Name for the MLflow experiment."),
    run_name_prefix: str = typer.Option("MaxentRun", help="Prefix for MLflow run names."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
    generate_prediction_rasters: bool = typer.Option(True, help="Whether to generate and save prediction rasters.")
):
    """
    Main script to run Species Distribution Models using MaxEnt.
    This script orchestrates data loading, preparation, model training, 
    evaluation, prediction saving, and MLflow logging.
    """
    setup_logging(level = logging.INFO, verbose = verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting SDM model run. Output directory: {output_dir}")

    # --- Configuration & Setup ---
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(mlflow_experiment_name)
            logger.info(f"MLflow experiment '{mlflow_experiment_name}' created with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"MLflow experiment '{mlflow_experiment_name}' found with ID: {experiment_id}")
        mlflow.set_experiment(experiment_name=mlflow_experiment_name) # Set for subsequent runs
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment '{mlflow_experiment_name}': {e}. MLflow logging might fail.", exc_info=True)

    # Calculate CPU usage
    if cpus is None:
        import os # Import os locally
        total_cpus = os.cpu_count() or 1 # Default to 1 if os.cpu_count() is None
        cpus = max(1, int(total_cpus * 0.8))
        logger.info(f"Automatically using {cpus} CPUs out of {total_cpus} available")
    else:
        logger.info(f"Using {cpus} CPUs as specified")
    n_parallel_jobs = max(1, cpus // max_threads_per_model)
    logger.info(f"Calculated {n_parallel_jobs} parallel jobs for model training.")

    # --- Data Loading ---
    logger.info("Loading data...")
    bat_occurrences_gdf = load_bat_data(bats_path, accuracy_threshold=accuracy_threshold)
    background_points_gdf, background_density_series = load_background_points(background_path)
    evs_dataset, ev_raster_path_obj = load_environmental_variables(ev_path) # ev_raster_path_obj is a Path
    # grid_points_gdf = load_grid_points(grid_points_path) # grid_points_gdf is used by original generate_training_data
                                                        # but prepare_occurrence_data now uses grid_gdf passed directly.
                                                        # It's created by generate_model_grid in occurrence.sampling.
                                                        # For now, we'll load it as it might be needed for other parts or context.
    grid_ref_gdf = load_grid_points(grid_points_path) # Renamed to avoid confusion with a grid generated on the fly

    ev_column_names = list(evs_dataset.data_vars.keys())
    logger.info(f"Environmental variables loaded: {ev_column_names}")

    # --- Annotate points with EV data ---
    logger.info("Annotating points with environmental variable data...")
    # Ensure ev_raster_path_obj is a string for elapid.annotate as used in the original annotate_points
    annotated_bats_gdf, annotated_background_gdf = annotate_points(
        bat_occurrences_gdf, 
        background_points_gdf, 
        str(ev_raster_path_obj), # Pass the path string
        ev_column_names
    )
    logger.info("Annotation complete.")

    # --- Iterate through species and activity types to train models ---
    # This part will replace the `generate_training_data` and `train_models_parallel` logic.
    
    all_species_names = annotated_bats_gdf.latin_name.unique().tolist()
    all_activity_types = annotated_bats_gdf.activity_type.unique().tolist()
    
    feature_configs = get_feature_config() # From maxent_model.py

    results_list = [] # To store results from each model run

    for species_name in all_species_names:
        for activity_code in all_activity_types:
            activity_type_enum = ActivityType(activity_code)
            logger.info(f"Processing model for: {species_name} - {activity_type_enum.value}")

            # Initialize variables that might not be set if an error occurs early
            mean_cv_score = np.nan
            std_cv_score = np.nan
            saved_model_path = None
            final_model = None # Initialize final_model too
            training_data_gdf_len_presence = 0 # For n_presence
            training_data_gdf_len_background = 0 # For n_background

            # 1. Filter bat data for current species and activity
            current_bats_filtered = filter_bats_data(
                annotated_bats_gdf, 
                latin_name=species_name, 
                activity_type=activity_type_enum.value
            )

            if len(current_bats_filtered) < min_presence:
                logger.warning(
                    f"Skipping {species_name} - {activity_type_enum.value} due to insufficient presence records "
                    f"({len(current_bats_filtered)} < {min_presence})."
                )
                # Still append a result row indicating skip
                results_list.append({
                    "species": species_name,
                    "activity": activity_type_enum.value,
                    "n_presence": len(current_bats_filtered), # Report initial presence count
                    "n_background": 0,
                    "mean_cv_score": np.nan,
                    "std_cv_score": np.nan,
                    "model_path": None,
                    "feature_names_used": [],
                    "status": f"Skipped: Low presence ({len(current_bats_filtered)} < {min_presence})"
                })
                continue

            # 2. Get feature list for this activity type
            # Ensure ActivityType enum is used as key for feature_configs
            if activity_type_enum not in feature_configs:
                logger.warning(f"No feature configuration found for activity type {activity_type_enum.value}. Skipping.")
                results_list.append({
                    "species": species_name,
                    "activity": activity_type_enum.value,
                    "n_presence": len(current_bats_filtered),
                    "n_background": 0,
                    "mean_cv_score": np.nan,
                    "std_cv_score": np.nan,
                    "model_path": None,
                    "feature_names_used": [],
                    "status": f"Skipped: No feature config for {activity_type_enum.value}"
                })
                continue
            current_feature_names = feature_configs[activity_type_enum]
            
            # Check if all features are present in the annotated data
            missing_features = [f for f in current_feature_names if f not in annotated_background_gdf.columns or f not in current_bats_filtered.columns]
            if missing_features:
                logger.error(f"Missing required features for {species_name} - {activity_type_enum.value}: {missing_features}. Skipping.")
                results_list.append({
                    "species": species_name,
                    "activity": activity_type_enum.value,
                    "n_presence": len(current_bats_filtered),
                    "n_background": 0,
                    "mean_cv_score": np.nan,
                    "std_cv_score": np.nan,
                    "model_path": None,
                    "feature_names_used": current_feature_names, # report attempted features
                    "status": f"Skipped: Missing features {missing_features}"
                })
                continue

            # 3. Prepare occurrence data for the model
            # `prepare_occurrence_data` needs `grid_gdf` which is the reference grid for spatial filtering.
            # This was loaded as `grid_ref_gdf`.
            # It expects `grid_gdf` to have a unique index or a 'grid_id' column.
            # `load_grid_points` loads a GeoDataFrame; we need to ensure it meets `prepare_occurrence_data` needs.
            # For now, assuming grid_ref_gdf is suitable as is. Its index will be used by default if 'grid_index' not present.
            
            try:
                training_data_gdf = prepare_occurrence_data(
                    presence_gdf=current_bats_filtered, 
                    background_gdf=annotated_background_gdf, 
                    background_density=background_density_series, # Passed from load_background_points
                    grid_gdf=grid_ref_gdf, # This is the reference grid for filtering
                    input_vars=current_feature_names,
                    drop_na=True, 
                    sample_weight_n_neighbors=5, # Default from original eval_train_model
                    filter_to_grid=True, # Default from original eval_train_model
                    subset_background=True # Default
                )
                training_data_gdf_len_presence = int(training_data_gdf["class"].sum())
                training_data_gdf_len_background = int((training_data_gdf["class"] == 0).sum())

                if training_data_gdf.empty or training_data_gdf_len_presence == 0:
                    logger.warning(f"No presence points remaining after data preparation for {species_name} - {activity_type_enum.value}. Skipping.")
                    status_message = "Skipped: No presences after prep"
                elif training_data_gdf_len_background == 0:
                    logger.warning(f"No background points remaining after data preparation for {species_name} - {activity_type_enum.value}. Skipping.")
                    status_message = "Skipped: No background after prep"
                else:
                    # This inner try-except is for model training and saving
                    try:
                        maxent_pipeline = create_maxent_pipeline(
                            feature_names=current_feature_names, 
                            maxent_n_jobs=max_threads_per_model
                        )
                        
                        final_model, cv_models, cv_scores = evaluate_and_train_maxent_model(
                            model=maxent_pipeline, 
                            occurrence_gdf=training_data_gdf, 
                            n_cv_folds=3, 
                            feature_columns=current_feature_names,
                            random_state_kfold=42
                        )
                        mean_cv_score = np.nanmean(cv_scores) if cv_scores.size > 0 else np.nan
                        std_cv_score = np.nanstd(cv_scores) if cv_scores.size > 0 else np.nan
                        logger.info(f"Model for {species_name} - {activity_type_enum.value}: Mean CV Score = {mean_cv_score:.4f}")

                        saved_model_path = save_trained_model(
                            model_object=final_model,
                            species_name=species_name.replace(" ", "_"),
                            activity_type_value=activity_type_enum.value.replace(" ", "_").lower(),
                            feature_names=current_feature_names,
                            output_dir=output_dir
                        )
                        status_message = "Success"

                    except Exception as e_model:
                        logger.error(f"Error training/evaluating/saving model for {species_name} - {activity_type_enum.value}: {e_model}", exc_info=True)
                        # mean_cv_score, std_cv_score, saved_model_path remain at their initial np.nan/None values
                        status_message = f"Error in model training: {e_model}"
            
            except Exception as e_prep: # This except corresponds to the outer try for prepare_occurrence_data
                logger.error(f"Error during data preparation for {species_name} - {activity_type_enum.value}: {e_prep}", exc_info=True)
                # training_data_gdf_len_presence/background remain 0, mean_cv_score/std_cv_score remain np.nan
                status_message = f"Error in data prep: {e_prep}"
            
            prediction_raster_path_str = None # Initialize
            if final_model and saved_model_path and generate_prediction_rasters and status_message == "Success":
                try:
                    logger.info(f"Generating prediction raster for {species_name} - {activity_type_enum.value}...")
                    pred_output_dir = output_dir / "predictions" / f"{species_name.replace(' ', '_')}_{activity_type_enum.value.replace(' ', '_').lower()}"
                    pred_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Construct a unique name for the prediction raster
                    prediction_raster_filename = f"prediction_{species_name.replace(' ', '_')}_{activity_type_enum.value.replace(' ', '_').lower()}.tif"
                    current_prediction_raster_path = pred_output_dir / prediction_raster_filename

                    # We need to provide a list of feature names in the correct order for prediction.
                    # The model pipeline (feature_selector step) was trained with `current_feature_names`.
                    # `predict_rasters_with_elapid_model` takes `raster_paths`. 
                    # If `ev_raster_path_obj` is a single multi-band GeoTIFF, `elapid` needs to know how to map its bands to `current_feature_names`.
                    # `elapid.geo.apply_model_to_array` uses the order of rasters in its `rasters` argument.
                    # For a single multi-band file, this implies bands should be in the order of `current_feature_names`.
                    # This usually requires the EV stack to be prepared with bands in a known, consistent order.
                    
                    # Assuming ev_raster_path_obj is a single file whose bands correspond to features.
                    # The `predict_rasters_with_elapid_model` expects a list of raster paths.
                    # If ev_raster_path_obj is a single stack, we pass it as a list of one item.
                    # The internal elapid logic must then correctly map bands to feature names used in the model.
                    # The model itself (ColumnTransformer) selects features by name. So elapid must present the named bands correctly.
                    
                    predict_rasters_with_elapid_model(
                        model=final_model, # The trained pipeline
                        raster_paths=[str(ev_raster_path_obj)], # Path to the EV stack
                        output_path=str(current_prediction_raster_path),
                        # template_raster_idx=0, # Default, use the first (and only) raster in list as template
                        # predict_proba=True, # Default
                        # creation_options={"COMPRESS": "LZW"} # Example options
                    )
                    prediction_raster_path_str = str(current_prediction_raster_path)
                    logger.info(f"Prediction raster saved to: {prediction_raster_path_str}")

                except Exception as e_pred:
                    logger.error(f"Error generating prediction raster for {species_name} - {activity_type_enum.value}: {e_pred}", exc_info=True)
                    status_message += f"; PredictionRasterError: {e_pred}" # Append to status
            
            results_list.append({
                "species": species_name,
                "activity": activity_type_enum.value,
                "n_presence": training_data_gdf_len_presence,
                "n_background": training_data_gdf_len_background,
                "mean_cv_score": mean_cv_score,
                "std_cv_score": std_cv_score,
                "model_path": str(saved_model_path) if saved_model_path else None,
                "prediction_raster_path": prediction_raster_path_str,
                "feature_names_used": current_feature_names if 'current_feature_names' in locals() else [],
                "status": status_message
            })

    # --- Consolidate and Save Results --- 
    if not results_list:
        logger.warning("No models were processed. Skipping results saving and MLflow logging.")
    else:
        results_summary_df = pd.DataFrame(results_list)
        
        # Save consolidated results to CSV
        results_csv_path = output_dir / "sdm_run_summary.csv" # Changed filename for clarity
        try:
            results_summary_df.to_csv(results_csv_path, index=False)
            logger.info(f"Saved run summary to: {results_csv_path}")
        except Exception as e_save_csv:
            logger.error(f"Failed to save run summary CSV: {e_save_csv}", exc_info=True)

        # Save results to Pickle (as in original)
        results_pkl_path = output_dir / "sdm_run_summary.pkl"
        try:
            results_summary_df.to_pickle(results_pkl_path)
            logger.info(f"Saved run summary to Pickle: {results_pkl_path}")
        except Exception as e_save_pkl:
            logger.error(f"Failed to save run summary Pickle: {e_save_pkl}", exc_info=True)

        # TODO: Consider adapting the logic for saving combined training occurrence data.
        # Original script had: 
        # occurrence_gdf = pd.concat([extract_occurrence_df(row) for _, row in results_df.iterrows()])
        # occurrence_gdf.to_parquet(output_dir / "training-occurrence-data.parquet")
        # This requires storing/reconstructing the `training_data_gdf` for each run or accessing it differently.
        # For now, this part is omitted for simplicity in the refactor, but can be added if essential.
        logger.info("Consolidated training data saving (from original script) is currently omitted.")

        # --- MLflow Logging for each successful model run ---
        logger.info("Logging model runs to MLflow...")
        for index, row in results_summary_df.iterrows():
            log_model_run_to_mlflow(
                model_run_series=row,
                ev_column_names=ev_column_names, # Full list of available EV names
                run_name_prefix=run_name_prefix,
                output_dir_for_artifacts=output_dir
            )

    logger.info("SDM model run script finished.")

if __name__ == "__main__":
    app() 