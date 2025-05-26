"""Model training functionality for SDM models."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np
import mlflow

from sdm.models.maxent_model import (
    create_maxent_pipeline,
    get_feature_config,
    ActivityType,
    evaluate_and_train_maxent_model,
    predict_rasters_with_elapid_model
)
from sdm.models.utils import prepare_occurrence_data
from sdm.occurrence.processing import filter_bats_data

logger = logging.getLogger(__name__)

def save_trained_model(
    model_object: Any, 
    species_name: str, 
    activity_type_value: str, 
    feature_names: List[str],
    output_dir: Path
) -> Path:
    """Saves a trained model object and its feature list."""
    import pickle
    import json

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
    
    return model_path

def log_model_run_to_mlflow(
    model_run_series: pd.Series,
    ev_column_names: List[str],
    run_name_prefix: str,
    output_dir_for_artifacts: Path
) -> None:
    """Logs a single model run's details, parameters, metrics, and model to MLflow."""
    import pickle
    import json

    if pd.isna(model_run_series.get("model_path")) or model_run_series.get("status") != "Success":
        logger.info(f"Skipping MLflow logging for {model_run_series['species']} - {model_run_series['activity']} due to status: {model_run_series.get('status')}")
        return

    try:
        with mlflow.start_run(
            run_name=f"{run_name_prefix}_{model_run_series['species']}_{model_run_series['activity']}",
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
            mlflow.log_artifact(str(feature_list_path), "model_artifacts/features")
            feature_list_path.unlink()
            
            # Log list of all EV columns
            ev_list_path = output_dir_for_artifacts / "temp_ev_column_list.json"
            with open(ev_list_path, "w") as f:
                json.dump(ev_column_names, f)
            mlflow.log_artifact(str(ev_list_path), "model_artifacts/environment_context")
            ev_list_path.unlink()

            # Load and log the scikit-learn model
            model_path = Path(model_run_series["model_path"])
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_object = pickle.load(f)
                mlflow.sklearn.log_model(model_object, "model")
            else:
                logger.warning(f"Model file not found at {model_path}, cannot log model to MLflow.")
            
            logger.info(f"Successfully logged run to MLflow: {run.info.run_name}")

    except Exception as e:
        logger.error(f"Error logging model to MLflow for {model_run_series['species']} - {model_run_series['activity']}: {e}", exc_info=True)

def train_single_model(
    species_name: str,
    activity_type: str,
    annotated_bats_gdf: gpd.GeoDataFrame,
    annotated_background_gdf: gpd.GeoDataFrame,
    grid_ref_gdf: gpd.GeoDataFrame,
    feature_configs: Dict[ActivityType, List[str]],
    output_dir: Path,
    min_presence: int = 15,
    max_threads_per_model: int = 2,
    generate_prediction_rasters: bool = True,
    ev_raster_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Train a single model for a species and activity type combination."""
    
    activity_type_enum = ActivityType(activity_type)
    logger.info(f"Processing model for: {species_name} - {activity_type_enum.value}")

    # Initialize variables
    mean_cv_score = np.nan
    std_cv_score = np.nan
    saved_model_path = None
    final_model = None
    training_data_gdf_len_presence = 0
    training_data_gdf_len_background = 0
    prediction_raster_path_str = None

    # Filter bat data
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
        return {
            "species": species_name,
            "activity": activity_type_enum.value,
            "n_presence": len(current_bats_filtered),
            "n_background": 0,
            "mean_cv_score": np.nan,
            "std_cv_score": np.nan,
            "model_path": None,
            "feature_names_used": [],
            "status": f"Skipped: Low presence ({len(current_bats_filtered)} < {min_presence})"
        }

    # Get feature list
    if activity_type_enum not in feature_configs:
        logger.warning(f"No feature configuration found for activity type {activity_type_enum.value}. Skipping.")
        return {
            "species": species_name,
            "activity": activity_type_enum.value,
            "n_presence": len(current_bats_filtered),
            "n_background": 0,
            "mean_cv_score": np.nan,
            "std_cv_score": np.nan,
            "model_path": None,
            "feature_names_used": [],
            "status": f"Skipped: No feature config for {activity_type_enum.value}"
        }
    
    current_feature_names = feature_configs[activity_type_enum]
    
    # Check if all features are present
    missing_features = [f for f in current_feature_names if f not in annotated_background_gdf.columns or f not in current_bats_filtered.columns]
    if missing_features:
        logger.error(f"Missing required features for {species_name} - {activity_type_enum.value}: {missing_features}. Skipping.")
        return {
            "species": species_name,
            "activity": activity_type_enum.value,
            "n_presence": len(current_bats_filtered),
            "n_background": 0,
            "mean_cv_score": np.nan,
            "std_cv_score": np.nan,
            "model_path": None,
            "feature_names_used": current_feature_names,
            "status": f"Skipped: Missing features {missing_features}"
        }

    try:
        # Prepare occurrence data
        training_data_gdf = prepare_occurrence_data(
            presence_gdf=current_bats_filtered, 
            background_gdf=annotated_background_gdf, 
            grid_gdf=grid_ref_gdf,
            input_vars=current_feature_names,
            drop_na=True, 
            sample_weight_n_neighbors=5,
            filter_to_grid=True,
            subset_background=True
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
            try:
                # Create and train model
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

                # Save model
                saved_model_path = save_trained_model(
                    model_object=final_model,
                    species_name=species_name.replace(" ", "_"),
                    activity_type_value=activity_type_enum.value.replace(" ", "_").lower(),
                    feature_names=current_feature_names,
                    output_dir=output_dir
                )
                status_message = "Success"

                # Generate prediction raster if requested
                if generate_prediction_rasters and ev_raster_path:
                    try:
                        logger.info(f"Generating prediction raster for {species_name} - {activity_type_enum.value}...")
                        pred_output_dir = output_dir / "predictions" / f"{species_name.replace(' ', '_')}_{activity_type_enum.value.replace(' ', '_').lower()}"
                        pred_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        prediction_raster_filename = f"prediction_{species_name.replace(' ', '_')}_{activity_type_enum.value.replace(' ', '_').lower()}.tif"
                        current_prediction_raster_path = pred_output_dir / prediction_raster_filename

                        predict_rasters_with_elapid_model(
                            model=final_model,
                            raster_paths=[str(ev_raster_path)],
                            output_path=str(current_prediction_raster_path)
                        )
                        prediction_raster_path_str = str(current_prediction_raster_path)
                        logger.info(f"Prediction raster saved to: {prediction_raster_path_str}")

                    except Exception as e_pred:
                        logger.error(f"Error generating prediction raster for {species_name} - {activity_type_enum.value}: {e_pred}", exc_info=True)
                        status_message += f"; PredictionRasterError: {e_pred}"

            except Exception as e_model:
                logger.error(f"Error training/evaluating/saving model for {species_name} - {activity_type_enum.value}: {e_model}", exc_info=True)
                status_message = f"Error in model training: {e_model}"
    
    except Exception as e_prep:
        logger.error(f"Error during data preparation for {species_name} - {activity_type_enum.value}: {e_prep}", exc_info=True)
        status_message = f"Error in data prep: {e_prep}"

    return {
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
    }
