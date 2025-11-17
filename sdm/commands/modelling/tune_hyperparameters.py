"""
Hyperparameter tuning for SDM models using Optuna.

This module implements hyperparameter optimization for MaxEnt models, including:
- Model hyperparameters (beta_multiplier, feature_types, etc.)
- Feature selection (sampling from roster)
- Background point sampling parameters
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import mlflow
import optuna
import yaml
import numpy as np
from datetime import datetime
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sdm.commands.modelling.train_sdm_models import train_sdm_models
from sdm.data.loaders.vector import load_background_points, load_bat_data
from sdm.data.processing import annotate_points
from sdm.raster.io import load_environmental_variables
from sdm.utils.io import (
    load_project_config,
    load_model_config,
    load_variables_config,
)
from sdm.utils.logging_utils import setup_logging
from sdm.types import ModelConfig, VariablesConfig

logger = logging.getLogger(__name__)


def suggest_maxent_hyperparameters(trial: optuna.Trial) -> Dict:
    """Suggest MaxEnt hyperparameters from Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    # Feature types - categorical choice
    # Use string representation to avoid Optuna warnings about lists
    feature_type_options = [
        "linear",
        "hinge",
        "linear,hinge",
        "linear,quadratic",
        "linear,hinge,quadratic",
    ]
    feature_types_str = trial.suggest_categorical("feature_types", feature_type_options)
    # Convert back to list
    feature_types = feature_types_str.split(",") if "," in feature_types_str else [feature_types_str]
    
    # Beta multiplier - log uniform distribution
    beta_multiplier = trial.suggest_float("beta_multiplier", 0.1, 10.0, log=True)
    
    # Beta values for specific feature types
    beta_lqp = trial.suggest_float("beta_lqp", 0.1, 5.0, log=True)
    beta_hinge = trial.suggest_float("beta_hinge", 0.1, 5.0, log=True)
    beta_threshold = trial.suggest_float("beta_threshold", 0.1, 5.0, log=True)
    beta_categorical = trial.suggest_float("beta_categorical", 0.1, 5.0, log=True)
    
    # Number of hinge/threshold features
    n_hinge_features = trial.suggest_int("n_hinge_features", 0, 20)
    n_threshold_features = trial.suggest_int("n_threshold_features", 0, 20)
    
    # Clamp
    clamp = trial.suggest_categorical("clamp", [True, False])
    
    # Tau
    tau = trial.suggest_float("tau", 0.1, 0.9)
    
    # Transform
    transform = trial.suggest_categorical("transform", ["cloglog", "logistic", "raw"])
    
    return {
        "feature_types": feature_types,
        "beta_multiplier": beta_multiplier,
        "beta_lqp": beta_lqp,
        "beta_hinge": beta_hinge,
        "beta_threshold": beta_threshold,
        "beta_categorical": beta_categorical,
        "n_hinge_features": n_hinge_features,
        "n_threshold_features": n_threshold_features,
        "clamp": clamp,
        "tau": tau,
        "transform": transform,
    }


def suggest_sampling_hyperparameters(trial: optuna.Trial) -> Dict:
    """Suggest sampling hyperparameters from Optuna trial.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of suggested sampling hyperparameters
    """
    # Background sampling
    background_factor = trial.suggest_int("background_factor", 5, 20)
    background_min_bg = trial.suggest_int("background_min_bg", 500, 2000, step=100)
    background_max_bg = trial.suggest_int("background_max_bg", 5000, 20000, step=500)
    
    # Subset parameters
    subset_background = trial.suggest_categorical("subset_background", [True, False])
    order_by_density_for_subset = trial.suggest_categorical(
        "order_by_density_for_subset", [True, False]
    )
    sample_weight_n_neighbors = trial.suggest_int("sample_weight_n_neighbors", 3, 10)
    
    return {
        "background": {
            "factor": background_factor,
            "min_bg": background_min_bg,
            "max_bg": background_max_bg,
        },
        "subset_background": subset_background,
        "order_by_density_for_subset": order_by_density_for_subset,
        "sample_weight_n_neighbors": sample_weight_n_neighbors,
    }


def suggest_feature_selection(
    trial: optuna.Trial, roster: List[str], activity_types: List[str]
) -> Dict[str, List[str]]:
    """Suggest feature selection from Optuna trial.
    
    Uses a binary selection approach where Optuna suggests which features to include.
    
    Args:
        trial: Optuna trial object
        roster: Full list of available features
        activity_types: List of activity types to select features for
        
    Returns:
        Dictionary mapping activity type to selected features
    """
    activity_feature_sets = {}
    
    for activity_type in activity_types:
        # Sample number of features to use (between 5 and min(30, len(roster)))
        n_features = trial.suggest_int(
            f"n_features_{activity_type}",
            5,
            min(30, len(roster)),
        )
        
        # Use a deterministic but varied selection based on trial number
        # This ensures reproducibility while still exploring different combinations
        # We'll use a hash-based approach to select features
        selected = []
        trial_seed = hash((trial.number, activity_type)) % (2**32)
        np.random.seed(trial_seed)
        
        # Sample without replacement
        selected_indices = np.random.choice(
            len(roster), size=min(n_features, len(roster)), replace=False
        )
        selected = [roster[i] for i in sorted(selected_indices)]
        
        activity_feature_sets[activity_type] = selected
    
    return activity_feature_sets


def create_temporary_configs(
    maxent_params: Dict,
    sampling_params: Dict,
    feature_selection: Dict[str, List[str]],
    base_model_config: ModelConfig,
    base_variables_config: VariablesConfig,
    temp_dir: Path,
) -> Tuple[Path, Path]:
    """Create temporary config files for a trial.
    
    Args:
        maxent_params: MaxEnt hyperparameters
        sampling_params: Sampling hyperparameters
        feature_selection: Feature selection per activity type
        base_model_config: Base model config to inherit from
        base_variables_config: Base variables config to inherit from
        temp_dir: Temporary directory for config files
        
    Returns:
        Tuple of (model_config_path, variables_config_path)
    """
    # Create model config
    model_config_dict = {
        "model": {
            "record_age_years": base_model_config.record_age_years,
            "maxent": {
                **maxent_params,
                # Keep other maxent params from base config
                "convergence_tolerance": base_model_config.maxent.convergence_tolerance,
                "use_lambdas": base_model_config.maxent.use_lambdas,
                "n_lambdas": base_model_config.maxent.n_lambdas,
                "class_weights": base_model_config.maxent.class_weights,
            },
            "sampling": {
                "min_presence": base_model_config.sampling.min_presence,
                "subset_occurrence": base_model_config.sampling.subset_occurrence,
                **sampling_params,
            },
        }
    }
    
    model_config_path = temp_dir / "model_config.yml"
    with open(model_config_path, "w") as f:
        yaml.dump(model_config_dict, f, default_flow_style=False)
    
    # Create variables config
    variables_config_dict = {
        "variables": {
            "roster": base_variables_config.roster,
            "activity_feature_sets": feature_selection,
        }
    }
    
    variables_config_path = temp_dir / "variables_config.yml"
    with open(variables_config_path, "w") as f:
        yaml.dump(variables_config_dict, f, default_flow_style=False)
    
    return model_config_path, variables_config_path


def objective(
    trial: optuna.Trial,
    project_config_path: Path,
    base_model_config: ModelConfig,
    base_variables_config: VariablesConfig,
    bats_file: Path,
    background_file: Path,
    ev_file: Path,
    grid_points_file: Optional[Path],
    output_dir: Path,
    subset_occurrence: int,
    species: Optional[List[str]],
    activity_types: Optional[List[str]],
    temp_dir: Path,
    annotated_bats: Optional[gpd.GeoDataFrame] = None,
    annotated_background: Optional[gpd.GeoDataFrame] = None,
    all_ev_columns: Optional[List[str]] = None,
) -> float:
    """Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        project_config_path: Path to project config
        base_model_config: Base model config
        base_variables_config: Base variables config
        bats_file: Path to bat occurrence data
        background_file: Path to background points
        ev_file: Path to environmental variables
        grid_points_file: Path to grid points (optional)
        output_dir: Output directory for training
        subset_occurrence: Number of occurrence records to use for tuning
        species: List of species to tune (optional)
        activity_types: List of activity types to tune (optional)
        temp_dir: Temporary directory for config files
        
    Returns:
        Mean CV AUC score (to maximize)
    """
    try:
        # Suggest hyperparameters
        maxent_params = suggest_maxent_hyperparameters(trial)
        sampling_params = suggest_sampling_hyperparameters(trial)
        
        # Get activity types from base config or use provided
        if activity_types is None:
            # Extract from base config
            activity_types_list = list(base_variables_config.activity_feature_sets.keys())
        else:
            activity_types_list = activity_types
        
        # Suggest feature selection
        feature_selection = suggest_feature_selection(
            trial, base_variables_config.roster, activity_types_list
        )
        
        # Create temporary config files
        model_config_path, variables_config_path = create_temporary_configs(
            maxent_params,
            sampling_params,
            feature_selection,
            base_model_config,
            base_variables_config,
            temp_dir,
        )
        
        # Use pre-annotated data if available (saves time by skipping annotation)
        # We pass the full annotated data - train_sdm_models will filter by features based on variables_config
        if annotated_bats is not None and annotated_background is not None:
            # Save full annotated data to temp files (don't filter - let train_sdm_models handle it)
            trial_bats_file = temp_dir / f"trial_{trial.number}_bats.parquet"
            trial_background_file = temp_dir / f"trial_{trial.number}_background.parquet"
            annotated_bats.to_parquet(trial_bats_file)
            annotated_background.to_parquet(trial_background_file)
            
            # Use the annotated files (train_sdm_models will skip annotation if data already has EV columns)
            trial_bats_file_path = trial_bats_file
            trial_background_file_path = trial_background_file
        else:
            # Fallback to original files (will be annotated by train_sdm_models)
            trial_bats_file_path = bats_file
            trial_background_file_path = background_file
        
        # Run training with subset data
        # Create nested MLflow run for this trial
        trial_run_name = f"trial_{trial.number}"
        with mlflow.start_run(nested=True, run_name=trial_run_name):
            # Log trial parameters to MLflow
            mlflow.log_params(trial.params)
            mlflow.log_metric("trial_number", trial.number)
            
            results_df = train_sdm_models(
                project_config_path=project_config_path,
                model_config_path=model_config_path,
                variables_config_path=variables_config_path,
                bats_file=trial_bats_file_path,
                background_file=trial_background_file_path,
                ev_file=ev_file,
                grid_points_file=grid_points_file,
                output_dir=output_dir / f"trial_{trial.number}",
                subset_occurrence=subset_occurrence,
                species=species,
                activity_types=activity_types,
                verbose=False,
            )
            
            # Log trial results
            if "cv_auc_mean" in results_df.columns:
                valid_aucs = results_df["cv_auc_mean"].dropna()
                if len(valid_aucs) > 0:
                    mean_auc = float(valid_aucs.mean())
                    mlflow.log_metric("mean_cv_auc", mean_auc)
                    mlflow.log_metric("n_models", len(valid_aucs))
                else:
                    mean_auc = 0.0
            else:
                mean_auc = 0.0
            
            # Calculate mean CV AUC across all models for return value
            if "cv_auc_mean" in results_df.columns:
                valid_aucs = results_df["cv_auc_mean"].dropna()
                if len(valid_aucs) > 0:
                    mean_auc = float(valid_aucs.mean())
                else:
                    mean_auc = 0.0
            else:
                logger.warning("No cv_auc_mean column in results, returning 0.0")
                mean_auc = 0.0
            
            # Report intermediate value for pruning
            trial.report(mean_auc, step=trial.number)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return mean_auc
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        # Return a low score for failed trials
        return 0.0


def write_best_configs(
    study: optuna.Study,
    base_model_config: ModelConfig,
    base_variables_config: VariablesConfig,
    output_model_config_path: Path,
    output_variables_config_path: Path,
    activity_types: Optional[List[str]] = None,
) -> None:
    """Write best hyperparameters to config files.
    
    Args:
        study: Optuna study object
        base_model_config: Base model config
        base_variables_config: Base variables config
        output_model_config_path: Path to write best model config
        output_variables_config_path: Path to write best variables config
        activity_types: List of activity types (optional)
    """
    best_trial = study.best_trial
    
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value (mean CV AUC): {best_trial.value:.4f}")
    logger.info("Best parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Extract best parameters
    best_maxent_params = {
        "feature_types": best_trial.params["feature_types"],
        "beta_multiplier": best_trial.params["beta_multiplier"],
        "beta_lqp": best_trial.params["beta_lqp"],
        "beta_hinge": best_trial.params["beta_hinge"],
        "beta_threshold": best_trial.params["beta_threshold"],
        "beta_categorical": best_trial.params["beta_categorical"],
        "n_hinge_features": best_trial.params["n_hinge_features"],
        "n_threshold_features": best_trial.params["n_threshold_features"],
        "clamp": best_trial.params["clamp"],
        "tau": best_trial.params["tau"],
        "transform": best_trial.params["transform"],
    }
    
    best_sampling_params = {
        "background": {
            "factor": best_trial.params["background_factor"],
            "min_bg": best_trial.params["background_min_bg"],
            "max_bg": best_trial.params["background_max_bg"],
        },
        "subset_background": best_trial.params["subset_background"],
        "order_by_density_for_subset": best_trial.params["order_by_density_for_subset"],
        "sample_weight_n_neighbors": best_trial.params["sample_weight_n_neighbors"],
    }
    
    # Reconstruct feature selection from best trial
    # We need to re-run the feature selection logic with the best trial's parameters
    if activity_types is None:
        activity_types_list = list(base_variables_config.activity_feature_sets.keys())
    else:
        activity_types_list = activity_types
    
    best_feature_selection = {}
    for activity_type in activity_types_list:
        n_features_key = f"n_features_{activity_type}"
        if n_features_key in best_trial.params:
            n_features = best_trial.params[n_features_key]
            # Reconstruct the feature selection using the same hash-based approach
            trial_seed = hash((best_trial.number, activity_type)) % (2**32)
            np.random.seed(trial_seed)
            selected_indices = np.random.choice(
                len(base_variables_config.roster),
                size=min(n_features, len(base_variables_config.roster)),
                replace=False,
            )
            selected = [base_variables_config.roster[i] for i in sorted(selected_indices)]
            best_feature_selection[activity_type] = selected
        else:
            # Fallback to base config
            best_feature_selection[activity_type] = base_variables_config.activity_feature_sets.get(
                activity_type, []
            )
    
    # Write model config
    model_config_dict = {
        "model": {
            "record_age_years": base_model_config.record_age_years,
            "maxent": {
                **best_maxent_params,
                "convergence_tolerance": base_model_config.maxent.convergence_tolerance,
                "use_lambdas": base_model_config.maxent.use_lambdas,
                "n_lambdas": base_model_config.maxent.n_lambdas,
                "class_weights": base_model_config.maxent.class_weights,
            },
            "sampling": {
                "min_presence": base_model_config.sampling.min_presence,
                "subset_occurrence": base_model_config.sampling.subset_occurrence,
                **best_sampling_params,
            },
        }
    }
    
    with open(output_model_config_path, "w") as f:
        yaml.dump(model_config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Wrote best model config to {output_model_config_path}")
    
    # Write variables config
    variables_config_dict = {
        "variables": {
            "roster": base_variables_config.roster,
            "activity_feature_sets": best_feature_selection,
        }
    }
    
    with open(output_variables_config_path, "w") as f:
        yaml.dump(variables_config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Wrote best variables config to {output_variables_config_path}")


def tune_hyperparameters(
    project_config_path: Path,
    model_config_path: Path,
    variables_config_path: Path,
    bats_file: Path,
    background_file: Path,
    ev_file: Path,
    output_dir: Path,
    grid_points_file: Optional[Path] = None,
    n_trials: int = 50,
    subset_occurrence: int = 100,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    verbose: bool = False,
) -> optuna.Study:
    """Run hyperparameter tuning using Optuna.
    
    Args:
        project_config_path: Path to project config
        model_config_path: Path to base model config
        variables_config_path: Path to base variables config
        bats_file: Path to bat occurrence data
        background_file: Path to background points
        ev_file: Path to environmental variables
        output_dir: Output directory for tuning results
        grid_points_file: Path to grid points (optional)
        n_trials: Number of Optuna trials to run
        subset_occurrence: Number of occurrence records to use for tuning
        species: List of species to tune (optional)
        activity_types: List of activity types to tune (optional)
        study_name: Name for Optuna study (optional)
        storage: Optuna storage URL (optional, for distributed tuning)
        verbose: Enable verbose logging
        
    Returns:
        Optuna study object
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("=== Starting Hyperparameter Tuning ===")
    
    # Load base configs
    project_config = load_project_config(project_config_path)
    base_model_config = load_model_config(model_config_path)
    base_variables_config = load_variables_config(variables_config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for trial configs
    temp_dir = Path(tempfile.mkdtemp(prefix="sdm_tune_"))
    logger.info(f"Using temporary directory: {temp_dir}")
    
    # Annotate points once before starting trials (this is expensive and doesn't change)
    logger.info("=== Annotating Points (one-time setup) ===")
    logger.info("Loading data for annotation...")
    bats_ant = load_bat_data(bats_file)
    background, _ = load_background_points(background_file)
    
    logger.info("Loading environmental variables...")
    ev_data, ev_raster_path = load_environmental_variables(ev_file)
    all_ev_columns = list(ev_data.data_vars.keys())
    
    # Filter to roster if specified
    if base_variables_config.roster:
        roster_matches = [col for col in all_ev_columns if col in base_variables_config.roster]
        if roster_matches:
            ev_columns = roster_matches
            logger.info(
                f"Filtered environmental variables to {len(ev_columns)} columns based on roster"
            )
        else:
            ev_columns = all_ev_columns
            logger.warning(
                "No roster variables matched available environmental variables; "
                f"using full set ({len(ev_columns)} columns)."
            )
    else:
        ev_columns = all_ev_columns
        logger.info(f"Found {len(ev_columns)} environmental variables")
    
    logger.info("Annotating points with environmental variables...")
    annotated_bats_gdf, annotated_background_gdf = annotate_points(
        bats_ant, background, ev_raster_path, all_ev_columns
    )
    logger.info("Annotation complete - will reuse for all trials")
    
    # Save annotated data to temporary files for reuse
    annotated_bats_file = temp_dir / "annotated_bats.parquet"
    annotated_background_file = temp_dir / "annotated_background.parquet"
    annotated_bats_gdf.to_parquet(annotated_bats_file)
    annotated_background_gdf.to_parquet(annotated_background_file)
    logger.info(f"Saved annotated data to temporary files for reuse")
    
    # Configure MLflow and create parent run for tuning session
    project_config = load_project_config(project_config_path)
    mlflow.set_tracking_uri(project_config.mlflow.tracking_uri)
    mlflow.set_experiment(project_config.mlflow.experiment_name)
    
    parent_run_name = f"Hyperparameter_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parent_run = mlflow.start_run(run_name=parent_run_name)
    logger.info(f"Started MLflow parent run: {parent_run_name} (ID: {parent_run.info.run_id})")
    
    try:
        # Log tuning configuration
        mlflow.log_params({
            "tuning_n_trials": n_trials,
            "tuning_subset_occurrence": subset_occurrence,
            "tuning_study_name": study_name or "sdm_hyperparameter_tuning",
        })
        if species:
            mlflow.log_param("tuning_species", ",".join(species))
        if activity_types:
            mlflow.log_param("tuning_activity_types", ",".join(activity_types))
        
        # Create Optuna study
        study_name = study_name or "sdm_hyperparameter_tuning"
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        
        # Create objective function with fixed arguments (including pre-annotated data)
        def objective_fn(trial):
            return objective(
                trial,
                project_config_path,
                base_model_config,
                base_variables_config,
                bats_file,
                background_file,
                ev_file,
                grid_points_file,
                output_dir,
                subset_occurrence,
                species,
                activity_types,
                temp_dir,
                annotated_bats=annotated_bats_gdf,
                annotated_background=annotated_background_gdf,
                all_ev_columns=all_ev_columns,
            )
        
        # Run optimization
        logger.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
        
        # Write best configs
        best_model_config_path = output_dir / "best_model_config.yml"
        best_variables_config_path = output_dir / "best_variables_config.yml"
        
        write_best_configs(
            study,
            base_model_config,
            base_variables_config,
            best_model_config_path,
            best_variables_config_path,
            activity_types,
        )
        
        # Log best results to MLflow parent run
        mlflow.log_metric("best_mean_cv_auc", study.best_value)
        mlflow.log_metric("n_trials_completed", len(study.trials))
        mlflow.log_metric("n_trials_pruned", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
        mlflow.log_metric("n_trials_complete", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
        
        # Log best parameters
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)
        
        # Log best config files as artifacts
        mlflow.log_artifact(str(best_model_config_path), "best_configs")
        mlflow.log_artifact(str(best_variables_config_path), "best_configs")
        
        # Log Optuna study summary
        study_summary = {
            "best_value": float(study.best_value),
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
        }
        import json
        study_summary_path = output_dir / "study_summary.json"
        with open(study_summary_path, "w") as f:
            json.dump(study_summary, f, indent=2)
        mlflow.log_artifact(str(study_summary_path), "best_configs")
        
        logger.info("=== Hyperparameter Tuning Complete ===")
        logger.info(f"Best mean CV AUC: {study.best_value:.4f}")
        logger.info(f"Best configs written to:")
        logger.info(f"  - {best_model_config_path}")
        logger.info(f"  - {best_variables_config_path}")
        logger.info(f"Results logged to MLflow parent run: {parent_run_name}")
        
    finally:
        # End parent run
        mlflow.end_run()
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    
    return study

