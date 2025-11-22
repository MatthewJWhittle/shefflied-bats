"""
Hyperparameter tuning for SDM models using Optuna.

This module implements hyperparameter optimization for MaxEnt models, including:
- Model hyperparameters (beta_multiplier, feature_types, etc.)
- Feature selection (sampling from roster)
- Background point sampling parameters
"""

import json
import logging
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import mlflow
import numpy as np
import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sdm.commands.modelling.train_sdm_models import (
    TrainingSetup,
    _summarize_cv_scores,
    setup_training_data,
    train_models_with_setup,
)
from sdm.models.maxent.maxent_model import ActivityType, DefaultMaxentConfig
from sdm.types import ModelConfig, VariablesConfig, TrainingResults
from sdm.utils.io import (
    load_model_config,
    load_project_config,
    load_variables_config,
)
from sdm.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logs_and_warnings():
    """Context manager to suppress verbose logging and warnings during trial execution."""
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered")
        
        # Temporarily reduce logging level for verbose modules
        old_levels = {}
        modules_to_quiet = [
            'sdm.commands.modelling.train_sdm_models',
            'sdm.models.maxent.maxent_model',
            'sdm.data.processing',
            'elapid',
        ]
        
        for module_name in modules_to_quiet:
            module_logger = logging.getLogger(module_name)
            old_levels[module_name] = module_logger.level
            module_logger.setLevel(logging.ERROR)
        
        try:
            yield
        finally:
            # Restore logging levels
            for module_name, level in old_levels.items():
                logging.getLogger(module_name).setLevel(level)


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


def _apply_feature_bounds(
    trial: optuna.Trial,
    activity_type: str,
    selected: List[str],
    roster: List[str],
    min_features: int = 5,
    max_features: int = 30,
) -> List[str]:
    """Apply minimum and maximum feature bounds.
    
    Args:
        trial: Optuna trial object
        activity_type: Activity type name
        selected: Currently selected features
        roster: Full list of available features
        min_features: Minimum number of features required
        max_features: Maximum number of features allowed
        
    Returns:
        Bounded list of selected features
    """
    # Ensure minimum number of features
    if len(selected) < min_features:
        remaining = [f for f in roster if f not in selected]
        n_needed = min_features - len(selected)
        selected.extend(remaining[:n_needed])
    
    # Limit to maximum features if needed
    if len(selected) > max_features:
        # Use Optuna to prioritize which features to keep
        feature_priorities = {}
        for feature in selected:
            priority_key = f"priority_{activity_type}_{feature}"
            priority = trial.suggest_float(priority_key, 0.0, 1.0)
            feature_priorities[feature] = priority
        
        # Keep top features by priority
        sorted_features = sorted(
            feature_priorities.items(), key=lambda x: x[1], reverse=True
        )
        selected = [f for f, _ in sorted_features[:max_features]]
    
    return selected


def suggest_feature_selection(
    trial: optuna.Trial, 
    roster: List[str], 
    activity_types: List[str],
) -> Dict[str, List[str]]:
    """Suggest feature selection from Optuna trial.
    
    Uses Optuna to learn which features work best by suggesting binary inclusion
    for each feature. Each feature gets a parameter key "{activity_type}_{feature}"
    with values [True, False]. This allows Optuna's TPE sampler to learn which
    features contribute most to model performance.
    
    This approach is simpler and more direct than using candidate sets - Optuna
    can learn from all features in the roster simultaneously.
    
    Args:
        trial: Optuna trial object
        roster: Full list of available features
        activity_types: List of activity types to select features for
        
    Returns:
        Dictionary mapping activity type to selected features
    """
    activity_feature_sets = {}
    
    for activity_type in activity_types:
        # For each feature in the roster, let Optuna decide whether to include it
        # Use feature name as the parameter key: "{activity_type}_{feature}"
        selected = []
        for feature in roster:
            include_key = f"{activity_type}_{feature}"
            include = trial.suggest_categorical(include_key, [True, False])
            if include:
                selected.append(feature)
        
        # Apply minimum and maximum bounds
        selected = _apply_feature_bounds(trial, activity_type, selected, roster)
        
        activity_feature_sets[activity_type] = selected
    
    return activity_feature_sets
    



def _calculate_mean_cv_auc(models: List[TrainingResults]) -> float:
    """Calculate mean CV AUC from training results.
    
    Args:
        models: List of TrainingResults objects
        
    Returns:
        Mean CV AUC score, or 0.0 if no valid scores
    """
    valid_aucs = []
    for model in models:
        if model.success and model.cv_scores is not None:
            cv_mean, _, n_valid, _ = _summarize_cv_scores(model.cv_scores)
            if n_valid > 0:
                valid_aucs.append(cv_mean)
    
    if len(valid_aucs) > 0:
        return float(np.mean(valid_aucs))
    return 0.0


def _create_maxent_config_from_params(maxent_params: Dict[str, Any]) -> DefaultMaxentConfig:
    """Create MaxEnt config from suggested parameters.
    
    Args:
        maxent_params: Dictionary of MaxEnt hyperparameters (keys match DefaultMaxentConfig parameter names)
        
    Returns:
        DefaultMaxentConfig object
    """
    # Use ** unpacking - DefaultMaxentConfig has defaults for all parameters
    return DefaultMaxentConfig(**maxent_params)


def _prepare_trial_hyperparameters(
    trial: optuna.Trial,
    base_variables_config: VariablesConfig,
    subset_occurrence: int,
    activity_types: Optional[List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[ActivityType, List[str]]]:
    """Prepare hyperparameters for a trial.
    
    Args:
        trial: Optuna trial object
        base_variables_config: Base variables config
        subset_occurrence: Number of occurrence records to use for tuning
        activity_types: List of activity types to tune (optional)
        
    Returns:
        Tuple of (maxent_params, sampling_params, feature_selection)
    """
    # Suggest hyperparameters
    maxent_params = suggest_maxent_hyperparameters(trial)
    sampling_params_dict = suggest_sampling_hyperparameters(trial)
    
    # Add subset_occurrence to sampling params
    sampling_params_dict["subset_occurrence"] = subset_occurrence
    
    # Get activity types from base config or use provided
    if activity_types is None:
        activity_types_list = list(base_variables_config.activity_feature_sets.keys())
    else:
        activity_types_list = activity_types
    
    # Suggest feature selection
    feature_selection_dict = suggest_feature_selection(
        trial, 
        base_variables_config.roster, 
        activity_types_list,
    )
    
    # Convert feature selection to ActivityType enum keys
    feature_selection: Dict[ActivityType, List[str]] = {
        ActivityType(k): v for k, v in feature_selection_dict.items()
    }
    
    return maxent_params, sampling_params_dict, feature_selection


def objective(
    trial: optuna.Trial,
    setup: TrainingSetup,
    base_model_config: ModelConfig,
    base_variables_config: VariablesConfig,
    subset_occurrence: int,
    activity_types: Optional[List[str]],
    n_cv_folds: int = 2,
) -> float:
    """Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        setup: TrainingSetup with shared data (reused across trials)
        base_model_config: Base model config
        base_variables_config: Base variables config
        subset_occurrence: Number of occurrence records to use for tuning
        activity_types: List of activity types to tune (optional)
        n_cv_folds: Number of CV folds (default: 2 for faster tuning)
        
    Returns:
        Mean CV AUC score (to maximize)
    """
    try:
        # Prepare hyperparameters
        maxent_params, sampling_params_dict, feature_selection = _prepare_trial_hyperparameters(
            trial, base_variables_config, subset_occurrence, activity_types
        )
        
        # Create MaxEnt config from suggested parameters
        model_config = _create_maxent_config_from_params(maxent_params)
        
        # Run training with subset data
        # Create nested MLflow run for this trial
        trial_run_name = f"trial_{trial.number}"
        with mlflow.start_run(nested=True, run_name=trial_run_name):
            # Log trial parameters to MLflow
            mlflow.log_params(trial.params)
            mlflow.log_metric("trial_number", trial.number)
            
            # Suppress verbose logging and warnings during training
            with suppress_logs_and_warnings():
                # Train models using modular function
                # Use fewer threads and CV folds for faster tuning
                models, _ = train_models_with_setup(
                    setup=setup,
                    model_config=model_config,
                    feature_selection=feature_selection,
                    sampling_params=sampling_params_dict,
                    min_presence=base_model_config.sampling.min_presence,
                    max_threads_per_model=1,  # Use 1 thread per model for tuning (faster, less overhead)
                    n_jobs=1,  # Single job per trial (parallelism handled by Optuna)
                    n_cv_folds=n_cv_folds,  # Use fewer folds for faster tuning
                    verbose=False,
                )
            
            # Calculate mean CV AUC from results
            mean_auc = _calculate_mean_cv_auc(models)
            
            if mean_auc > 0.0:
                mlflow.log_metric("mean_cv_auc", mean_auc)
                mlflow.log_metric("n_models", len([m for m in models if m.success and m.cv_scores is not None]))
                # Log score to console (logger automatically respects level)
                logger.info(f"Trial {trial.number}: mean CV AUC = {mean_auc:.4f}")
            else:
                logger.debug(f"Trial {trial.number}: No valid models with CV scores")
            
            # Report intermediate value for pruning
            trial.report(mean_auc, step=trial.number)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return mean_auc
    
    except optuna.TrialPruned:
        # Re-raise pruning exceptions
        raise
    except Exception as e:
        logger.debug(f"Trial {trial.number} failed: {e}")
        # Return a low score for failed trials
        return 0.0


def _reconstruct_feature_selection(
    trial_params: Dict[str, Any],
    roster: List[str],
    activity_types: List[str],
) -> Dict[str, List[str]]:
    """Reconstruct feature selection from trial parameters.
    
    This function uses the same logic as suggest_feature_selection but reads
    from trial parameters instead of making Optuna suggestions.
    
    Args:
        trial_params: Trial parameters dictionary
        roster: Full list of available features
        activity_types: List of activity types
        
    Returns:
        Dictionary mapping activity type to selected features
    """
    activity_feature_sets = {}
    
    for activity_type in activity_types:
        # Extract features that were set to True in the trial
        # Parameter key format: "{activity_type}_{feature}"
        selected = []
        for feature in roster:
            feature_key = f"{activity_type}_{feature}"
            if feature_key in trial_params and trial_params[feature_key]:
                selected.append(feature)
        
        # Reconstruct bounds (same logic as _apply_feature_bounds but from params)
        # Apply maximum bound if needed
        if len(selected) > 30:
            feature_priorities = {}
            for feature in selected:
                priority_key = f"priority_{activity_type}_{feature}"
                if priority_key in trial_params:
                    feature_priorities[feature] = trial_params[priority_key]
            
            if feature_priorities:
                sorted_features = sorted(
                    feature_priorities.items(), key=lambda x: x[1], reverse=True
                )
                selected = [f for f, _ in sorted_features[:30]]
            else:
                # Fallback: just take first 30
                selected = selected[:30]
        
        # Apply minimum bound
        if len(selected) < 5:
            remaining = [f for f in roster if f not in selected]
            n_needed = 5 - len(selected)
            selected.extend(remaining[:n_needed])
        
        activity_feature_sets[activity_type] = selected
    
    return activity_feature_sets


def _extract_maxent_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MaxEnt hyperparameters from trial parameters.
    
    Args:
        trial_params: Trial parameters dictionary
        
    Returns:
        Dictionary of MaxEnt hyperparameters
    """
    # Convert feature_types from string back to list (Optuna stores it as string)
    feature_types_str = trial_params["feature_types"]
    if isinstance(feature_types_str, str):
        feature_types = feature_types_str.split(",") if "," in feature_types_str else [feature_types_str]
    else:
        feature_types = feature_types_str
    
    return {
        "feature_types": feature_types,
        "beta_multiplier": trial_params["beta_multiplier"],
        "beta_lqp": trial_params["beta_lqp"],
        "beta_hinge": trial_params["beta_hinge"],
        "beta_threshold": trial_params["beta_threshold"],
        "beta_categorical": trial_params["beta_categorical"],
        "n_hinge_features": trial_params["n_hinge_features"],
        "n_threshold_features": trial_params["n_threshold_features"],
        "clamp": trial_params["clamp"],
        "tau": trial_params["tau"],
        "transform": trial_params["transform"],
    }


def _extract_sampling_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sampling hyperparameters from trial parameters.
    
    Args:
        trial_params: Trial parameters dictionary
        
    Returns:
        Dictionary of sampling hyperparameters
    """
    return {
        "background": {
            "factor": trial_params["background_factor"],
            "min_bg": trial_params["background_min_bg"],
            "max_bg": trial_params["background_max_bg"],
        },
        "subset_background": trial_params["subset_background"],
        "order_by_density_for_subset": trial_params["order_by_density_for_subset"],
        "sample_weight_n_neighbors": trial_params["sample_weight_n_neighbors"],
    }


def _write_model_config(
    maxent_params: Dict[str, Any],
    sampling_params: Dict[str, Any],
    base_model_config: ModelConfig,
    output_path: Path,
) -> None:
    """Write model configuration to file.
    
    Args:
        maxent_params: MaxEnt hyperparameters
        sampling_params: Sampling hyperparameters
        base_model_config: Base model config
        output_path: Path to write config file
    """
    model_config_dict = {
        "model": {
            "record_age_years": base_model_config.record_age_years,
            "maxent": {
                **maxent_params,
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
    
    with open(output_path, "w") as f:
        yaml.dump(model_config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"Wrote best model config to {output_path}")


def _write_variables_config(
    feature_selection: Dict[str, List[str]],
    base_variables_config: VariablesConfig,
    output_path: Path,
) -> None:
    """Write variables configuration to file.
    
    Args:
        feature_selection: Feature selection per activity type
        base_variables_config: Base variables config
        output_path: Path to write config file
    """
    variables_config_dict = {
        "variables": {
            "roster": base_variables_config.roster,
            "activity_feature_sets": feature_selection,
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(variables_config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"Wrote best variables config to {output_path}")


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
    
    # Log detailed parameters at DEBUG level (logger automatically respects level)
    logger.debug(f"Best trial: {best_trial.number}")
    logger.debug(f"Best value (mean CV AUC): {best_trial.value:.4f}")
    logger.debug("Best parameters:")
    for key, value in best_trial.params.items():
        logger.debug(f"  {key}: {value}")
    
    # Extract best parameters
    best_maxent_params = _extract_maxent_params(best_trial.params)
    best_sampling_params = _extract_sampling_params(best_trial.params)
    
    # Reconstruct feature selection from best trial
    if activity_types is None:
        activity_types_list = list(base_variables_config.activity_feature_sets.keys())
    else:
        activity_types_list = activity_types
    
    best_feature_selection = _reconstruct_feature_selection(
        best_trial.params,
        base_variables_config.roster,
        activity_types_list,
    )
    
    # Write config files
    _write_model_config(
        best_maxent_params,
        best_sampling_params,
        base_model_config,
        output_model_config_path,
    )
    
    _write_variables_config(
        best_feature_selection,
        base_variables_config,
        output_variables_config_path,
    )


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
    n_jobs: int = 1,
    n_cv_folds: int = 2,
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
        n_jobs: Number of parallel jobs for running trials (default: 1, sequential)
        n_cv_folds: Number of CV folds for tuning (default: 2, faster than 3)
        verbose: Enable verbose logging
        
    Returns:
        Optuna study object
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("Starting hyperparameter tuning...")
    
    # Load base configs
    project_config = load_project_config(project_config_path)
    base_model_config = load_model_config(model_config_path)
    base_variables_config = load_variables_config(variables_config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up shared training data once (reused across all trials)
    logger.debug("Setting up shared training data (one-time)...")
    setup = setup_training_data(
        project_config_path=project_config_path,
        variables_config_path=variables_config_path,
        bats_file=bats_file,
        background_file=background_file,
        ev_file=ev_file,
        grid_points_file=grid_points_file,
        species=species,
        activity_types=activity_types,
        verbose=verbose,
    )
    logger.debug("Setup complete - shared data will be reused for all trials")
    
    # Configure MLflow and create parent run for tuning session
    project_config = load_project_config(project_config_path)
    mlflow.set_tracking_uri(project_config.mlflow.tracking_uri)
    mlflow.set_experiment(project_config.mlflow.experiment_name)
    
    parent_run_name = f"Hyperparameter_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parent_run = mlflow.start_run(run_name=parent_run_name)
    logger.debug(f"Started MLflow parent run: {parent_run_name} (ID: {parent_run.info.run_id})")
    
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
        # More aggressive pruning for faster tuning: prune earlier and more often
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        
        # Enqueue first trial with all features enabled (as suggested in the article)
        # This gives Optuna a baseline to compare against
        if activity_types is None:
            activity_types_list = list(base_variables_config.activity_feature_sets.keys())
        else:
            activity_types_list = activity_types
        
        default_trial_params = {}
        # Add all features as True for each activity type
        for activity_type in activity_types_list:
            for feature in base_variables_config.roster:
                default_trial_params[f"{activity_type}_{feature}"] = True
        
        # Also add default values for other hyperparameters
        # (Optuna will suggest these, but we can provide sensible defaults)
        default_trial_params.update({
            "feature_types": "linear,hinge",
            "beta_multiplier": 1.0,
            "beta_lqp": 1.0,
            "beta_hinge": 1.0,
            "beta_threshold": 1.0,
            "beta_categorical": 1.0,
            "n_hinge_features": 10,
            "n_threshold_features": 10,
            "clamp": True,
            "tau": 0.5,
            "transform": "cloglog",
            "background_factor": 10,
            "background_min_bg": 1000,
            "background_max_bg": 10000,
            "subset_background": True,
            "order_by_density_for_subset": True,
            "sample_weight_n_neighbors": 5,
        })
        
        study.enqueue_trial(default_trial_params)
        logger.debug("Enqueued first trial with all features enabled")
        
        # Create objective function with fixed arguments (including shared setup)
        def objective_fn(trial: optuna.Trial) -> float:
            return objective(
                trial,
                setup,
                base_model_config,
                base_variables_config,
                subset_occurrence,
                activity_types,
                n_cv_folds,
            )
        
        # Run optimization with parallel trials if n_jobs > 1
        logger.info(f"Starting optimization with {n_trials} trials (n_jobs={n_jobs}, n_cv_folds={n_cv_folds})...")
        logger.info("Note: Verbose logs and warnings are suppressed during trials for cleaner output.")
        if n_jobs > 1:
            logger.info(f"Running {n_jobs} trials in parallel for faster optimization")
        
        # Suppress Optuna's default trial logging (we log to MLflow instead)
        # Suppress all Optuna loggers to avoid redundant "Trial X finished..." messages
        optuna_loggers = [
            logging.getLogger("optuna"),
            logging.getLogger("optuna.study"),
            logging.getLogger("optuna.trial"),
        ]
        old_optuna_levels = {}
        for optuna_logger in optuna_loggers:
            old_optuna_levels[optuna_logger] = optuna_logger.level
            optuna_logger.setLevel(logging.WARNING)  # Only show warnings/errors from Optuna
        
        # Suppress warnings during optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message="invalid value encountered")
            study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        
        # Restore Optuna logging levels
        for optuna_logger, old_level in old_optuna_levels.items():
            optuna_logger.setLevel(old_level)
        
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
        
        study_summary_path = output_dir / "study_summary.json"
        with open(study_summary_path, "w") as f:
            json.dump(study_summary, f, indent=2)
        mlflow.log_artifact(str(study_summary_path), "best_configs")
        
        logger.info("=" * 80)
        logger.info("✓ Hyperparameter tuning complete")
        logger.info("=" * 80)
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best mean CV AUC: {study.best_value:.4f}")
        logger.info(f"Trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        logger.info(f"Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"Best configs written to:")
        logger.info(f"  - {best_model_config_path}")
        logger.info(f"  - {best_variables_config_path}")
        logger.debug(f"Results logged to MLflow parent run: {parent_run_name}")
        
    finally:
        # End parent run
        mlflow.end_run()
    
    return study

