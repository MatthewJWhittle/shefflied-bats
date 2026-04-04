"""
Hyperparameter tuning for SDM models using Optuna.

This module implements hyperparameter optimization for MaxEnt models, including:
- Model hyperparameters (beta_multiplier, feature_types, etc.)
- Feature selection (sampling from roster)

Uses the new modular approach with efficient one-time data preparation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import optuna.visualization as vis
import yaml
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from pydantic import BaseModel

from sdm.commands.modelling.train_sdm_models import (
    simplify_boundary,
    prepare_species_training_data,
    prepare_training_data,
    _summarize_cv_scores,
    _configure_mlflow_from_config,
)
from sdm.data.loaders.vector import load_bat_data
from sdm.models.maxent.maxent_model import (
    ActivityType,
    DefaultMaxentConfig,
    create_maxent_pipeline,
    cross_validate_maxent_model,
)
from sdm.occurrence.sampling import (
    BackgroundMethod,
    TransformMethod,
)
from sdm.raster.io import load_environmental_variables
from sdm.utils.io import (
    load_model_config,
    load_project_config,
    load_boundary,
    get_tuning_config_path,
)
from sdm.utils.logging_utils import setup_logging
from sdm.types import ModelConfig, VariablesConfig, TrainingData, BackgroundConfig
from sdm.commands.modelling.utils import get_model_id
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class TuningResult(BaseModel):
    """Summary of tuning results for a single species–activity study."""

    latin_name: str
    activity_type: str
    study_name: str

    best_objective: float
    mean_cv_auc: Optional[float]
    std_cv_auc: Optional[float]

    n_features: Optional[int]
    n_features_available: int

    best_trial_number: int
    n_trials: int
    n_trials_pruned: int
    n_trials_complete: int

    stability_penalty: float
    feature_penalty: float
    correlation_penalty: Optional[float] = None
    min_presence: int
    n_cv_folds: int

    grid_size_m: float
    n_background_points: int
    background_method: str
    transform_method: str


def suggest_maxent_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest MaxEnt hyperparameters from Optuna trial.
    
    Fixed parameters (for consistency across species):
    - clamp: True (restrict suitability to training range)
    - tau: 0.5 (regularization strength)
    - transform: "cloglog" (output transform for consistent map interpretation)
    
    Tuned parameters:
    - feature_types: Selection from ["linear", "hinge", "product", "quadratic"] (at least linear must be included)
    - beta_multiplier, beta_lqp, beta_hinge, beta_threshold, beta_categorical: Regularization
    - n_hinge_features: Number of hinge features (only if hinge is selected)
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of suggested hyperparameters
    """
    # Feature types - individual selection from ["linear", "hinge", "product", "quadratic"]
    available_feature_types = ["linear", "hinge", "product", "quadratic"]
    selected_feature_types = []
    
    for feature_type in available_feature_types:
        include = trial.suggest_categorical(
            f"feature_type_{feature_type}",
            [True, False]
        )
        if include:
            selected_feature_types.append(feature_type)
    
    # Ensure at least linear is included (most basic feature type, required)
    if "linear" not in selected_feature_types:
        selected_feature_types.append("linear")
    
    feature_types = sorted(selected_feature_types)  # Sort for consistency
    
    # Beta multiplier - log uniform distribution
    beta_multiplier = trial.suggest_float("beta_multiplier", 1, 5.0, step=0.1)
    
    # Beta values for specific feature types
    beta_lqp = trial.suggest_float("beta_lqp", 0.5, 3.0, step=0.1)
    beta_hinge = trial.suggest_float("beta_hinge", 1, 4.0, step=0.1)
    beta_threshold = trial.suggest_float("beta_threshold", 1, 3.0, step=0.1)
    beta_categorical = trial.suggest_float("beta_categorical", 0.5, 3.0, step=0.1)
    
    # Number of hinge features (only if hinge is in selected feature types)
    has_hinge = "hinge" in selected_feature_types
    if has_hinge:
        n_hinge_features = trial.suggest_int("n_hinge_features", 10, 15)
    else:
        n_hinge_features = 0
    
    # Threshold not in available feature types
    n_threshold_features = 0
    
    # Fixed parameters for consistency across species (for comparable suitability maps)
    clamp = True  # Fixed: restrict suitability to training range
    tau = 0.5  # Fixed: regularization strength
    transform = "cloglog"  # Fixed: output transform for consistent map interpretation
    
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



def precompute_correlation_matrix(
    training_gdf: gpd.GeoDataFrame,
    available_features: List[str],
    sample_size: int = 3000,
) -> Optional[pd.DataFrame]:
    """Precompute correlation matrix for all available features.
    
    Args:
        training_gdf: Training GeoDataFrame with features and class column
        available_features: List of all available feature names
        sample_size: Number of points to sample for correlation calculation
        
    Returns:
        Correlation matrix DataFrame, or None if calculation fails
    """
    try:
        # Extract feature data and remove NaN values
        feature_data = training_gdf[available_features].dropna()
        
        if len(feature_data) == 0:
            logger.warning(
                "No valid data after dropping NaN values for correlation calculation"
            )
            return None
        
        # Random sample if data is larger than sample_size
        if len(feature_data) > sample_size:
            feature_data = feature_data.sample(n=sample_size, random_state=42)
        
        # Check for constant features (zero variance) - these cause correlation issues
        feature_std = feature_data.std()
        constant_features = feature_std[feature_std == 0].index.tolist()
        if constant_features:
            logger.warning(
                f"Constant features detected (zero variance): {constant_features}, "
                f"excluding from correlation calculation"
            )
            # Remove constant features from data
            feature_data = feature_data.drop(columns=constant_features)
        
        # Calculate Pearson correlation matrix
        corr_matrix = feature_data.corr(method="pearson")
        
        return corr_matrix
        
    except Exception as e:
        logger.warning(
            f"Error precomputing correlation matrix: {e}"
        )
        return None


def calculate_feature_correlation_penalty(
    selected_features: List[str],
    precomputed_corr_matrix: Optional[pd.DataFrame],
) -> float:
    """Calculate mean absolute correlation penalty for selected features.

    Uses a precomputed correlation matrix to extract the relevant subset
    for the selected features and returns mean absolute correlation
    (excluding diagonal) as a penalty metric.

    Args:
        selected_features: List of selected feature names
        precomputed_corr_matrix: Precomputed correlation matrix for all features

    Returns:
        Mean absolute correlation value (0.0 to 1.0), or 0.0 if calculation fails
    """
    # Edge case: single feature or no features - no correlation to calculate
    if len(selected_features) < 2:
        return 0.0

    # If no precomputed matrix, return 0 (shouldn't happen, but handle gracefully)
    if precomputed_corr_matrix is None:
        return 0.0

    try:
        # Filter to only selected features that exist in the correlation matrix
        available_in_matrix = [
            f for f in selected_features
            if f in precomputed_corr_matrix.columns
        ]

        if len(available_in_matrix) < 2:
            return 0.0

        # Extract submatrix for selected features
        corr_submatrix = precomputed_corr_matrix.loc[
            available_in_matrix, available_in_matrix
        ]

        # Extract upper triangle (excluding diagonal) and calculate mean absolute correlation
        abs_correlations = corr_submatrix.abs()
        upper_tri_vals = abs_correlations.to_numpy()[
            np.triu_indices_from(abs_correlations, k=1)
        ]

        if upper_tri_vals.size == 0:
            return 0.0

        mean_abs_corr = float(np.nanmean(upper_tri_vals))

        if np.isnan(mean_abs_corr) or np.isinf(mean_abs_corr):
            return 0.0

        return mean_abs_corr

    except Exception as e:
        logger.warning(
            f"Error calculating feature correlation penalty: {e}, "
            f"returning penalty 0.0"
        )
        return 0.0


def pick_features(
    trial: optuna.Trial,
    available_features: List[str],
) -> List[str]:
    """Suggest feature selection for a single activity type.
    
    Uses binary selection where Optuna suggests whether to include each feature.
    This makes feature selection part of the search space.
    
    Note: activity_type parameter is kept for API consistency but not used in
    parameter names since each study is for a single species-activity combination.
    
    Args:
        trial: Optuna trial object
        roster: Full list of available features
        activity_type: Activity type (kept for API consistency, not used in param names)
        
    Returns:
        List of selected features
    """
    selected = []
    
    # For each feature, suggest whether to include it (binary decision)
    # Parameter name is just the feature name since activity type is fixed per study
    for feature in available_features:
        include = trial.suggest_categorical(
            f"feature_{feature}",
            [True, False]
        )
        if include:
            selected.append(feature)
    
    return sorted(selected)


def eval_model(
    data: TrainingData,
    max_threads_per_model: int = 1,
    n_cv_folds: int = 3,
) -> Tuple[float, float]:
    """Evaluate a model and return CV scores and stability for tuning."""
    # Create model
    model = create_maxent_pipeline(
        feature_names=data.model_features,
        maxent_n_jobs=max_threads_per_model,
        model_config=data.maxent_config,
    )
    
    # Verify all required features are in the data
    missing_features = list(set(data.model_features) - set(data.occurrence.columns))

    if missing_features:
        raise ValueError(
            f"Missing features {missing_features} for {data.latin_name} - {data.activity_type}"
        )

    if data.occurrence["class"].isna().any():
        raise ValueError(
            f"NaN values in class column for {data.latin_name} - {data.activity_type}"
        )
    
    # Only do CV evaluation (no final model training for tuning)
    _cv_models, cv_scores = cross_validate_maxent_model(
        model=model,
        occurrence_gdf=data.occurrence,
        n_folds=n_cv_folds,
        metric_fn=roc_auc_score,
        feature_columns=data.model_features,  # Explicitly pass feature columns
    )
    
    # Get mean and std CV scores
    cv_mean, cv_std, _, _ = _summarize_cv_scores(cv_scores)
    
    return cv_mean, cv_std


def objective_train_model(
    trial: optuna.Trial,
    species_training_gdf: gpd.GeoDataFrame,
    base_model_config: ModelConfig,
    available_features: List[str],
    latin_name: str,
    activity_type: str,
    min_presence: int,
    n_cv_folds: int = 3,
    stability_penalty: float = 0.05,
    feature_penalty: float = 0.0005,
    target_features : int = 15,
    correlation_penalty: float = 0.02,
    precomputed_corr_matrix: Optional[pd.DataFrame] = None,
) -> float:
    """Optuna objective for a single species–activity combination.
    
    Args:
        trial: Optuna trial object.
        species_training_gdf: Pre-computed training data (presence + background,
            with EVs, class, and sample_weight) for this species–activity pair.
        base_model_config: Base model config.
        available_features: List of available variables for tuning / feature selection.
        latin_name: Species latin name.
        activity_type: Activity type.
        min_presence: Minimum presence records required.
        n_cv_folds: Number of CV folds.
        stability_penalty: Penalty for high standard deviation.
        feature_penalty: Penalty for high number of features.
        correlation_penalty: Penalty multiplier for feature correlation.
        precomputed_corr_matrix: Precomputed correlation matrix for all features.
        
    Returns:
        Objective value derived from CV AUC (to maximize).
    """
    # Basic integrity checks on pre-computed training data
    if len(species_training_gdf) == 0:
        raise ValueError(
            f"Trial {trial.number}: No training data for {latin_name} - {activity_type}"
        )
    
    if "class" not in species_training_gdf.columns:
        raise ValueError(
            f"Trial {trial.number}: No 'class' column for {latin_name} - {activity_type}"
        )
    
    n_presence = len(species_training_gdf[species_training_gdf["class"] == 1])
    if n_presence < min_presence:
        raise ValueError(
            f"Trial {trial.number}: Insufficient presence records ({n_presence} < {min_presence}) "
            f"for {latin_name} - {activity_type}"
        )

    # Suggest hyperparameters (only model params and features)
    maxent_params = suggest_maxent_hyperparameters(trial)
    
    # Suggest feature selection for this single activity type
    selected_features = pick_features(
        trial, available_features,
    )
    
    # Create model config
    model_config = DefaultMaxentConfig(
        **maxent_params,
        convergence_tolerance=base_model_config.maxent.convergence_tolerance,
        use_lambdas=base_model_config.maxent.use_lambdas,
        n_lambdas=base_model_config.maxent.n_lambdas,
        class_weights=base_model_config.maxent.class_weights,
    )
    
    # Verify all required features are in the data
    missing_features = [f for f in selected_features if f not in species_training_gdf.columns]
    if missing_features:
        raise ValueError(
            f"Trial {trial.number}: Missing features {missing_features} for {latin_name} - {activity_type}"
        )

    # Create training data object
    training_data = TrainingData(
        latin_name=latin_name,
        activity_type=ActivityType(activity_type),
        occurrence=species_training_gdf,
        maxent_config=model_config,
        model_features=selected_features,
    )
    
    # Train model and get scores
    mean_auc, std_auc = eval_model(
        data=training_data,
        max_threads_per_model=1,  # Use 1 thread per model for tuning
        n_cv_folds=n_cv_folds,
    )
    
    # If all CV folds failed (NaN scores), prune this trial
    # This allows Optuna to continue with other trials instead of crashing
    if np.isnan(mean_auc) or np.isnan(std_auc):
        logger.warning(
            f"Trial {trial.number}: No valid scores returned for {latin_name} - {activity_type}, "
            f"pruning trial"
        )
        raise optuna.TrialPruned(
            f"No valid CV scores (all folds failed) for {latin_name} - {activity_type}"
        )
    
    # Calculate feature penalty
    n_features = len(selected_features)
    excess = max(0, n_features - target_features)
    feature_penalty_score = (n_features * feature_penalty) + (feature_penalty * excess ** 2)
    
    # Calculate correlation penalty using precomputed matrix
    mean_abs_correlation = calculate_feature_correlation_penalty(
        selected_features=selected_features,
        precomputed_corr_matrix=precomputed_corr_matrix,
    )
    correlation_penalty_score = correlation_penalty * mean_abs_correlation
    
    # Calculate composite objective score
    stability_penalty_score = stability_penalty * std_auc  # Penalize high std (instability)
    objective_value = mean_auc - stability_penalty_score - feature_penalty_score - correlation_penalty_score
    
    
    # Store metrics and metadata as user attributes for logging / analysis
    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("std_auc", std_auc)
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("n_cv_folds", n_cv_folds)
    trial.set_user_attr("mean_abs_correlation", mean_abs_correlation)
    trial.set_user_attr("correlation_penalty_score", correlation_penalty_score)
    
    # Report intermediate value for pruning
    trial.report(objective_value, step=trial.number)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return objective_value


def write_best_config(
    study: optuna.Study,
    base_model_config: ModelConfig,
    base_variables_config: VariablesConfig,
    output_model_config_path: Path,
    output_variables_config_path: Path,
    latin_name: str,
    activity_type: str,
) -> None:
    """Write best hyperparameters to config files for a single species-activity combination.
    
    Args:
        study: Optuna study object
        base_model_config: Base model config
        base_variables_config: Base variables config
        output_model_config_path: Path to write best model config
        output_variables_config_path: Path to write best variables config
        latin_name: Species latin name
        activity_type: Activity type
    """
    best_trial = study.best_trial
    
    logger.info(f"Best trial for {latin_name} - {activity_type}: {best_trial.number}")
    logger.info(
        "Best objective value (mean AUC - stability_penalty - feature_penalty - correlation_penalty): "
        f"{best_trial.value:.4f}"
    )
    
    # Log mean AUC if available
    mean_auc = best_trial.user_attrs.get("mean_auc")
    if mean_auc is not None:
        logger.info(f"Corresponding mean CV AUC: {mean_auc:.4f}")
    
    # Extract feature types from trial parameters
    available_feature_types = ["linear", "hinge", "product", "quadratic"]
    selected_feature_types = []
    
    for feature_type in available_feature_types:
        param_name = f"feature_type_{feature_type}"
        if best_trial.params.get(param_name, False):
            selected_feature_types.append(feature_type)
    
    # Ensure at least linear is included (should always be True, but handle edge case)
    if "linear" not in selected_feature_types:
        selected_feature_types.append("linear")
    
    feature_types = sorted(selected_feature_types)  # Sort for consistency
    
    # Get n_hinge_features (only if hinge was selected)
    has_hinge = "hinge" in feature_types
    if has_hinge:
        n_hinge_features = best_trial.params.get("n_hinge_features", 10)
    else:
        n_hinge_features = 0
    
    # Threshold not in available feature types
    n_threshold_features = 0
    
    # Fixed parameters (not tuned, consistent across all models)
    clamp = True
    tau = 0.5
    transform = "cloglog"
    
    best_maxent_params = {
        "feature_types": feature_types,
        "beta_multiplier": best_trial.params["beta_multiplier"],
        "beta_lqp": best_trial.params["beta_lqp"],
        "beta_hinge": best_trial.params["beta_hinge"],
        "beta_threshold": best_trial.params["beta_threshold"],
        "beta_categorical": best_trial.params["beta_categorical"],
        "n_hinge_features": n_hinge_features,
        "n_threshold_features": n_threshold_features,
        "clamp": clamp,
        "tau": tau,
        "transform": transform,
    }
    
    # Reconstruct selected features from binary decisions
    roster = base_variables_config.variables
    selected_features = []
    for feature in roster:
        param_name = f"feature_{feature}"
        if param_name in best_trial.params and best_trial.params[param_name]:
            selected_features.append(feature)
    
    # If no features were selected (shouldn't happen, but handle gracefully)
    if len(selected_features) == 0:
        logger.warning(f"No features selected for {latin_name} - {activity_type}, using all features")
        selected_features = roster.copy()
    
    # Create model config - only include maxent section (what we're tuning)
    model_config_dict = {
        "model": {
            "maxent": {
                **best_maxent_params,
                "convergence_tolerance": base_model_config.maxent.convergence_tolerance,
                "use_lambdas": base_model_config.maxent.use_lambdas,
                "n_lambdas": base_model_config.maxent.n_lambdas,
                "class_weights": base_model_config.maxent.class_weights,
            },
        }
    }
    
    # Create variables config with just the selected features list
    # Simple structure: just a list of variables
    variables_config_dict = {
        "variables": selected_features
    }
    
    # Write configs
    with open(output_model_config_path, "w") as f:
        yaml.dump(model_config_dict, f, default_flow_style=False, sort_keys=False)
    
    with open(output_variables_config_path, "w") as f:
        yaml.dump(variables_config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Best configs written for {latin_name} - {activity_type}")


def save_tuning_plots(
    study: optuna.Study,
    output_dir: Path,
    latin_name: str,
    activity_type: str,
    precomputed_corr_matrix: Optional[pd.DataFrame] = None,
) -> None:
    """Save optimization plots for a completed study.
    
    Saves:
    - Objective function over trials
    - Mean CV AUC over trials (from user_attrs)
    - Feature selection heatmap (features on/off per trial, organized by correlation)
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save plots
        latin_name: Species latin name
        activity_type: Activity type
        precomputed_corr_matrix: Optional precomputed correlation matrix for organizing features
    """
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        logger.warning(
            f"No completed trials for {latin_name} - {activity_type}, skipping plots"
        )
        return
    
    try:
        # Plot 1: Optimization history (objective function)
        # Try using Optuna's native visualization first
        try:
            fig = vis.plot_optimization_history(study)
            fig.update_layout(
                title=f"Optimization History - {latin_name} - {activity_type}",
                xaxis_title="Trial Number",
                yaxis_title="Objective Value",
            )
            objective_plot_path = output_dir / "optimization_history.png"
            fig.write_image(str(objective_plot_path), width=800, height=600, scale=2)
            logger.debug(f"Saved optimization history plot to {objective_plot_path}")
        except Exception as e:
            # Fallback to matplotlib if plotly/kaleido is not available
            logger.debug(f"Plotly not available, using matplotlib for objective plot: {e}")
            trial_numbers = [t.number for t in completed_trials]
            objective_values = [t.value for t in completed_trials if t.value is not None]
            trial_nums_with_values = [
                t.number for t in completed_trials if t.value is not None
            ]
            
            if len(objective_values) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(trial_nums_with_values, objective_values, 'g-o', markersize=4, linewidth=1.5)
                ax.set_xlabel("Trial Number")
                ax.set_ylabel("Objective Value")
                ax.set_title(f"Optimization History - {latin_name} - {activity_type}")
                ax.grid(True, alpha=0.3)
                
                objective_plot_path = output_dir / "optimization_history.png"
                fig.savefig(objective_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.debug(f"Saved optimization history plot to {objective_plot_path}")
        
        # Plot 2: Mean CV AUC over trials
        trial_numbers = [t.number for t in completed_trials]
        mean_aucs = [
            t.user_attrs.get("mean_auc", np.nan) 
            for t in completed_trials
        ]
        
        # Filter out NaN values
        valid_data = [
            (num, auc) 
            for num, auc in zip(trial_numbers, mean_aucs) 
            if not np.isnan(auc)
        ]
        
        if len(valid_data) > 0:
            valid_nums, valid_aucs = zip(*valid_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(valid_nums, valid_aucs, 'b-o', markersize=4, linewidth=1.5)
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Mean CV AUC")
            ax.set_title(f"Mean CV AUC Over Trials - {latin_name} - {activity_type}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim((0, 1))
            
            auc_plot_path = output_dir / "mean_auc_history.png"
            fig.savefig(auc_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved mean AUC history plot to {auc_plot_path}")
        else:
            logger.warning(
                f"No valid mean_auc values found for {latin_name} - {activity_type}"
            )
        
        # Plot 3: Feature selection heatmap
        # Extract feature selections from all completed trials
        feature_selections = {}  # feature_name -> list of (trial_number, is_selected)
        trial_numbers = sorted([t.number for t in completed_trials])
        
        # Collect all feature selections
        for trial in completed_trials:
            for param_name, param_value in trial.params.items():
                if param_name.startswith("feature_"):
                    feature_name = param_name.replace("feature_", "")
                    if feature_name not in feature_selections:
                        feature_selections[feature_name] = {}
                    feature_selections[feature_name][trial.number] = bool(param_value)
        
        # Filter to only features selected in at least one trial
        features_with_selections = {
            feat: selections 
            for feat, selections in feature_selections.items()
            if any(selections.values())
        }
        
        if len(features_with_selections) > 0:
            # Determine sorting method - use correlation-based clustering if available
            used_correlation_sorting = False
            if precomputed_corr_matrix is not None and len(features_with_selections) > 2:
                # Organize by correlation using hierarchical clustering
                try:
                    from scipy.cluster.hierarchy import linkage, leaves_list
                    from scipy.spatial.distance import squareform
                    
                    # Get features that appear in both selections and correlation matrix
                    available_features = list(features_with_selections.keys())
                    features_in_corr = [
                        f for f in available_features 
                        if f in precomputed_corr_matrix.columns
                    ]
                    
                    if len(features_in_corr) >= 2:
                        # Extract correlation submatrix for selected features
                        corr_submatrix = precomputed_corr_matrix.loc[features_in_corr, features_in_corr]
                        
                        # Convert correlation to distance (1 - |correlation|)
                        # Higher correlation = lower distance
                        distance_matrix = 1 - np.abs(corr_submatrix.values)
                        
                        # Convert to condensed distance matrix for linkage
                        condensed_distances = squareform(distance_matrix, checks=False)
                        
                        # Perform hierarchical clustering
                        linkage_matrix = linkage(condensed_distances, method='ward')
                        
                        # Get feature order from dendrogram (leaves)
                        leaf_order = leaves_list(linkage_matrix)
                        sorted_features_by_corr = [features_in_corr[i] for i in leaf_order]
                        
                        # Add any features not in correlation matrix at the end
                        features_not_in_corr = [f for f in available_features if f not in features_in_corr]
                        sorted_features = sorted_features_by_corr + features_not_in_corr
                        used_correlation_sorting = True
                        
                        logger.debug(f"Organized {len(features_in_corr)} features by correlation")
                    else:
                        # Fall back to frequency sorting if not enough features in correlation matrix
                        feature_selection_counts = {
                            feat: sum(1 for selected in selections.values() if selected)
                            for feat, selections in features_with_selections.items()
                        }
                        sorted_features = sorted(
                            features_with_selections.keys(),
                            key=lambda f: feature_selection_counts[f],
                            reverse=True
                        )
                except ImportError:
                    logger.debug("scipy not available, falling back to frequency sorting")
                    # Fall back to frequency sorting
                    feature_selection_counts = {
                        feat: sum(1 for selected in selections.values() if selected)
                        for feat, selections in features_with_selections.items()
                    }
                    sorted_features = sorted(
                        features_with_selections.keys(),
                        key=lambda f: feature_selection_counts[f],
                        reverse=True
                    )
                except Exception as e:
                    logger.debug(f"Error in correlation-based sorting: {e}, falling back to frequency sorting")
                    # Fall back to frequency sorting
                    feature_selection_counts = {
                        feat: sum(1 for selected in selections.values() if selected)
                        for feat, selections in features_with_selections.items()
                    }
                    sorted_features = sorted(
                        features_with_selections.keys(),
                        key=lambda f: feature_selection_counts[f],
                        reverse=True
                    )
            else:
                # Sort features by selection frequency (most common at top)
                feature_selection_counts = {
                    feat: sum(1 for selected in selections.values() if selected)
                    for feat, selections in features_with_selections.items()
                }
                sorted_features = sorted(
                    features_with_selections.keys(),
                    key=lambda f: feature_selection_counts[f],
                    reverse=True
                )
            
            # Limit to top 50 features to keep plot readable
            max_features = 50
            if len(sorted_features) > max_features:
                sorted_features = sorted_features[:max_features]
                logger.debug(
                    f"Limiting feature selection plot to top {max_features} most selected features "
                    f"(out of {len(feature_selections)} total)"
                )
            
            # Build matrix: rows are features, columns are trials
            matrix = []
            for feature in sorted_features:
                row = [
                    features_with_selections[feature].get(trial_num, False)
                    for trial_num in trial_numbers
                ]
                matrix.append(row)
            
            # Create heatmap
            # Use fixed width for readability, height scales with number of features
            fig_width = 14  # Fixed width for consistent readability
            fig_height = max(8, len(sorted_features) * 0.4)  # Height scales with features, minimum 8
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Convert boolean matrix to int (True=1, False=0) for imshow
            matrix_array = np.array(matrix, dtype=int)
            
            # Create heatmap: 1 = selected (filled), 0 = not selected (empty)
            im = ax.imshow(matrix_array, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
            
            # Set ticks
            ax.set_xticks(range(len(trial_numbers)))
            ax.set_xticklabels([str(n) for n in trial_numbers], rotation=45, ha='right')
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            
            # Labels
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Feature")
            title_suffix = "organized by correlation" if used_correlation_sorting else "sorted by selection frequency"
            ax.set_title(f"Feature Selection Heatmap - {latin_name} - {activity_type}\n"
                        f"(Features {title_suffix}, showing top {len(sorted_features)} most selected)")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.set_ticklabels(['Not Selected', 'Selected'])
            
            # Adjust layout
            plt.tight_layout()
            
            feature_plot_path = output_dir / "feature_selection_heatmap.png"
            fig.savefig(feature_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved feature selection heatmap to {feature_plot_path}")
        else:
            logger.warning(
                f"No feature selections found for {latin_name} - {activity_type}"
            )
            
    except Exception as e:
        logger.warning(
            f"Error saving plots for {latin_name} - {activity_type}: {e}. "
            f"Continuing without plots."
        )


def save_trial_data_csv(
    study: optuna.Study,
    output_dir: Path,
    latin_name: str,
    activity_type: str,
    stability_penalty: float,
    feature_penalty: float,
    target_features: int,
    correlation_penalty: float,
) -> None:
    """Write a CSV of trial data (identifiers + objective and its components) for plotting.

    Uses existing trial.value and user_attrs; derives stability_penalty_score,
    excess_features, and feature_penalty_score from constants. No MaxEnt hyperparameters.
    """
    rows = []
    for t in study.trials:
        mean_auc = t.user_attrs.get("mean_auc")
        std_auc = t.user_attrs.get("std_auc")
        n_features = t.user_attrs.get("n_features")
        mean_abs_corr = t.user_attrs.get("mean_abs_correlation")
        corr_penalty_score = t.user_attrs.get("correlation_penalty_score")

        if std_auc is not None and not (np.isnan(std_auc)):
            stability_penalty_score = stability_penalty * std_auc
        else:
            stability_penalty_score = None

        if n_features is not None:
            excess_features = max(0, n_features - target_features)
            feature_penalty_score = (n_features * feature_penalty) + (
                feature_penalty * excess_features**2
            )
        else:
            excess_features = None
            feature_penalty_score = None

        rows.append({
            "latin_name": latin_name,
            "activity_type": activity_type,
            "study_name": study.study_name,
            "trial_number": t.number,
            "state": t.state.name,
            "objective_value": t.value,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "stability_penalty": stability_penalty,
            "stability_penalty_score": stability_penalty_score,
            "feature_penalty": feature_penalty,
            "target_features": target_features,
            "n_features": n_features,
            "excess_features": excess_features,
            "feature_penalty_score": feature_penalty_score,
            "mean_abs_correlation": mean_abs_corr,
            "correlation_penalty": correlation_penalty,
            "correlation_penalty_score": corr_penalty_score,
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "trial_data.csv"
    df.to_csv(csv_path, index=False)
    logger.debug(f"Saved trial data CSV to {csv_path}")


def tune_hyperparameters(
    project_config_path: Path,
    model_config_path: Path,
    variables_config_path: Path,
    bats_file: Path,
    ev_file: Path,
    output_dir: Path,
    n_trials: int = 50,
    stability_penalty: float = 0.25,
    feature_penalty: float = 0.001,
    target_features: int = 10,
    correlation_penalty: float = 0.05,
    correlation_sample_size: int = 10_000,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = -1,
    n_cv_folds: int = 3,
    verbose: bool = False,
    # Background point generation parameters (fixed, not tuned)
    n_background_points: int = 4000,
    background_method: BackgroundMethod = BackgroundMethod.CONTRAST,
    background_value: float = 0.00,
    sigma: float = 6.5,
    transform_method: TransformMethod = TransformMethod.PRESENCE,
    # Species-specific processing parameters (fixed, not tuned)
    grid_size_m: float = 2000,
    d_min: float = 500,
    d_max: float = np.inf,
    sample_weight_n_neighbors: int = 10,
) -> Optional[optuna.Study]:
    """Run hyperparameter tuning using Optuna with the new modular approach.
    
    This function:
    - Does expensive operations once (convert EVs, annotate points, generate background)
    - Uses grid sampling to subset data for faster tuning
    - Only tunes model parameters and feature selection (not background points)
    
    Args:
        project_config_path: Path to project config
        model_config_path: Path to base model config
        variables_config_path: Path to base variables config
        bats_file: Path to bat occurrence data
        ev_file: Path to environmental variables
        output_dir: Output directory for tuning results
        n_trials: Number of Optuna trials to run
        stability_penalty: Penalty for high standard deviation
        feature_penalty: Penalty for high number of features
        correlation_penalty: Penalty multiplier for feature correlation
        correlation_sample_size: Number of points to sample for correlation calculation
        species: List of species to tune (optional)
        activity_types: List of activity types to tune (optional)
        study_name: Name for Optuna study (optional)
        storage: Optuna storage URL (optional, for distributed tuning)
        n_jobs: Number of parallel jobs for running trials (default: 1, sequential)
        n_cv_folds: Number of CV folds for tuning (default: 2, faster than 3)
        verbose: Enable verbose logging
        n_background_points: Number of background points per activity (fixed)
        background_method: Background generation method (fixed)
        background_value: Background value (fixed)
        sigma: Gaussian smoothing sigma (fixed)
        transform_method: Transform method (fixed)
        grid_size_m: Grid size for training data sampling (fixed)
        d_min: Minimum distance for background filtering (fixed)
        d_max: Maximum distance for background filtering (fixed)
        sample_weight_n_neighbors: Number of neighbors for sample weighting (fixed)
        n_cv_folds: Number of CV folds for evaluation (aligned with training)
        n_jobs: Number of parallel trials (None = sequential, 1 = sequential, >1 = parallel)
        
    Returns:
        Optuna study object
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("=== Starting Hyperparameter Tuning (Modular Approach) ===")
    
    # Load base configs
    project_config = load_project_config(project_config_path)
    base_model_config = load_model_config(model_config_path)
    
    # Get background config from model config (use provided values as overrides)
    bg_config = base_model_config.background or BackgroundConfig()
    
    # Use function parameters if provided, otherwise use config values (which have defaults)
    # BackgroundConfig has default enum values, so background_method and transform_method are never None
    n_background_points = n_background_points if n_background_points != 4000 else bg_config.n_background_points
    background_method = background_method if background_method != BackgroundMethod.CONTRAST else bg_config.background_method
    background_value = background_value if background_value != 0.00 else bg_config.background_value
    sigma = sigma if sigma != 6.5 else bg_config.sigma
    transform_method = transform_method if transform_method != TransformMethod.PRESENCE else bg_config.transform_method
    
    # Type assertions to satisfy linter (these are guaranteed by default values)
    assert background_method is not None, "background_method should be set by BackgroundConfig"
    assert transform_method is not None, "transform_method should be set by BackgroundConfig"
    
    logger.info(
        f"Background point config: n={n_background_points}, method={background_method.value}, "
        f"value={background_value}, sigma={sigma}, transform={transform_method.value}"
    )
    
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure MLflow
    _configure_mlflow_from_config(project_config)
    
    # ========================================================================
    # LOAD CORE DATA (IN-MEMORY OBJECTS)
    # ========================================================================
    logger.info("Loading input data for hyperparameter tuning...")
    
    # Load raw bat occurrence data (presence-only)
    occurrence_gdf = load_bat_data(bats_file)
    occurrence_gdf = occurrence_gdf[occurrence_gdf.accuracy <= 100]
    
    # Optional filtering of occurrence records
    if species is not None:
        logger.info(f"Filtering to species: {', '.join(species)}")
        occurrence_gdf = occurrence_gdf[occurrence_gdf.latin_name.isin(species)]
    
    if activity_types is not None:
        logger.info(f"Filtering to activity types: {', '.join(activity_types)}")
        occurrence_gdf = occurrence_gdf[occurrence_gdf.activity_type.isin(activity_types)]
    
    # Load boundary
    boundary_path = Path(project_config.paths.boundary)
    boundary = load_boundary(
        filepath=boundary_path, buffer_distance=0, target_crs=27700
    )
    boundary = simplify_boundary(boundary, tolerance=100)
    
    # Load environmental variables
    evs_to_model, _ = load_environmental_variables(ev_file)
    
    # ========================================================================
    # ONE-TIME STUDY DATA PREPARATION (IN-MEMORY)
    # ========================================================================
    (
        presence_with_evs_gdf,
        background_with_evs_gdf,
        available_features,
    ) = prepare_training_data(
        occurrence_gdf=occurrence_gdf,
        boundary=boundary,
        evs_to_model=evs_to_model,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        transform_method=transform_method,
    )

    base_variables_config = VariablesConfig(
        variables=available_features,
    )
    # ========================================================================
    # OPTUNA TUNING - PER SPECIES-ACTIVITY COMBINATION
    # ========================================================================
    parent_run_name = f"Hyperparameter_Tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _parent_run = mlflow.start_run(run_name=parent_run_name)
    logger.info(f"Started MLflow parent run: {parent_run_name}")
    
    # Aggregate results across all combinations
    all_results: List[TuningResult] = []
    
    try:
        # Log tuning configuration
        mlflow.log_params({
            "tuning_n_trials": n_trials,
            "tuning_approach": "per_species_activity",
        })
        if species:
            mlflow.log_param("tuning_species", ",".join(species))
        if activity_types:
            mlflow.log_param("tuning_activity_types", ",".join(activity_types))
        
        # Get all species-activity combinations present in the (annotated) data
        combinations_df = (
            presence_with_evs_gdf[["latin_name", "activity_type"]]
            .drop_duplicates()
        )
        combinations = list(combinations_df.itertuples(index=False, name=None))
        logger.info(f"Tuning {len(combinations)} species-activity combinations...")
        
        # Loop over each species-activity combination
        for idx, (latin_name, activity_type) in enumerate(combinations, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Tuning {idx}/{len(combinations)}: {latin_name} - {activity_type}")
            logger.info(f"{'='*80}")
            
            # Check if we have data for this combination
            species_presence_with_evs_gdf = presence_with_evs_gdf[
                (presence_with_evs_gdf["latin_name"] == latin_name) &
                (presence_with_evs_gdf["activity_type"] == activity_type)
            ]
            n_presence = len(species_presence_with_evs_gdf)
            
            if n_presence < base_model_config.sampling.min_presence:
                logger.warning(
                    f"Skipping {latin_name} - {activity_type}: "
                    f"insufficient presence records ({n_presence} < {base_model_config.sampling.min_presence})"
                )
                continue
            
            # Prepare training data once for this combination (shared across trials)
            species_training_gdf = prepare_species_training_data(
                presence_data=presence_with_evs_gdf,
                background_data=background_with_evs_gdf,
                latin_name=latin_name,
                activity_type=activity_type,
                ev_columns=available_features,
                grid_size_m=grid_size_m,
                d_min=d_min,
                d_max=d_max,
                sample_weight_n_neighbors=sample_weight_n_neighbors,
                random_state=42,
            )

            # Basic sanity checks before starting tuning for this combination
            if len(species_training_gdf) == 0:
                logger.warning(
                    f"Skipping {latin_name} - {activity_type}: "
                    f"no training data generated by prepare_species_training_data"
                )
                continue

            if "class" not in species_training_gdf.columns:
                logger.warning(
                    f"Skipping {latin_name} - {activity_type}: "
                    f"missing 'class' column in training data"
                )
                continue

            n_presence_training = len(species_training_gdf[species_training_gdf["class"] == 1])
            if n_presence_training < base_model_config.sampling.min_presence:
                logger.warning(
                    f"Skipping {latin_name} - {activity_type}: "
                    f"only {n_presence_training} presence records in training data "
                    f"(minimum {base_model_config.sampling.min_presence} required)"
                )
                continue

            # Create study name for this combination
            study_name_combination = f"{study_name or 'sdm_tuning'}_{latin_name}_{activity_type}".replace(" ", "_")
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)
            
            study = optuna.create_study(
                study_name=study_name_combination,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True,
            )
            
            # Precompute correlation matrix once for this species-activity combination
            logger.debug(f"Precomputing correlation matrix for {latin_name} - {activity_type}...")
            precomputed_corr_matrix = precompute_correlation_matrix(
                training_gdf=species_training_gdf,
                available_features=available_features,
                sample_size=correlation_sample_size,
            )
            
            # Create objective function for this combination
            def objective_fn(trial: optuna.Trial):
                return objective_train_model(
                    trial,
                    species_training_gdf=species_training_gdf,
                    base_model_config=base_model_config,
                    available_features=available_features,
                    latin_name=latin_name,
                    activity_type=activity_type,
                    min_presence=base_model_config.sampling.min_presence,
                    n_cv_folds=n_cv_folds,
                    stability_penalty=stability_penalty,
                    feature_penalty=feature_penalty,
                    target_features=target_features,
                    correlation_penalty=correlation_penalty,
                    precomputed_corr_matrix=precomputed_corr_matrix,
                )
            
            # Run optimization (Optuna will handle initialization with random trials)
            n_jobs_optuna = n_jobs if n_jobs is not None else 1
            logger.info(f"Starting optimization with {n_trials} trials (n_jobs={n_jobs_optuna}, n_cv_folds={n_cv_folds})...")
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                n_jobs=n_jobs_optuna,
                show_progress_bar=verbose,
            )
            
            # Check if any trials completed successfully
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            
            if n_completed > 0:
                # Write best configs for this combination
                combination_dir = get_tuning_config_path(output_dir, get_model_id([latin_name, activity_type]))
                combination_dir.mkdir(parents=True, exist_ok=True)
                
                # Use consistent file names (not "best_*") for easy loading
                model_config_path = combination_dir / "model_config.yml"
                variables_config_path = combination_dir / "variables_config.yml"
                
                write_best_config(
                    study,
                    base_model_config,
                    base_variables_config,
                    model_config_path,
                    variables_config_path,
                    latin_name,
                    activity_type,
                )
                
                # Save tuning plots
                save_tuning_plots(
                    study,
                    combination_dir,
                    latin_name,
                    activity_type,
                    precomputed_corr_matrix=precomputed_corr_matrix,
                )
                save_trial_data_csv(
                    study,
                    combination_dir,
                    latin_name,
                    activity_type,
                    stability_penalty=stability_penalty,
                    feature_penalty=feature_penalty,
                    target_features=target_features,
                    correlation_penalty=correlation_penalty,
                )
                # Store results for this study
                best_trial = study.best_trial
                mean_cv_auc = best_trial.user_attrs.get("mean_auc")
                std_cv_auc = best_trial.user_attrs.get("std_auc")
                n_features = best_trial.user_attrs.get("n_features")

                all_results.append(
                    TuningResult(
                        latin_name=latin_name,
                        activity_type=activity_type,
                        study_name=study.study_name,
                        best_objective=float(study.best_value),
                        mean_cv_auc=float(mean_cv_auc) if mean_cv_auc is not None else None,
                        std_cv_auc=float(std_cv_auc) if std_cv_auc is not None else None,
                        n_features=int(n_features) if n_features is not None else None,
                        n_features_available=len(available_features),
                        best_trial_number=best_trial.number,
                        n_trials=len(study.trials),
                        n_trials_pruned=len(
                            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                        ),
                        n_trials_complete=n_completed,
                        stability_penalty=stability_penalty,
                        feature_penalty=feature_penalty,
                        correlation_penalty=correlation_penalty,
                        min_presence=base_model_config.sampling.min_presence,
                        n_cv_folds=n_cv_folds,
                        grid_size_m=grid_size_m,
                        n_background_points=n_background_points,
                        background_method=background_method.value,
                        transform_method=transform_method.value,
                    )
                )
            else:
                # All trials were pruned - skip writing configs
                logger.warning(
                    f"Skipping config write for {latin_name} - {activity_type}: "
                    f"all {len(study.trials)} trials were pruned (no valid models)"
                )
            

        logger.info("\n=== Hyperparameter Tuning Complete ===")
        logger.info(f"Tuned {len(all_results)} species-activity combinations")
        if all_results:
            mean_best_auc = np.mean([r.best_objective for r in all_results])
            logger.info(f"Mean best objective value: {mean_best_auc:.4f}")

            # Write summary CSV with one row per species–activity model
            results_df = pd.DataFrame([r.model_dump() for r in all_results])
            csv_path = output_dir / "tuning_results.csv"
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Wrote tuning summary CSV to {csv_path}")

        logger.info(f"Results logged to MLflow parent run: {parent_run_name}")
        
    finally:
        # End parent run
        mlflow.end_run()
