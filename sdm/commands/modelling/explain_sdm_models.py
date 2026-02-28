"""
SHAP-based model interpretability for SDM models.

Calculates SHAP values for trained SDM models and generates interpretability
plots (feature importance bar plots and dependence plots).
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
import pickle
import shutil
import yaml

import numpy as np
import pandas as pd
import xarray as xr
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sdm.utils.logging_utils import setup_logging
from sdm.raster.io import load_environmental_variables
from sdm.models.core.feature_subsetter import FeatureSubsetter

logger = logging.getLogger(__name__)


def load_model_index(models_dir: Path) -> pd.DataFrame:
    """Load the index of available models."""
    index_path = models_dir / "model_results.csv"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Model index not found at {index_path}")
    
    return pd.read_csv(index_path)


def filter_models(
    model_index: pd.DataFrame,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter models based on species and activity type criteria."""
    filtered = model_index.copy()
    
    if species:
        filtered = filtered[filtered.latin_name.isin(species)]
    if activity_types:
        filtered = filtered[filtered.activity_type.isin(activity_types)]
        
    return filtered


def load_model(model_path: Path) -> Any:
    """Load a pickled model from disk."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def sample_points_from_xarray_dataset(
    evs_dataset: xr.Dataset,
    n_samples: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Efficiently sample points from xarray Dataset to pandas DataFrame.
    
    This avoids the slow full GeoDataFrame conversion by sampling directly
    from the xarray Dataset using isel on stacked spatial dimensions.
    
    Args:
        evs_dataset: xarray Dataset with environmental variables
        n_samples: Number of points to sample
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled points (columns are EV names, rows are samples)
    """
    # Get data variables
    data_vars = list(evs_dataset.data_vars)
    if not data_vars:
        raise ValueError("No data variables found in environmental variables dataset")
    
    # Stack spatial dimensions to create a flat array for efficient indexing
    stacked = evs_dataset.stack(point=('x', 'y'))
    
    # Get total number of valid points (non-NaN)
    # Check which points have all valid values across all variables
    valid_mask = stacked.to_array().notnull().all(dim='variable')
    valid_indices = np.where(valid_mask.values)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("No valid points found in environmental variables dataset")
    
    # Sample from valid indices
    n_samples = min(n_samples, len(valid_indices))
    rng = np.random.RandomState(random_state)
    sampled_indices = rng.choice(valid_indices, size=n_samples, replace=False)
    
    # Select sampled points using isel
    sampled_stacked = stacked.isel(point=sampled_indices)
    
    # Convert to DataFrame
    # For a stacked Dataset, to_dataframe() creates a DataFrame with MultiIndex
    # The coordinates x and y are already available as separate coordinate arrays
    # and will be included as columns in the DataFrame
    df = sampled_stacked.to_dataframe()
    
    # The x and y coordinates are already columns (from coordinate arrays)
    # We just need to drop the MultiIndex, not convert it to columns
    df = df.reset_index(drop=True)
    
    # The DataFrame should now have variables as columns, plus x, y, spatial_ref
    # Drop coordinate columns and keep only EV columns
    ev_columns = [col for col in df.columns if col in data_vars]
    if not ev_columns:
        # Fallback: exclude known coordinate columns
        ev_columns = [col for col in df.columns if col not in ['x', 'y', 'point', 'spatial_ref']]
    
    df = df[ev_columns]
    
    # Drop any remaining NaN rows (shouldn't be any, but just in case)
    df = df.dropna()
    
    return df


def extract_model_features(model: Any) -> List[str]:
    """
    Extract feature names from a model's FeatureSubsetter step.
    
    Args:
        model: Trained sklearn Pipeline with FeatureSubsetter step
        
    Returns:
        List of feature names used by the model
    """
    # Use dictionary-style access to get the feature_selection step (preferred method)
    try:
        feature_subsetter = model["feature_selection"]
        if hasattr(feature_subsetter, 'feature_names'):
            return feature_subsetter.feature_names
    except (KeyError, TypeError):
        pass
    
    # Fallback: try named_steps attribute
    if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:
        feature_subsetter = model.named_steps['feature_selection']
        if hasattr(feature_subsetter, 'feature_names'):
            return feature_subsetter.feature_names
    
    # Fallback: iterate through steps
    if hasattr(model, 'steps'):
        feature_subsetter = next(
            (step[1] for step in model.steps if isinstance(step[1], FeatureSubsetter)),
            None
        )
        if feature_subsetter:
            return feature_subsetter.feature_names
    
    # Fallback: try to get feature names from the model itself
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    
    raise ValueError("Could not extract feature names from model")


def compute_shap_for_model(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    n_background: int = 100,
    n_explain: int = 200,
    random_state: int = 42,
    positive_class: int = 1
) -> Tuple[shap.Explainer, shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for a single model.

    Args:
        model: Fitted sklearn Pipeline with predict_proba
        X: DataFrame of input features
        feature_names: List of feature names used by the model
        n_background: Number of background samples for SHAP explainer
        n_explain: Number of points to explain
        random_state: Random seed
        positive_class: Index of positive class in predict_proba output
        
    Returns:
        Tuple of (explainer, shap_values, X_explain)
    """
    # Ensure we only use the requested features, in a consistent order
    # Verify all requested features exist in X
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Features {missing} not found in input DataFrame. Available: {list(X.columns)}")
    
    # Subset to only the model's features
    X = X[list(feature_names)].copy()
    
    # Verify we have the correct number of features
    assert X.shape[1] == len(feature_names), \
        f"Expected {len(feature_names)} features, got {X.shape[1]}"
    
    # Background sample
    n_background = min(n_background, len(X))
    background = X.sample(n=n_background, random_state=random_state)
    
    # Prediction function returning probability for positive class
    def predict_fn(data):
        proba = model.predict_proba(data)
        return proba[:, positive_class]
    
    # Create SHAP explainer with permutation algorithm
    explainer = shap.Explainer(
        predict_fn,
        background,
        algorithm="permutation",
        model_output="probability",
    )
    
    # Data to explain
    n_explain = min(n_explain, len(X))
    X_explain = X.sample(n=n_explain, random_state=random_state + 1)
    
    # Calculate SHAP values
    shap_values = explainer(X_explain)
    
    
    return explainer, shap_values, X_explain


def plot_filtered_shap_dependence(
    feature_name: str,
    X_df: pd.DataFrame,
    shap_values: shap.Explanation,
    p_low: float = 1,
    p_high: float = 99,
    interaction_index: Optional[str] = None,
    fig=None,
    ax=None,
    figsize: Tuple[int, int] = (6, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    SHAP dependence plot filtered to percentile range.

    Args:
        feature_name: Feature to plot
        X_df: Input features DataFrame
        shap_values: SHAP Explanation object
        p_low, p_high: Percentile bounds for trimming
        interaction_index: Feature used for colour (None = same feature)
        fig, ax: Matplotlib Figure/Axes (optional)
        figsize: Figure size if creating new figure
        
    Returns:
        Tuple of (fig, ax)
    """
    # Compute percentile bounds
    low, high = np.percentile(X_df[feature_name], [p_low, p_high])
    
    # Filter rows to percentile limits
    mask = (X_df[feature_name] >= low) & (X_df[feature_name] <= high)
    X_filtered = X_df.loc[mask]
    shap_filtered = shap_values.values[mask, :]
    
    # Create fig/ax only if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if fig is None:
            fig = ax.figure
    
    # Plot dependence (pass array directly, not Explanation object)
    shap.dependence_plot(
        feature_name,
        shap_filtered,
        X_filtered,
        interaction_index=interaction_index,
        ax=ax,
        show=False,
    )
    
    return fig, ax


def generate_shap_plots(
    model_id: str,
    latin_name: str,
    activity_type: str,
    shap_values: shap.Explanation,
    X_explain: pd.DataFrame,
    model_output_dir: Path,
    top_features: Optional[int] = None
) -> Dict[str, Path]:
    """
    Generate SHAP plots for a single model.
    
    Args:
        model_id: Unique identifier for the model
        latin_name: Species Latin name
        activity_type: Activity type
        shap_values: SHAP Explanation object
        X_explain: DataFrame of explained points
        model_output_dir: Directory to save plots (must exist and be cleared)
        top_features: Number of top features for dependence plots (None = all features)
        
    Returns:
        Dictionary mapping plot type to file path
    """
    plot_paths = {}
    
    # Feature importance bar plot
    # Get the actual number of features from SHAP values
    n_features = shap_values.values.shape[1] if hasattr(shap_values, 'values') else len(X_explain.columns)
    # Display all features (or up to a reasonable max)
    max_display = n_features
    
    importance_path = model_output_dir / "shap_importance.png"
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance - {latin_name} - {activity_type}")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['importance'] = importance_path
    logger.debug(f"Saved importance plot to {importance_path} (showing {max_display} of {n_features} features)")
    
    # Get feature names
    feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else X_explain.columns.tolist()
    
    # Determine which features to plot
    if top_features is None:
        # Plot all features
        features_to_plot = feature_names
        logger.debug(f"Generating dependence plots for all {len(features_to_plot)} features")
    else:
        # Get top features by mean absolute SHAP value
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-top_features:][::-1]
        features_to_plot = [feature_names[i] for i in top_indices]
        logger.debug(f"Generating dependence plots for top {len(features_to_plot)} of {len(feature_names)} features")
    
    # Dependence plots for selected features
    for i, feature_name in enumerate(features_to_plot):
        # Replace special characters in feature name for filename
        safe_feature_name = feature_name.replace("/", "_").replace("\\", "_")
        dep_path = model_output_dir / f"shap_dependence_{safe_feature_name}.png"
        fig, ax = plot_filtered_shap_dependence(
            feature_name,
            X_explain,
            shap_values,
            p_low=5,
            p_high=95,
            figsize=(8, 4)
        )
        ax.set_title(f"SHAP Dependence - {feature_name}")
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths[f'dependence_{feature_name}'] = dep_path
        logger.debug(f"Saved dependence plot to {dep_path}")
    
    return plot_paths


def write_shap_scores_yaml(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: Path
) -> None:
    """
    Write SHAP scores to a YAML file in variables_config.yml format.
    
    Calculates mean absolute SHAP values per feature, sorts them in descending order,
    and writes to YAML file with 4 decimal places.
    
    Args:
        shap_values: SHAP Explanation object containing SHAP values
        feature_names: List of feature names (must match SHAP values order)
        output_path: Path where YAML file should be written
    """
    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create dictionary mapping feature names to SHAP scores
    shap_scores_dict = {
        feature_name: float(round(score, 4))
        for feature_name, score in zip(feature_names, mean_abs_shap)
    }
    
    # Sort by SHAP score in descending order
    sorted_scores = dict(sorted(shap_scores_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Create YAML structure
    yaml_data = {
        'variables': sorted_scores
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"Wrote SHAP scores YAML to {output_path}")


def process_single_model(
    row: pd.Series,
    models_dir: Path,
    ev_df: pd.DataFrame,
    output_dir: Path,
    n_background: int,
    n_explain: int,
    top_features: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Process a single model: load, compute SHAP, generate plots.
    
    This function is designed to be called in parallel.
    
    Args:
        row: DataFrame row with model information
        models_dir: Directory containing models
        ev_df: DataFrame of environmental variables
        output_dir: Directory for output plots
        n_background: Number of background samples
        n_explain: Number of points to explain
        top_features: Number of top features for dependence plots
        random_state: Random seed
        
    Returns:
        Dictionary with results for this model
    """
    latin_name = row.latin_name
    activity_type = row.activity_type
    model_id = f"{latin_name}_{activity_type}"
    
    try:
        logger.info(f"Processing {model_id}...")
        
        # Load model
        # model_path in CSV is relative to project root, use it directly
        model_path = Path(row.model_path)
        model = load_model(model_path)
        
        # Extract feature names from model's FeatureSubsetter
        feature_names = extract_model_features(model)
        logger.info(f"Model {model_id} uses {len(feature_names)} features: {feature_names}")
        
        # Check for missing features
        missing_features = set(feature_names) - set(ev_df.columns)
        if missing_features:
            logger.warning(
                f"Model {model_id} requires features not in EV data: {missing_features}"
            )
            feature_names = [f for f in feature_names if f in ev_df.columns]
        
        if not feature_names:
            raise ValueError(f"No valid features found for model {model_id}")
        
        # Subset EV DataFrame to only include model's features (like in notebook)
        # Ensure we only use features that exist in both the model and EV data
        available_features = [f for f in feature_names if f in ev_df.columns]
        if len(available_features) != len(feature_names):
            missing = set(feature_names) - set(available_features)
            logger.warning(f"Model {model_id} has {len(missing)} features not in EV data: {missing}")
        
        ev_df_subset = ev_df[available_features].copy()
        logger.info(f"Subsetted EV data to {len(available_features)} features for model {model_id}: {available_features}")
        
        # Verify the subset only contains model features
        assert set(ev_df_subset.columns) == set(available_features), \
            f"EV subset columns don't match model features! Got {set(ev_df_subset.columns)}, expected {set(available_features)}"
        
        # Compute SHAP values (only for model's features)
        explainer, shap_values, X_explain = compute_shap_for_model(
            model=model,
            X=ev_df_subset,
            feature_names=available_features,
            n_background=n_background,
            n_explain=n_explain,
            random_state=random_state
        )
        
        # Verify SHAP values only have the expected number of features
        n_shap_features = shap_values.values.shape[1] if hasattr(shap_values, 'values') else len(X_explain.columns)
        if n_shap_features != len(available_features):
            logger.warning(f"SHAP values have {n_shap_features} features but model uses {len(available_features)}")
        else:
            logger.debug(f"SHAP values correctly computed for {n_shap_features} features")
        
        # Create subdirectory for this species + activity type
        # Replace spaces with underscores for directory names
        safe_latin_name = latin_name.replace(" ", "_")
        safe_activity_type = activity_type.replace(" ", "_")
        model_output_dir = output_dir / safe_latin_name / safe_activity_type
        
        # Clear directory if it exists, then create it
        if model_output_dir.exists():
            shutil.rmtree(model_output_dir)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get feature names from SHAP values (in correct order)
        shap_feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else available_features
        
        # Write SHAP scores to YAML file
        yaml_path = model_output_dir / "variables_with_shap_scores.yml"
        write_shap_scores_yaml(
            shap_values=shap_values,
            feature_names=shap_feature_names,
            output_path=yaml_path
        )
        logger.info(f"Wrote SHAP scores YAML to {yaml_path}")
        
        # Generate plots (pass top_features directly - None means all features)
        plot_paths = generate_shap_plots(
            model_id=model_id,
            latin_name=latin_name,
            activity_type=activity_type,
            shap_values=shap_values,
            X_explain=X_explain,
            model_output_dir=model_output_dir,
            top_features=top_features  # None = all features, int = top N features
        )
        
        logger.info(f"✓ Completed {model_id}")
        
        return {
            'latin_name': latin_name,
            'activity_type': activity_type,
            'model_id': model_id,
            'success': True,
            'n_features': len(feature_names),
            'n_explain': len(X_explain),
            'plot_paths': plot_paths,
            'yaml_path': str(yaml_path),
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Failed to process {model_id}: {e}", exc_info=True)
        return {
            'latin_name': latin_name,
            'activity_type': activity_type,
            'model_id': model_id,
            'success': False,
            'n_features': None,
            'n_explain': None,
            'plot_paths': {},
            'error': str(e)
        }


def explain_sdm_models(
    ev_path: Path = Path("data/evs/evs-to-model.tif"),
    models_dir: Path = Path("data/sdm_models"),
    output_dir: Path = Path("data/sdm_predictions"),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    n_ev_pool: int = 10_000,
    n_explain: int = 200,
    n_background: int = 1000,
    top_features: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate SHAP values for SDM models and generate interpretability plots.

    Samples a pool of points from the EV raster, then for each model samples
    background and explain sets from that pool (same process regardless of
    pool or sample sizes).

    Args:
        ev_path: Path to environmental variables raster
        models_dir: Directory containing trained models
        output_dir: Directory for output plots
        species: Optional list of species to explain (Latin names)
        activity_types: Optional list of activity types to explain
        n_ev_pool: Number of points to sample from the EV raster (shared pool
            for all models).
        n_explain: Number of points to explain
        n_background: Number of background samples for the SHAP explainer
        top_features: Number of top features for dependence plots (None = all)
        n_jobs: Number of parallel workers (None = auto)
        verbose: Enable verbose logging

    Returns:
        DataFrame containing explanation results
    """
    setup_logging(level=logging.INFO, verbose=verbose)
    
    logger.info("Starting SHAP explanation pipeline...")
    
    # Load model index
    logger.debug("Loading model index...")
    model_index = load_model_index(models_dir)
    logger.info(f"Found {len(model_index)} models in index")
    
    # Filter models
    filtered_index = filter_models(model_index, species, activity_types)
    logger.info(f"Selected {len(filtered_index)} models for explanation")
    
    if len(filtered_index) == 0:
        logger.warning("No models match the specified criteria")
        raise ValueError("No models match the specified criteria")
    
    # Load environmental variables as xarray Dataset
    logger.debug("Loading environmental variables...")
    evs_dataset, _ = load_environmental_variables(ev_path)
    
    # Sample a pool from the EV raster (once for all models); each model
    # samples its background and explain sets from this pool.
    logger.info(f"Sampling up to {n_ev_pool} points from EV dataset (pool for SHAP)...")
    ev_df = sample_points_from_xarray_dataset(
        evs_dataset=evs_dataset,
        n_samples=n_ev_pool,
        random_state=42
    )
    logger.info(f"Sampled {len(ev_df)} valid points from EV dataset")
    
    # Process models in parallel
    # Determine number of workers
    if n_jobs is None:
        import os
        n_workers = os.cpu_count() or 1
    else:
        n_workers = n_jobs
    
    logger.info(f"Computing SHAP values for {len(filtered_index)} models using {n_workers} parallel worker(s)...")
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(process_single_model)(
            row=row,
            models_dir=models_dir,
            ev_df=ev_df,
            output_dir=output_dir,
            n_background=n_background,
            n_explain=n_explain,
            top_features=top_features,
            random_state=42
        )
        for _, row in filtered_index.iterrows()
    )
    logger.info(f"Parallel processing complete. Processed {len(results)} models.")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results summary
    results_path = output_dir / "explanation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Explanation results saved to {results_path}")
    
    # Summary statistics
    n_success = results_df['success'].sum()
    n_failed = len(results_df) - n_success
    logger.info(f"✓ Explanation pipeline complete: {n_success} succeeded, {n_failed} failed")
    
    return results_df

