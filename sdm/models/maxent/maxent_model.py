# Core MaxEnt (Elapid-based) model training, evaluation, and prediction logic.

import logging
from typing import List, Tuple, Optional, Callable, Any, Union, Dict # Added Union, Dict
from pathlib import Path
from enum import StrEnum # Added StrEnum import

import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, clone
import rasterio as rio # For rio.enums and types used in apply_model_to_rasters
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import elapid as ela
from elapid.types import to_iterable # Used by elapid internals, good to be aware of
from elapid.utils import (
    NoDataException,
    check_raster_alignment,
    create_output_raster_profile,
    get_raster_band_indexes,
    # tqdm_opts, # Not directly used here, elapid.utils.get_tqdm handles it
    # get_tqdm, # Not directly used here, elapid.geo.apply_model_to_array handles its own tqdm
)
from elapid.geo import apply_model_to_array # Core raster prediction function from elapid

# Potentially import from .utils if model-specific utils are there
# from .utils import prepare_occurrence_data # Example, if it were MaxEnt specific

logger = logging.getLogger(__name__)

def extract_split_data(
    gdf: gpd.GeoDataFrame, 
    idx: np.ndarray, 
    feature_columns: Optional[List[str]] = None,
    class_col: str = "class", 
    weight_col: str = "sample_weight",
    geometry_col: str = "geometry"
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Extracts X (features), y (class labels), and sample weights from a GeoDataFrame 
    based on provided indices.

    Args:
        gdf: Input GeoDataFrame containing features, class, weights, and geometry.
        idx: NumPy array of indices to select rows for this split.
        feature_columns: Optional list of column names to be used as features (X).
                         If None, all columns except class, weight, and geometry are used.
        class_col: Name of the column containing class labels.
        weight_col: Name of the column containing sample weights. Can be None.
        geometry_col: Name of the geometry column.

    Returns:
        A tuple (X, y, weights):
        X (pd.DataFrame): Feature data.
        y (pd.Series): Class labels.
        weights (pd.Series, optional): Sample weights. None if weight_col is not found or None.
    """
    split_gdf = gdf.iloc[idx].copy()
    
    if feature_columns is None:
        cols_to_drop = [class_col, geometry_col]
        if weight_col and weight_col in split_gdf.columns: # Only drop if it exists
            cols_to_drop.append(weight_col)
        X = split_gdf.drop(columns=cols_to_drop)
    else:
        X = split_gdf[feature_columns]
        
    y = split_gdf[class_col]
    
    weights: Optional[pd.Series] = None
    if weight_col and weight_col in split_gdf.columns:
        weights = split_gdf[weight_col]
    elif weight_col:
        logger.warning(f"Sample weight column '{weight_col}' not found in GeoDataFrame. Proceeding without weights.")

    return X, y, weights


def cross_validate_maxent_model(
    model: BaseEstimator, 
    occurrence_gdf: gpd.GeoDataFrame, 
    metric_fn: Callable = roc_auc_score, 
    n_folds: int = 3,
    feature_columns: Optional[List[str]] = None,
    random_state_kfold: Optional[int] = None # For reproducibility of GeographicKFold
) -> Tuple[List[BaseEstimator], np.ndarray]:
    """
    Performs geographic cross-validation for a MaxEnt-like (scikit-learn compatible) model.

    Args:
        model: The scikit-learn compatible model instance (e.g., elapid.MaxentModel).
        occurrence_gdf: GeoDataFrame with occurrence data (features, class, sample_weight, geometry).
        metric_fn: Callable function to calculate a performance metric (e.g., roc_auc_score).
        n_folds: Number of folds for geographic cross-validation.
        feature_columns: List of feature column names. If None, inferred.
        random_state_kfold: Random state for GeographicKFold for reproducible splits.

    Returns:
        Tuple of (trained_models_per_fold, metric_scores_per_fold).
    """
    # elapid.GeographicKFold is a good default choice here
    # It can take a random_state if provided in elapid versions that support it.
    # Check elapid documentation for exact signature if random_state is critical.
    try:
        gfolds = ela.GeographicKFold(n_splits=n_folds, random_state=random_state_kfold)
    except TypeError: # Older elapid might not have random_state
        logger.warning("GeographicKFold does not support random_state in this elapid version. Using default.")
        gfolds = ela.GeographicKFold(n_splits=n_folds)
        
    fold_metrics = []
    trained_models = []

    logger.info(f"Starting {n_folds}-fold geographic cross-validation...")
    for i, (train_idx, test_idx) in enumerate(gfolds.split(occurrence_gdf)):
        logger.info(f"Processing fold {i+1}/{n_folds}")
        current_model = clone(model) # Use a fresh clone for each fold

        X_train, y_train, w_train = extract_split_data(occurrence_gdf, train_idx, feature_columns=feature_columns)
        X_test, y_test, _ = extract_split_data(occurrence_gdf, test_idx, feature_columns=feature_columns) # Weights not used for test metric here

        if len(y_test.unique()) < 2:
            logger.warning(f"Skipping fold {i+1} due to only one class in the test set.")
            fold_metrics.append(np.nan) # Or some other indicator for a skipped fold
            trained_models.append(None) # No model for this fold
            continue
        
        try:
            # Elapid MaxentModel expects sample_weight as `maxent__sample_weight` in fit params
            # Or if using a Pipeline, it would be `stepname__maxent__sample_weight`
            # Assuming model is a direct MaxentModel or compatible with this param name.
            # We need to find the correct parameter name for sample_weight.
            # For a direct elapid.MaxentModel, it is 'sample_weight'.
            # If it's part of a scikit-learn pipeline, it would be '<estimator_name>__sample_weight'.
            # This is a common point of confusion. For now, assume direct elapid.MaxentModel style.
            fit_params = {}
            if w_train is not None:
                # Check if model is a pipeline to construct prefixed param name
                if hasattr(current_model, 'steps'): # It's a pipeline
                    # Assuming maxent is the last step, or find its name
                    maxent_step_name = current_model.steps[-1][0] 
                    fit_params[f'{maxent_step_name}__sample_weight'] = w_train
                else: # Assume it's a direct estimator
                    fit_params['sample_weight'] = w_train
            
            current_model.fit(X_train, y_train, **fit_params)
            
            y_pred_proba = current_model.predict_proba(X_test)[:, 1] # Probability of class 1
            metric_value = metric_fn(y_test, y_pred_proba)
            fold_metrics.append(metric_value)
            trained_models.append(current_model)
            logger.info(f"Fold {i+1} metric ({metric_fn.__name__}): {metric_value:.4f}")
        except Exception as e:
            logger.error(f"Error during training/evaluation of fold {i+1}: {e}", exc_info=True)
            fold_metrics.append(np.nan)
            trained_models.append(None)

    return trained_models, np.array(fold_metrics)


def train_final_maxent_model(
    model: BaseEstimator, 
    occurrence_gdf: gpd.GeoDataFrame, 
    feature_columns: Optional[List[str]] = None
) -> BaseEstimator:
    """Trains a MaxEnt-like model on the entire dataset."""
    logger.info("Training final model on all data...")
    final_model = clone(model)
    train_idx = np.arange(len(occurrence_gdf))
    X_train, y_train, w_train = extract_split_data(occurrence_gdf, train_idx, feature_columns=feature_columns)
    
    fit_params = {}
    if w_train is not None:
        if hasattr(final_model, 'steps'):
            maxent_step_name = final_model.steps[-1][0]
            fit_params[f'{maxent_step_name}__sample_weight'] = w_train
        else:
            fit_params['sample_weight'] = w_train
            
    final_model.fit(X_train, y_train, **fit_params)
    logger.info("Final model training complete.")
    return final_model


def evaluate_and_train_maxent_model(
    model: BaseEstimator, 
    occurrence_gdf: gpd.GeoDataFrame, 
    metric_fn: Callable = roc_auc_score, 
    n_cv_folds: int = 3,
    feature_columns: Optional[List[str]] = None,
    random_state_kfold: Optional[int] = None
) -> Tuple[BaseEstimator, List[BaseEstimator], np.ndarray]:
    """
    Performs cross-validation and then trains a final model on all data.
    """
    logger.info("Starting model evaluation and final training process...")
    cv_models, cv_scores = cross_validate_maxent_model(
        model=model, # Pass the original model for cloning inside CV
        occurrence_gdf=occurrence_gdf,
        metric_fn=metric_fn,
        n_folds=n_cv_folds,
        feature_columns=feature_columns,
        random_state_kfold=random_state_kfold
    )
    
    logger.info(f"CV Mean {metric_fn.__name__}: {np.nanmean(cv_scores):.4f} (+/- {np.nanstd(cv_scores):.4f})")
    
    final_trained_model = train_final_maxent_model(
        model=model, # Pass the original model for cloning
        occurrence_gdf=occurrence_gdf,
        feature_columns=feature_columns
    )
    
    return final_trained_model, cv_models, cv_scores


def predict_rasters_with_elapid_model(
    model: BaseEstimator,
    raster_paths: List[Union[str, Path]], # List of paths to individual raster files (features)
    output_path: Union[str, Path],
    # resampling_method: rio.enums.Resampling = rio.enums.Resampling.average, # elapid apply_model_to_array handles internal resampling based on template
    # count: int = 1, # Output is usually 1 band (probability)
    # dtype: str = "float32", # elapid default is float32
    # driver: str = "GTiff", # elapid default
    # compress: str = "deflate", # elapid default
    # bigtiff: bool = True, # elapid default if needed
    template_raster_idx: int = 0, # Index in raster_paths to use as template for output grid
    windowed_prediction: bool = True, # Use elapid's windowed processing for large rasters
    predict_proba: bool = True, # True for probability map, False for binary (if model supports .predict)
    # ignore_sklearn: bool = True, # elapid param, True if model is not sklearn, False if it is.
                                     # For elapid.MaxentModel, this should be True.
    quiet_progress: bool = False,
    creation_options: Optional[Dict[str, Any]] = None # For rasterio creation options
) -> None:
    """
    Applies a trained (Elapid-compatible) model to a stack of rasters to generate a prediction map.
    This function leverages `elapid.geo.apply_model_to_array`.

    Args:
        model: Trained scikit-learn compatible model (e.g., elapid.MaxentModel).
        raster_paths: List of paths to GeoTIFF files, one for each feature variable, in the order expected by the model.
        output_path: Path to save the output prediction raster (GeoTIFF).
        template_raster_idx: Index of the raster in `raster_paths` to use as a template for the output grid and CRS.
        windowed_prediction: If True, use windowed processing (recommended for large rasters).
        predict_proba: If True, predict probabilities (uses model.predict_proba). If False, predict classes (uses model.predict).
        quiet_progress: If True, suppress tqdm progress bar from elapid.
        creation_options: Rasterio creation options (e.g., {'COMPRESS': 'LZW'}).
    """
    logger.info(f"Applying model to rasters to generate prediction map: {output_path}")
    
    raster_paths_str = [str(p) for p in raster_paths]

    # Elapid's apply_model_to_array handles raster alignment, profile creation, etc.
    # It assumes the model is NOT an sklearn pipeline if ignore_sklearn=True (default for Maxent).
    # If your model IS an sklearn pipeline, set ignore_sklearn=False.
    is_sklearn_pipeline = hasattr(model, 'steps')

    apply_model_to_array(
        model=model,
        rasters=raster_paths_str, # elapid expects list of strings
        output_path=str(output_path),
        template_idx=template_raster_idx,
        windowed=windowed_prediction,
        predict_proba=predict_proba,
        ignore_sklearn=not is_sklearn_pipeline, # True if not sklearn pipeline (e.g. direct Elapid model)
        quiet=quiet_progress,
        # driver=driver, # Uses elapid defaults or user-set via creation_options
        # compress=compress,
        # bigtiff=bigtiff,
        # count=count,
        # dtype=dtype,
        # resampling=resampling_method, # elapid handles resampling to template
        ** (creation_options or {}) # Pass rasterio creation options
    )
    logger.info("Prediction map saved successfully.")


def create_maxent_pipeline(
    feature_names: List[str], 
    maxent_beta_multiplier: float = 2.5, # example of making params configurable
    maxent_n_jobs: int = 1 # Threads for MaxentModel itself
    # Add other MaxentModel params as needed
) -> Pipeline:
    """Creates a scikit-learn Pipeline for MaxEnt modeling.
    Includes feature selection (passthrough), scaling, and the Elapid MaxentModel.

    Args:
        feature_names: List of feature names to be selected by ColumnTransformer.
        maxent_beta_multiplier: Beta multiplier for the Maxent model.
        maxent_n_jobs: Number of threads for the MaxentModel.

    Returns:
        A scikit-learn Pipeline instance.
    """
    logger.info(f"Creating MaxEnt pipeline for features: {feature_names}")
    
    # Feature selector: uses ColumnTransformer to select only the desired features
    # and passes them through without transformation at this stage.
    feature_selector = ColumnTransformer(
        transformers=[
            ("selector", "passthrough", feature_names)
        ],
        remainder="drop" # Drop any columns not in feature_names
    )
    
    # Scaler: Standardizes features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    
    # Maxent Model from Elapid
    maxent_estimator = ela.MaxentModel(
        feature_types=["linear", "quadratic", "hinge", "product"], # Default Elapid features
        beta_multiplier=maxent_beta_multiplier,
        beta_lqp=1.0, # Default
        beta_hinge=1.0, # Default
        beta_threshold=1.0, # Default
        beta_categorical=1.0, # Default
        n_hinge_features=10, # Increased from 5 in original script
        n_threshold_features=10, # Increased from 5 in original script
        # transform="cloglog", # Default in elapid if not specified, or choose one
        clamp=True,
        # tau=0.5, # Default in elapid
        convergence_tolerance=1e-5, # Default in elapid
        # use_lambdas="best", # Default in elapid
        # project_log_file=None, # Default in elapid
        # response_curves=False, # Default in elapid
        # jackknife=False, # Default in elapid
        # diagnostics=False, # Default in elapid
        # compute_auc=False, # Default in elapid, CV handles AUC
        n_cpus=maxent_n_jobs # Threads for the model itself
    )

    pipeline = Pipeline([
        ("feature_selection", feature_selector),
        ("scaling", scaler),
        ("maxent", maxent_estimator)
    ])
    

    
    logger.info("MaxEnt pipeline created successfully.")
    return pipeline

# Enum for activity types, useful for get_feature_config
class ActivityType(StrEnum):
    ROOST = "Roost"
    IN_FLIGHT = "In flight"

def get_feature_config() -> Dict[ActivityType, List[str]]: # Changed to use ActivityType enum
    """
    Gets the list of feature names to include in the model for different activity types.
    """
    return {
        ActivityType.IN_FLIGHT: [
            "ceh_landcover_improved_grassland",
            "ceh_landcover_suburban",
            "bgs_coast_distance_to_coast",
            "ceh_landcover_arable",
            "ceh_landcover_suburban_500m",
            "ceh_landcover_arable_500m",
            "climate_bioclim_bio_7",
            "vom_vegetation_height_max",
            "terrain_stats_slope",
            "ceh_landcover_broadleaved_woodland",
            "climate_bioclim_bio_9",
            "climate_stats_temp_ann_var",
            "os_cover_water",
            "climate_bioclim_bio_3",
            "ceh_landcover_improved_grassland_500m",
            "ceh_landcover_urban",
            "climate_stats_prec_ann_avg",
            "climate_stats_temp_ann_avg",
            "os_distance_distance_to_major_roads",
            "terrain_dtm",
            "os_distance_distance_to_buildings",
            "ceh_landcover_urban_500m",
            "climate_stats_wind_ann_avg",
        ],
        ActivityType.ROOST: [
            "ceh_landcover_suburban",
            "vom_vegetation_height_max",
            "os_distance_distance_to_buildings",
            "vom_vegetation_height_mean_500m",
            "ceh_landcover_suburban_500m",
            "bgs_coast_distance_to_coast",
            "ceh_landcover_broadleaved_woodland_500m",
            "ceh_landcover_improved_grassland",
            "terrain_stats_roughness",
            "climate_bioclim_bio_3",
            "climate_bioclim_bio_9",
            "ceh_landcover_broadleaved_woodland",
            "ceh_landcover_arable_500m",
            "ceh_landcover_improved_grassland_500m",
            "ceh_landcover_grassland",
            "climate_stats_temp_ann_avg",
        ],
    } 