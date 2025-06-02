# Core MaxEnt (Elapid-based) model training, evaluation, and prediction logic.
import warnings
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
from elapid.models import MaxentModel as BaseMaxentModel
from elapid.types import to_iterable # Used by elapid internals, good to be aware of
from elapid.utils import (
    NoDataException,
    check_raster_alignment,
    create_output_raster_profile,
    get_raster_band_indexes,
    # tqdm_opts, # Not directly used here, elapid.utils.get_tqdm handles it
    get_tqdm, # Not directly used here, elapid.geo.apply_model_to_array handles its own tqdm
    tqdm_opts, # Used by elapid.geo.apply_model_to_array
)
from elapid.geo import apply_model_to_array # Core raster prediction function from elapid
from elapid.models import MaxentConfig
from sdm.models.core.feature_subsetter import FeatureSubsetter


tqdm = get_tqdm()


# Potentially import from .utils if model-specific utils are there
# from .utils import prepare_occurrence_data # Example, if it were MaxEnt specific

logger = logging.getLogger(__name__)


class DefaultMaxentConfig(MaxentConfig):
    """
    A default MaxentConfig object that can be used to create a MaxentModel.
    """
    def __init__(
            self,
            feature_types: List[str] = ["linear", "hinge", "product"],
            beta_multiplier: float = 1.5,
            beta_lqp: float = 1.0,
            beta_hinge: float = 1.0,
            beta_threshold: float = 1.0,
            beta_categorical: float = 1.0,
            n_hinge_features: int = 10,
            n_threshold_features: int = 10,
            clamp: bool = True,
            convergence_tolerance: float = 1e-5,
            use_lambdas: str = "best",
            n_lambdas: int = 100,
            class_weights: Union[str, float] = 100,
            n_cpus: int = 1,
            use_sklearn: bool = True,
            tau: float = 0.5,
            transform: str = "cloglog",
    ):
        super().__init__()
        self.feature_types = feature_types
        self.beta_multiplier=beta_multiplier
        self.beta_lqp=beta_lqp # Default
        self.beta_hinge=beta_hinge # Default
        self.beta_threshold=beta_threshold # Default
        self.beta_categorical=beta_categorical # Default
        self.n_hinge_features=n_hinge_features # Increased from 5 in original script
        self.n_threshold_features=n_threshold_features # Increased from 5 in original script
        self.clamp=clamp
        self.convergence_tolerance=convergence_tolerance # Default in elapid
        self.use_lambdas=use_lambdas
        self.n_lambdas=n_lambdas
        self.class_weights=class_weights
        self.n_cpus=n_cpus
        self.use_sklearn=use_sklearn
        self.tau=tau
        self.transform=transform

# add a from_config classmethod to MaxentConfig
class MaxentModel(BaseMaxentModel):
    """
    A wrapper around elapid.MaxentModel that allows for configuration via a MaxentConfig object.
    This is useful for setting the model parameters via a config file.
    """
    @classmethod
    def from_config(cls, config: MaxentConfig, n_cpus: int = 1) -> "MaxentModel":
        """
        Create a MaxentModel from a MaxentConfig object.
        """
        return cls(
            feature_types=config.feature_types,
            tau=config.tau,
            transform=config.transform, # type: ignore
            clamp=config.clamp,
            scorer=config.scorer,
            beta_multiplier=config.beta_multiplier,
            beta_lqp=config.beta_lqp,
            beta_hinge=config.beta_hinge,
            beta_threshold=config.beta_lqp,
            beta_categorical=config.beta_categorical,
            n_hinge_features=config.n_hinge_features,
            n_threshold_features=config.n_threshold_features,
            convergence_tolerance=config.tolerance,
            use_lambdas=config.use_lambdas,
            n_lambdas=config.n_lambdas,
            class_weights=config.class_weights,
            n_cpus=n_cpus,
            use_sklearn=True,
        )

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

    This function performs cross-validation and then trains a final model on all data.
    It is a wrapper around cross_validate_maxent_model and train_final_maxent_model.

    Args:
        model: The scikit-learn compatible model instance (e.g., elapid.MaxentModel).
        occurrence_gdf: GeoDataFrame with occurrence data (features, class, sample_weight, geometry).
        metric_fn: Callable function to calculate a performance metric (e.g., roc_auc_score).
        n_cv_folds: Number of folds for geographic cross-validation.
        feature_columns: List of feature column names. If None, inferred.
        random_state_kfold: Random state for GeographicKFold for reproducible splits.

    Returns:
        Tuple of (final_trained_model, cv_models, cv_scores).
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
    maxent_n_jobs: int = 1, # Threads for MaxentModel itself
    model_config: MaxentConfig = DefaultMaxentConfig(),
    # Add other MaxentModel params as needed
) -> Pipeline:
    """Creates a scikit-learn Pipeline for MaxEnt modeling.
    Includes feature selection (custom FeatureSubsetter), scaling, and the Elapid MaxentModel.

    Args:
        feature_names: List of feature names to be selected by FeatureSubsetter.
        maxent_beta_multiplier: Beta multiplier for the Maxent model.
        maxent_n_jobs: Number of threads for the MaxentModel.

    Returns:
        A scikit-learn Pipeline instance.
    """
    logger.info(f"Creating MaxEnt pipeline for features: {feature_names}")
    
    # Feature selector: uses custom FeatureSubsetter to select only the desired features
    feature_selector = FeatureSubsetter(feature_names=feature_names)
    
    # Scaler: Standardizes features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()

    # Maxent Model from Elapid
    maxent_estimator = MaxentModel.from_config(model_config, n_cpus=maxent_n_jobs)

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



def apply_model_to_rasters(
    model: BaseEstimator,
    raster_paths: list,
    output_path: str,
    feature_names: Optional[List[str]] = None,
    resampling: rio.enums.Enum = rio.enums.Resampling.average,
    count: int = 1,
    dtype: str = "float32",
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
    template_idx: int = 0,
    windowed: bool = True,
    window_size: int = 1024,  # Increased from default
    predict_proba: bool = False,
    ignore_sklearn: bool = True,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Applies a trained model to a list of raster datasets.

    Args:
        model: object with a model.predict() function
        raster_paths: raster paths of covariates to apply the model to
        output_path: path to the output file to create
        feature_names: Optional list of feature names that the model expects
        resampling: resampling algorithm for reprojection
        count: number of bands in the prediction output
        dtype: the output raster data type
        nodata: output nodata value
        driver: output raster format
        compress: compression to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        template_idx: index of the raster file to use as a template
        windowed: apply the model using windowed read/write
        window_size: size of processing windows (default: 1024)
        predict_proba: use model.predict_proba() instead of model.predict()
        ignore_sklearn: silence sklearn warning messages
        quiet: silence progress bar output
        **kwargs: additonal keywords to pass to model.predict()
    """
    raster_paths = to_iterable(raster_paths)
    windows, dst_profile = create_output_raster_profile(
        raster_paths, template_idx, count=count, windowed=windowed,
        nodata=nodata, compress=compress, driver=driver, bigtiff=bigtiff
    )
    
    # Adjust window size if specified
    if windowed and window_size > 0:
        windows = [rio.windows.Window(
            col_off=w.col_off,
            row_off=w.row_off,
            width=min(window_size, w.width),
            height=min(window_size, w.height)
        ) for w in windows]
    
    nbands, band_idx = get_raster_band_indexes(raster_paths)
    aligned = check_raster_alignment(raster_paths)
    nodata = nodata or 0

    if ignore_sklearn:
        warnings.filterwarnings("ignore", category=UserWarning)

    srcs = [rio.open(raster_path) for raster_path in raster_paths]
    
    # Get band names and handle feature selection
    band_names = []
    for src in srcs:
        if (feature_names is not None) and not src.descriptions:
            raise ValueError(f"Raster {src.name} has no descriptions, but feature_names are provided.")
        band_names.extend(src.descriptions or [f"band_{i}" for i in range(src.count)])
    
    matching_indices = None
    if feature_names is not None:
        matching_indices = [i for i, name in enumerate(band_names) if name in feature_names]
        if not matching_indices:
            raise ValueError(f"No matching bands found for features: {feature_names}")
        if len(matching_indices) != len(feature_names):
            missing = set(feature_names) - set(band_names)
            raise ValueError(f"Some features not found in raster bands: {missing}")
        nbands = len(matching_indices)
        band_idx = [0] + [1] * nbands

    if not aligned:
        vrt_options = {
            "resampling": resampling,
            "transform": dst_profile["transform"],
            "crs": dst_profile["crs"],
            "height": dst_profile["height"],
            "width": dst_profile["width"],
        }
        srcs = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

    with rio.open(output_path, "w", **dst_profile) as dst:
        for window in tqdm(windows, desc="Window", disable=quiet, **tqdm_opts):
            covariates = np.zeros((nbands, window.height, window.width), dtype=np.float32)
            nodata_idx = np.ones_like(covariates, dtype=bool)

            try:
                if matching_indices is not None:
                    for i, idx in enumerate(matching_indices):
                        src_idx = idx // srcs[0].count
                        band_idx = idx % srcs[0].count
                        data = srcs[src_idx].read(band_idx + 1, window=window, masked=True)
                        covariates[i] = data
                        nodata_idx[i] = data.mask
                else:
                    for i, src in enumerate(srcs):
                        data = src.read(window=window, masked=True)
                        covariates[band_idx[i] : band_idx[i + 1]] = data
                        nodata_idx[band_idx[i] : band_idx[i + 1]] = data.mask

                if nodata_idx.any(axis=0).all():
                    raise NoDataException()

                predictions = apply_model_to_array(
                    model, covariates, nodata, nodata_idx,
                    count=count, dtype=dtype, predict_proba=predict_proba, **kwargs
                )
                dst.write(predictions, window=window)

            except NoDataException:
                continue

def apply_models_to_raster(
    models: Dict[str, BaseEstimator],
    raster_path: Union[str, Path],
    output_path: Union[str, Path],
    dtype: str = "float32",
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
    windowed: bool = True,
    window_size: int = 128,
    predict_proba: bool = True,
    ignore_sklearn: bool = True,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Applies multiple trained models to a single raster dataset.

    This function applies each model to the same raster data and combines their predictions
    into a single multi-band raster, where each band corresponds to a model's prediction.

    Args:
        models: Dictionary mapping model identifiers to trained model objects
        raster_path: Path to the input raster file containing features
        output_path: Path to save the output prediction raster (GeoTIFF)
        dtype: Output raster data type
        nodata: Output nodata value
        driver: Output raster format
        compress: Compression to apply to the output file
        bigtiff: Specify the output file as a bigtiff (for rasters > 2GB)
        windowed: Apply the models using windowed read/write
        window_size: Size of processing windows
        predict_proba: Use model.predict_proba() instead of model.predict()
        ignore_sklearn: Silence sklearn warning messages
        quiet: Silence progress bar output
        **kwargs: Additional keywords to pass to model.predict()
    """
    logger.info(f"Applying {len(models)} models to raster: {raster_path}")
    
    # Convert paths to strings
    raster_path = str(raster_path)
    output_path = str(output_path)

    # Create output profile with number of bands equal to number of models
    windows, dst_profile = create_output_raster_profile(
        [raster_path], 0, count=len(models), windowed=windowed,
        nodata=nodata, compress=compress, driver=driver, bigtiff=bigtiff
    )
    
    # Create larger windows based on window_size
    if windowed and window_size > 0:
        with rio.open(raster_path) as src:
            height = src.height
            width = src.width
            
            # Calculate number of windows needed
            n_windows_h = (height + window_size - 1) // window_size
            n_windows_w = (width + window_size - 1) // window_size
            
            # Create windows
            windows = []
            for i in range(n_windows_h):
                for j in range(n_windows_w):
                    row_off = i * window_size
                    col_off = j * window_size
                    win_height = min(window_size, height - row_off)
                    win_width = min(window_size, width - col_off)
                    windows.append(rio.windows.Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=win_width,
                        height=win_height
                    ))
            
            logger.info(f"Created {len(windows)} windows of size {window_size}x{window_size}")
    
    # Get band names from raster
    with rio.open(raster_path) as src:
        band_names = src.descriptions or [f"band_{i}" for i in range(src.count)]
    
    if ignore_sklearn:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Open source raster
    src = rio.open(raster_path)
    
    # Create output raster with model identifiers as band descriptions
    with rio.open(output_path, "w", **dst_profile) as dst:
        # Set band descriptions to model identifiers
        dst.descriptions = list(models.keys())
        
        for window in tqdm(windows, desc="Processing windows", disable=quiet, **tqdm_opts):
            # Read window data
            data = src.read(window=window, masked=True)
            
            # Skip if all data in window is nodata
            if data.mask.all():
                continue
            
            # Convert to DataFrame
            n_features, height, width = data.shape
            X = pd.DataFrame(
                data.reshape(n_features, -1).T,  # Reshape to (samples, features)
                columns=band_names
            )
            
            # Apply each model and store predictions
            predictions = np.zeros((len(models), height, width), dtype=np.float32)
            
            for i, (model_id, model) in enumerate(models.items()):
                try:
                    # Apply model to window data
                    if predict_proba:
                        pred = model.predict_proba(X)[:, 1]  # Probability of class 1
                    else:
                        pred = model.predict(X)
                    
                    # Reshape prediction back to window dimensions
                    pred = pred.reshape(height, width)
                    
                    # Store prediction in output array
                    predictions[i] = pred
                    
                except Exception as e:
                    logger.error(f"Error applying model {model_id} to window: {e}")
                    predictions[i] = nodata
            
            # Write predictions to output raster
            dst.write(predictions, window=window)
    
    logger.info(f"Successfully applied {len(models)} models to raster. Output saved to: {output_path}")
