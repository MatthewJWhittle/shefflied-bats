"""
MaxEnt Species Distribution Modelling for Sheffield Bats.

This module implements the MaxEnt modelling pipeline for bat species distribution
in the Sheffield area, including data preparation, model training, and evaluation.
Uses a modular approach where each function does one thing and can be easily composed.
"""

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, cast, Any, Any
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import elapid as ela
from scipy.spatial import cKDTree
import topojson as tp

from sdm.data.loaders.vector import load_bat_data, load_background_points
from sdm.data.processing.core import annotate_points
from sdm.models.maxent.maxent_model import (
    ActivityType,
    DefaultMaxentConfig,
    MaxentConfig,
    create_maxent_pipeline,
    evaluate_and_train_maxent_model,
)
from sdm.occurrence.sampling import (
    generate_background_points_from_data,
    BackgroundMethod,
    TransformMethod,
)
from sdm.raster.io import load_environmental_variables
from sdm.utils.rng import apply_training_seed
from sdm.utils.io import (
    load_project_config,
    load_model_config,
    load_variables_config,
    load_boundary,
    CONFIG_PATH,
    MODEL_CONFIG_PATH,
)
from sdm.utils.sdm_config import (
    model_config_file_paths,
    repo_relative_config_path,
    resolve_model_config_dir,
    resolve_sdm_config_root,
)
from sdm.utils.logging_utils import setup_logging
from sdm.types import (
    TrainingData,
    TrainingResults,
    ProjectConfig,
    VariablesConfig,
    ModelConfig,
    BackgroundConfig,
    SamplingConfig,
    CvConfig,
)
from sdm.commands.modelling.utils import get_model_id
from sdm.models.core.threshold import write_run_thresholds
from sdm.models.maxent.folds import (
    DEFAULT_BUFFERED_LOO_WARN_ABOVE,
    FoldMethod,
    FoldParams,
    buffered_loo_presence_warning,
    fold_assignments_frame,
    presence_counts_per_fold,
    resolve_fold_params,
)
from sdm.utils.fs import join_uri_path, local_materialized_path, uri_exists
from sdm.utils.model_runs import (
    artifact_parent_uri,
    build_run_output_uri,
    join_run_artifact,
    model_run_write_context,
    resolve_data_path,
    resolve_output_root,
    run_segment_from_uri,
)
from sdm.utils.mlflow_tracking import (
    TrainingSessionMetadata,
    configure_mlflow,
    log_models_to_mlflow,
    resolve_mlflow_config,
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=pd.DataFrame)


# ============================================================================
# Modular data preparation functions
# ============================================================================


def convert_evs_to_geodataframe(evs_dataset: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Convert environmental variables xarray Dataset to GeoDataFrame.
    
    Args:
        evs_dataset: xarray Dataset with environmental variables
        
    Returns:
        GeoDataFrame with environmental variables as columns and grid points as rows
    """
    ev_df = evs_dataset.to_dataframe()
    ev_df.reset_index(inplace=True)
    ev_gdf = gpd.GeoDataFrame(ev_df, geometry=gpd.points_from_xy(ev_df.x, ev_df.y))
    ev_gdf.crs = evs_dataset.rio.crs
    ev_gdf = ev_gdf.drop(columns=["spatial_ref"])
    ev_gdf.rename(columns={"x": "grid_x", "y": "grid_y"}, inplace=True)
    
    # Drop rows with missing values
    ev_gdf.dropna(inplace=True)
    
    return ev_gdf


def extract_grid_points(evs_dataset: xr.Dataset) -> gpd.GeoDataFrame:
    """Point GeoDataFrame over the EV raster grid (for distance / neighbourhood features)."""
    return convert_evs_to_geodataframe(evs_dataset)


def annotate_points_with_evs(
    points: gpd.GeoDataFrame,
    ev_gdf: gpd.GeoDataFrame,
    tolerance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Annotate points with environmental variables using spatial join.
    
    Args:
        points: GeoDataFrame of points to annotate
        ev_gdf: GeoDataFrame with environmental variables
        tolerance: Maximum distance for nearest neighbor join (defaults to raster resolution)
        
    Returns:
        Annotated GeoDataFrame
    """
    if tolerance is None:
        resolution = ev_gdf.rio.resolution()
        tolerance = abs(resolution[0])
    
    annotated = gpd.sjoin_nearest(
        points, ev_gdf, how="inner", max_distance=tolerance
    )
    annotated.drop(columns=["index_right"], inplace=True)
    
    return annotated


def simplify_boundary(boundary: gpd.GeoDataFrame, tolerance: float = 100) -> gpd.GeoDataFrame:
    """
    Simplify boundary geometry using TopoJSON simplification.
    
    Args:
        boundary: GeoDataFrame with boundary geometry
        tolerance: Simplification tolerance
        
    Returns:
        Simplified boundary GeoDataFrame
    """
    boundary_tp = tp.Topology(boundary, prequantize=True)
    boundary_tp.toposimplify(tolerance, inplace=True)
    simplified = boundary_tp.to_gdf()
    simplified["geometry"] = simplified.geometry.make_valid()
    simplified["geometry"] = simplified.geometry.buffer(0)
    return simplified


def prepare_training_data(
    occurrence_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    evs_to_model: xr.Dataset,
    n_background_points: int,
    background_method: BackgroundMethod,
    background_value: float,
    sigma: float,
    transform_method: TransformMethod,
    *,
    seed: int,
    background_occurrence_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, List[str]]:
    """
    Prepare study-level training data shared across all species-activity combinations.
    
    This takes in-memory objects (occurrence records, boundary, EV dataset) and returns:
    - presence_with_evs_gdf: all presence points annotated with environmental variables
    - background_with_evs_gdf: all background points annotated with environmental variables
    - ev_columns: list of environmental variable column names

    ``background_occurrence_gdf`` is the record set whose density drives background
    sampling. Pass all bat records so that a run filtered to some species or activity
    types samples background exactly as a full run does. When omitted,
    ``occurrence_gdf`` is used.
    """
    logger.info("Preparing shared training data (presence and background with EVs)")

    # Derive available feature names from EV dataset
    ev_columns: List[str] = list(evs_to_model.data_vars.keys())

    # Convert EV dataset to GeoDataFrame once
    ev_gdf = convert_evs_to_geodataframe(evs_to_model)

    # Prepare presence data for EV annotation
    split_keys = ["latin_name", "activity_type"]
    occurrence_for_annotation_gdf = occurrence_gdf[split_keys + ["geometry"]].copy()
    occurrence_for_annotation_gdf["class"] = 1

    # Annotate presence points with EVs
    resolution = evs_to_model.rio.resolution()
    tolerance = abs(resolution[0])
    presence_with_evs_gdf = annotate_points_with_evs(
        occurrence_for_annotation_gdf,
        ev_gdf,
        tolerance=tolerance,
    )

    # Generate background points from all occurrence data (not just the filtered subset)
    background_source_gdf = (
        background_occurrence_gdf if background_occurrence_gdf is not None else occurrence_gdf
    )
    background_points_gdf, _ = generate_background_points_from_data(
        occurrence_data=background_source_gdf,
        boundary=boundary,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        sigma=sigma,
        transform_method=transform_method,
        clip_to_boundary=True,
        seed=seed,
    )

    # Annotate background points with EVs
    background_with_evs_gdf = annotate_points_with_evs(
        background_points_gdf,
        ev_gdf,
        tolerance=tolerance,
    )
    background_with_evs_gdf["class"] = 0

    logger.info(
        "Shared training data prepared: %d presence points, %d background points",
        len(presence_with_evs_gdf),
        len(background_with_evs_gdf),
    )

    return presence_with_evs_gdf, background_with_evs_gdf, ev_columns


# ============================================================================
# Species-specific data processing functions
# ============================================================================


def get_species_presence_data(
    presence: gpd.GeoDataFrame,
    latin_name: str,
    activity_type: str,
) -> gpd.GeoDataFrame:
    """
    Filter presence data for a specific species and activity type.
    
    Args:
        presence: All presence data
        latin_name: Species latin name
        activity_type: Activity type
        
    Returns:
        Filtered presence data
    """
    assert "latin_name" in presence.columns, "latin_name column not found in presence data"
    assert "activity_type" in presence.columns, "activity_type column not found in presence data"
    return presence[
        (presence.activity_type == activity_type) &
        (presence.latin_name == latin_name)
    ]


def drop_duplicate_grid_points(
    gdf: gpd.GeoDataFrame,
    index_cols: List[str] = ["grid_x", "grid_y"],
) -> gpd.GeoDataFrame:
    """
    Drop duplicate points based on grid coordinates.
    
    Args:
        gdf: GeoDataFrame to deduplicate
        index_cols: Columns to use for deduplication
        
    Returns:
        Deduplicated GeoDataFrame
    """
    return gdf.drop_duplicates(subset=index_cols)


def filter_background_by_distance(
    background_data: gpd.GeoDataFrame,
    presence_data: gpd.GeoDataFrame,
    d_min: float = 500,
    d_max: float = np.inf,
) -> gpd.GeoDataFrame:
    """
    Filter background points by distance from presence points.

    Uses ``scipy.spatial.cKDTree`` for nearest presence distance (projected
    Point XY). Avoids the O(P×B) pairwise matrix previously built with
    ``shapely.distance``.
    
    Args:
        background_data: Background points
        presence_data: Presence points
        d_min: Minimum distance from presence (meters)
        d_max: Maximum distance from presence (meters)
        
    Returns:
        Filtered background points
    """
    if len(presence_data) == 0:
        return background_data
    if len(background_data) == 0:
        return background_data

    presence_xy = np.column_stack(
        [
            presence_data.geometry.x.to_numpy(dtype=float),
            presence_data.geometry.y.to_numpy(dtype=float),
        ]
    )
    background_xy = np.column_stack(
        [
            background_data.geometry.x.to_numpy(dtype=float),
            background_data.geometry.y.to_numpy(dtype=float),
        ]
    )
    p_dist, _ = cKDTree(presence_xy).query(background_xy, k=1)

    filtered = background_data[
        (p_dist >= d_min) & (p_dist <= d_max)
    ]
    
    return filtered


def grid_sample_points(
    gdf: gpd.GeoDataFrame,
    stratify_cols: Optional[List[str]] = None,
    grid_size_x: float = 100,
    grid_size_y: float = 100,
    *,
    random_state: int,
) -> gpd.GeoDataFrame:
    """
    Sample one point per grid cell, optionally stratified by columns.
    
    Args:
        gdf: GeoDataFrame to sample
        stratify_cols: Columns to stratify by (one point per cell per combination)
        grid_size_x: Grid cell size in x direction
        grid_size_y: Grid cell size in y direction
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled GeoDataFrame
    """
    gdf = gdf.copy()
    centroids = gdf.geometry.centroid
    
    x_bins = (centroids.x // grid_size_x).astype(int)
    y_bins = (centroids.y // grid_size_y).astype(int)
    
    gdf["_x_bin"] = x_bins
    gdf["_y_bin"] = y_bins
    
    if stratify_cols is None:
        stratify_cols = []
    
    sampled = gdf.groupby(stratify_cols + ["_x_bin", "_y_bin"], group_keys=False).apply(
        lambda x: x.sample(1, random_state=random_state)
    )
    sampled.drop(columns=["_x_bin", "_y_bin"], inplace=True)
    
    return sampled


def apply_sample_weights(
    gdf: gpd.GeoDataFrame,
    n_neighbors: int = 10,
) -> gpd.GeoDataFrame:
    """
    Apply distance-based sample weights to points.
    
    Args:
        gdf: GeoDataFrame with geometry
        n_neighbors: Number of neighbors for distance weighting
        
    Returns:
        GeoDataFrame with sample_weight column added
    """
    gdf = gdf.copy()
    gdf["sample_weight"] = ela.distance_weights(
        gdf.geometry,
        n_neighbors=n_neighbors,
    )
    # elapid returns NaN when n_points < n_neighbors (or all distances equal);
    # fill so a later dropna does not erase real post-thinning presence cells.
    gdf["sample_weight"] = gdf["sample_weight"].fillna(1.0)
    return gdf


def prepare_species_training_data(
    presence_data: gpd.GeoDataFrame,
    background_data: gpd.GeoDataFrame,
    latin_name: str,
    activity_type: str,
    ev_columns: List[str],
    grid_size_m: float = 500,
    d_min: float = 500,
    d_max: float = np.inf,
    n_max_background: Optional[int] = None,
    sort_density: bool = False,
    sample_weight_n_neighbors: int = 10,
    *,
    random_state: int,
) -> gpd.GeoDataFrame:
    """
    Prepare training data for a single species-activity combination.
    
    This function composes multiple smaller functions to:
    1. Filter presence data for species/activity
    2. Drop duplicates
    3. Filter background by distance
    4. Grid sample both presence and background
    5. Apply sample weights
    6. Combine into final training dataset
    
    Args:
        presence_data: All annotated presence data
        background_data: All annotated background data
        latin_name: Species latin name
        activity_type: Activity type
        ev_columns: List of environmental variable column names
        grid_size_m: Grid cell size for sampling (meters)
        d_min: Minimum distance from presence for background (meters)
        d_max: Maximum distance from presence for background (meters)
        n_max_background: Maximum number of background points (defaults to 10x presence)
        sort_density: Whether to sort background by density before sampling
        sample_weight_n_neighbors: Number of neighbors for sample weighting
        random_state: Random seed for reproducibility in sampling operations
        
    Returns:
        Combined training data GeoDataFrame with class, sample_weight, and EV columns
    """
    # Filter presence data
    species_presence = get_species_presence_data(
        presence_data, latin_name, activity_type
    )
    
    if len(species_presence) == 0:
        logger.warning(f"No presence data for {latin_name} - {activity_type}")
        return gpd.GeoDataFrame()
    
    # Drop duplicates
    species_presence = drop_duplicate_grid_points(species_presence)
    
    # Filter background by distance
    species_absence = filter_background_by_distance(
        background_data, species_presence, d_min=d_min, d_max=d_max
    )
    species_absence = drop_duplicate_grid_points(species_absence)
    
    # Grid sample
    species_presence = grid_sample_points(
        species_presence, grid_size_x=grid_size_m, grid_size_y=grid_size_m, random_state=random_state
    )
    species_absence = grid_sample_points(
        species_absence, grid_size_x=grid_size_m, grid_size_y=grid_size_m, random_state=random_state
    )
    
    # Limit background points
    if n_max_background is None:
        n_max_background = len(species_presence) * 10
    

    
    if sort_density and "density" in species_absence.columns:
        species_absence = species_absence.sort_values(
            by="density", ascending=False
        ).head(n_max_background)
    else:
        species_absence = species_absence.sample(
            n=min(len(species_absence), n_max_background), replace=False, random_state=random_state
        )
    
    # Apply sample weights
    species_presence = apply_sample_weights(
        species_presence, n_neighbors=sample_weight_n_neighbors
    )
    species_absence = apply_sample_weights(
        species_absence, n_neighbors=sample_weight_n_neighbors
    )
    
    # Combine and select columns
    species_data = pd.concat([species_presence, species_absence])
    columns = ["class", "sample_weight", "geometry"] + ev_columns
    species_data = species_data[columns]

    # Drop only incomplete environmental variables — never drop rows solely
    # because sample_weight was NaN (that hid real post-thinning cell counts).
    species_data = species_data.dropna(subset=ev_columns)
    
    return species_data


def generate_training_data(
    bats_ant: gpd.GeoDataFrame,
    background_points_gdf: gpd.GeoDataFrame,
    background_density_series: Any,
    grid_points: Optional[gpd.GeoDataFrame],
    latin_names: List[str],
    activity_types: List[str],
    ev_columns: List[str],
    min_presence: int,
    subset: Optional[int],
    subset_background: bool,
    order_by_density_for_subset: bool,
    sample_weight_n_neighbors: int,
    background_min_bg: int,
    background_max_bg: int,
    background_factor: int,
    model_config: MaxentConfig,
    feature_selection: Dict[ActivityType, List[str]],
    grid_size_m: float = 2000,
    *,
    seed: int,
) -> List[TrainingData]:
    """Build ``TrainingData`` rows for each species × activity (shared annotated tables)."""
    _ = grid_points, subset, subset_background, background_density_series  # reserved / future use

    def _model_features_for(activity_str: str) -> List[str]:
        for act, feats in feature_selection.items():
            if act.value == activity_str:
                use = [f for f in feats if f in ev_columns]
                return use if use else list(ev_columns)
        return list(ev_columns)

    training_data: List[TrainingData] = []
    for latin_name, activity_type in product(latin_names, activity_types):
        n_presence_est = len(
            bats_ant[
                (bats_ant["latin_name"] == latin_name)
                & (bats_ant["activity_type"] == activity_type)
            ]
        )
        n_max_background = min(
            background_max_bg,
            max(background_min_bg, n_presence_est * background_factor),
        )
        species_data = prepare_species_training_data(
            presence_data=bats_ant,
            background_data=background_points_gdf,
            latin_name=latin_name,
            activity_type=activity_type,
            ev_columns=ev_columns,
            grid_size_m=grid_size_m,
            d_min=500,
            d_max=np.inf,
            n_max_background=n_max_background,
            sort_density=order_by_density_for_subset,
            sample_weight_n_neighbors=sample_weight_n_neighbors,
            random_state=seed,
        )
        if len(species_data) == 0 or "class" not in species_data.columns:
            continue
        n_presence = len(species_data[species_data["class"] == 1])
        if n_presence < min_presence:
            continue
        training_data.append(
            TrainingData(
                latin_name=latin_name,
                activity_type=activity_type,
                occurrence=species_data,
                maxent_config=model_config,
                model_features=_model_features_for(activity_type),
            )
        )
    return training_data


# ============================================================================
# Model training functions
# ============================================================================


def _summarize_cv_scores(
    cv_scores: Optional[np.ndarray],
) -> Tuple[float, float, int, int]:
    """Return (mean, std, n_valid, n_total) for a CV score array, safely handling NaNs."""
    if cv_scores is None or len(cv_scores) == 0:
        return np.nan, np.nan, 0, 0
    
    valid_scores = cv_scores[~np.isnan(cv_scores)]
    n_total = len(cv_scores)
    n_valid = len(valid_scores)
    
    if n_valid == 0:
        return np.nan, np.nan, 0, n_total
    
    return float(valid_scores.mean()), float(valid_scores.std()), n_valid, n_total


def train_single_model(
    data: TrainingData,
    max_threads_per_model: int,
    n_cv_folds: int = 3,
    min_presence: int = 15,
    *,
    seed: int,
    fold_params: Optional[FoldParams] = None,
) -> TrainingResults:
    """Train a single MaxEnt model for a given set of training data."""
    try:
        apply_training_seed(seed)
        activity_type = ActivityType(data.activity_type)
        latin_name = data.latin_name
        logger.debug(f"Training model for {latin_name} ({activity_type.value})...")

        resolved_folds = fold_params or resolve_fold_params(CvConfig(), model_seed=seed)
        model_config = data.maxent_config
        model_features = data.model_features
        
        # Verify features list is not empty
        if len(model_features) == 0:
            raise ValueError(
                f"model_features list is empty for {latin_name} - {activity_type}. "
                f"At least one feature must be provided."
            )
        
        # Verify all required features are in the data
        missing_features = list(set(model_features) - set(data.occurrence.columns))
        if missing_features:
            raise ValueError(
                f"Missing features {missing_features} in occurrence data for {latin_name} - {activity_type}. "
                f"Available columns: {list(data.occurrence.columns)}"
            )
        
        # Create model with appropriate thread count
        model = create_maxent_pipeline(
            feature_names=model_features,
            maxent_n_jobs=max_threads_per_model,
            model_config=model_config,
        )
        logger.info(f"Model features: {model_features}")
        n_presence = len(data.occurrence[data.occurrence["class"] == 1])
        if resolved_folds.method is FoldMethod.BUFFERED_LOO:
            loo_warning = buffered_loo_presence_warning(
                n_presence,
                resolved_folds.buffered_loo_warn_above,
                latin_name=latin_name,
                activity_type=activity_type.value,
            )
            if loo_warning:
                logger.warning(loo_warning)
        if resolved_folds.method is FoldMethod.KMEANS:
            # Prefer explicit fold_params; otherwise honour the n_cv_folds argument.
            effective_n_folds = (
                resolved_folds.n_folds if fold_params is not None else n_cv_folds
            )
            # do 2 folds if average presence per fold would fall below min_presence
            if effective_n_folds > 0 and (n_presence // effective_n_folds) < min_presence:
                effective_n_folds = 2
                logger.info(
                    "Using 2 folds for %s - %s because n_presence < %s",
                    latin_name,
                    activity_type.value,
                    min_presence,
                )
        elif resolved_folds.method is FoldMethod.BLOCKS:
            effective_n_folds = resolved_folds.n_folds
        elif resolved_folds.method is FoldMethod.BUFFERED_LOO:
            effective_n_folds = int(n_presence)
        else:
            raise ValueError(f"unknown fold method: {resolved_folds.method!r}")

        effective_params = FoldParams(
            method=resolved_folds.method,
            n_folds=effective_n_folds,
            random_state=resolved_folds.random_state,
            block_size_m=resolved_folds.block_size_m,
            block_n_tries=resolved_folds.block_n_tries,
            buffer_m=resolved_folds.buffer_m,
            buffered_loo_warn_above=resolved_folds.buffered_loo_warn_above,
        )
        fold_presence = presence_counts_per_fold(data.occurrence, effective_params)

        final_model, cv_models, cv_scores, validation_scores = evaluate_and_train_maxent_model(
            model=model,
            occurrence_gdf=data.occurrence,
            n_cv_folds=effective_n_folds,
            metric_fn=roc_auc_score,
            feature_columns=model_features,  # Explicitly pass feature columns to avoid using extra columns as features
            collect_validation_scores=True,
            random_state=effective_params.random_state,
            fold_method=effective_params.method,
            block_size_m=effective_params.block_size_m,
            block_n_tries=effective_params.block_n_tries,
            buffer_m=effective_params.buffer_m,
        )
        
        if final_model is None:
            raise ValueError("Model training failed - final_model is None")
        
        cv_mean, cv_std, n_valid, n_total = _summarize_cv_scores(cv_scores)
        if n_valid > 0:
            logger.debug(
                "✓ %s - %s: CV AUC = %.4f ± %.4f (%d/%d folds valid)",
                latin_name,
                activity_type.value,
                cv_mean,
                cv_std,
                n_valid,
                n_total,
            )
        else:
            logger.debug(
                "✗ %s - %s: No valid CV scores", latin_name, activity_type.value
            )
        logger.info(
            "%s - %s: fold_method=%s scored %d/%d folds; presence per fold=%s",
            latin_name,
            activity_type.value,
            effective_params.method.value,
            n_valid,
            n_total,
            fold_presence,
        )
        
        return TrainingResults(
            latin_name=latin_name,
            activity_type=activity_type.value,
            final_model=final_model,
            cv_models=cv_models if cv_models is not None else None,
            cv_scores=cv_scores,
            validation_scores=validation_scores,
            cv_n_folds=effective_n_folds if effective_n_folds else n_total,
            fold_method=effective_params.method.value,
            fold_params=effective_params.to_metadata(),
            fold_presence_counts=fold_presence,
            success=True,
            error=None,
        )
    except Exception as e:
        logger.error(
            f"Error training model for {data.latin_name} - {data.activity_type}: {e}"
        )
        return TrainingResults(
            latin_name=data.latin_name,
            activity_type=data.activity_type,
            final_model=None,
            cv_models=None,
            cv_scores=None,
            success=False,
            error=str(e),
        )


def train_models_parallel(
    training_data: List[TrainingData],
    max_threads_per_model: int = 1,
    n_jobs: Optional[int] = None,
    *,
    seed: int,
    fold_params: Optional[FoldParams] = None,
) -> List[TrainingResults]:
    """Train MaxEnt models in parallel for each set of training data.
    
    Args:
        training_data: List of TrainingData objects, one per species-activity combination.
            Each TrainingData must include maxent_config and model_features.
        max_threads_per_model: Maximum threads per model (default 1, parallelization handled at higher level)
        n_jobs: Number of parallel jobs (None = auto)
        
    Returns:
        List of TrainingResults
    """
    # Calculate optimal number of jobs if not specified
    if n_jobs is None:
        total_cpus = os.cpu_count()
        if total_cpus is None:
            total_cpus = 1.0
        # Use 80% of available CPUs by default
        n_jobs = max(1, int(total_cpus * 0.8) // max_threads_per_model)
    
    logger.debug(
        f"Training with {n_jobs} parallel jobs, {max_threads_per_model} threads per model"
    )
    
    # Execute training in parallel using ProcessPoolExecutor
    results: List[TrainingResults] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Create a list of futures
        futures = []
        for data in training_data:
            identifier = data.identifier()
            
            futures.append(
                executor.submit(
                    train_single_model,
                    data,
                    max_threads_per_model,
                    seed=seed,
                    fold_params=fold_params,
                )
            )
        
        # Collect results as they complete
        for future in tqdm(futures, total=len(futures), desc="Training models"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel training: {e}")
                # Create a failed result
                results.append(
                    TrainingResults(
                        latin_name="unknown",
                        activity_type="unknown",
                        final_model=None,
                        cv_models=None,
                        cv_scores=None,
                        success=False,
                        error=str(e),
                    )
                )
    
    # Extract successful results
    successful_results = [r for r in results if r.success]
    
    # Report failed models
    failed_models = [
        f"{r.latin_name} - {r.activity_type}: {r.error}"
        for r in results
        if not r.success
    ]
    if failed_models:
        logger.warning("Failed models:")
        for failure in failed_models:
            logger.warning(f"  {failure}")
    
    logger.info(
        f"Successfully trained {len(successful_results)} models out of {len(training_data)} attempts"
    )
    
    # Print summary of model accuracies
    if successful_results:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        for result in successful_results:
            cv_mean, cv_std, n_valid, n_total = _summarize_cv_scores(result.cv_scores)
            if n_valid > 0:
                logger.info(
                    "  %s - %s: AUC = %.4f ± %.4f (%d/%d folds valid)",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                    cv_mean,
                    cv_std,
                    n_valid,
                    n_total,
                )
            else:
                logger.info(
                    "  %s - %s: No valid scores",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                )
        logger.info("=" * 80 + "\n")
    
    return successful_results


def _finite_float_or_none(x: Any) -> Any:
    """JSON-friendly float: map NaN/inf to None."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def _json_safe(value: Any) -> Any:
    """Convert values for JSON serialization (numpy scalars, etc.)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


_MAXENT_CONFIG_KEYS = (
    "feature_types",
    "tau",
    "transform",
    "clamp",
    "scorer",
    "beta_multiplier",
    "beta_lqp",
    "beta_hinge",
    "beta_threshold",
    "beta_categorical",
    "n_hinge_features",
    "n_threshold_features",
    "convergence_tolerance",
    "tolerance",
    "use_lambdas",
    "n_lambdas",
    "class_weights",
    "n_cpus",
    "use_sklearn",
)


def _maxent_config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Serialize MaxEnt / elapid config objects to plain dicts for ``package.json``."""
    out: Dict[str, Any] = {}
    for key in _MAXENT_CONFIG_KEYS:
        if not hasattr(cfg, key):
            continue
        val = getattr(cfg, key)
        if callable(val):
            continue
        out[key] = _json_safe(val)
    return out


def prepare_results_dataframe(
    models: List[TrainingResults],
    training_data: List[TrainingData],
) -> pd.DataFrame:
    """Prepare a DataFrame with model results."""
    results = []
    for model, data in zip(models, training_data):
        mean_cv, std_cv, _, _ = _summarize_cv_scores(model.cv_scores)
        results.append(
            {
                "identifier": model.identifier(),
                "latin_name": model.latin_name,
                "activity_type": model.activity_type,
                "mean_cv_score": mean_cv,
                "std_cv_score": std_cv,
                "n_presence": len(data.occurrence[data.occurrence["class"] == 1]),
                "n_background": len(data.occurrence[data.occurrence["class"] == 0]),
            }
        )
    return pd.DataFrame(results)


def save_training_data(
    training_data: List[TrainingData],
    output_dir: Path,
) -> Path:
    """
    Combine the training data into a single parquet file with an identifier column.
    
    Args:
        training_data: List of TrainingData objects
        output_dir: Path to output directory
        
    Returns:
        Path to the saved training data file
    """
    training_data_path = output_dir / "training_data.parquet"
    
    # Add identifier to each DataFrame and combine
    dfs = []
    for data in training_data:
        df = data.occurrence.copy()
        df["identifier"] = data.identifier()
        dfs.append(df)
    
    training_data_df = pd.concat(dfs, ignore_index=True)
    training_data_df.to_parquet(training_data_path)
    return training_data_path


def save_models(
    models: List[TrainingResults],
    training_data: List[TrainingData],
    output_dir: Path,
    *,
    trainer: str = "maxent",
    deepmaxent_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Save each successful model as a small on-disk package for training outputs and API upload.

    Layout per species–activity (filesystem-safe ``model_id`` from ``get_model_id``)::

        {output_dir}/{model_id}/model.pkl      — fitted sklearn pipeline
        {output_dir}/{model_id}/package.json — feature list, MaxEnt params, CV metrics, counts

    Args:
        models: Training results (only entries with ``final_model`` are written).
        training_data: Parallel list of training data (matched by ``identifier()``).
        output_dir: Root directory for all model packages.
        trainer: ``maxent`` (default) or ``deepmaxent``.
        deepmaxent_metadata: Optional joint-model fields recorded in each package.

    Returns:
        Map from ``TrainingResults.identifier()`` to path of ``model.pkl`` inside the package
        directory (suitable for existing loaders that open the pickle path).
    """
    data_by_id = {d.identifier(): d for d in training_data}
    model_paths: Dict[str, Path] = {}
    for model in models:
        mid = model.identifier()
        if model.final_model is None:
            logger.warning("Skipping save for %s (no final_model)", mid)
            continue
        data = data_by_id.get(mid)
        if data is None:
            logger.warning(
                "Skipping save for %s (no matching TrainingData for identifier)", mid
            )
            continue

        pkg_dir = output_dir / get_model_id([model.latin_name, model.activity_type])
        pkg_dir.mkdir(parents=True, exist_ok=True)
        model_path = pkg_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model.final_model, f)

        mean_cv, std_cv, n_valid, n_total = _summarize_cv_scores(model.cv_scores)
        n_presence = int(len(data.occurrence[data.occurrence["class"] == 1]))
        n_background = int(len(data.occurrence[data.occurrence["class"] == 0]))
        package: Dict[str, Any] = {
            "schema_version": 1,
            "identifier": mid,
            "model_id": get_model_id([model.latin_name, model.activity_type]),
            "latin_name": model.latin_name,
            "activity_type": model.activity_type,
            "model_limited": bool(data.model_limited),
            "trainer": trainer,
            "feature_names": list(data.model_features),
            "maxent_config": _maxent_config_to_dict(data.maxent_config),
            "metrics": {
                "mean_cv_auc": _finite_float_or_none(mean_cv),
                "std_cv_auc": _finite_float_or_none(std_cv),
                "n_cv_folds": model.cv_n_folds,
                "n_cv_folds_valid": n_valid,
                "n_cv_folds_total": n_total,
                "n_presence": n_presence,
                "n_background": n_background,
                "training_success": model.success,
                "fold_method": model.fold_method,
                "fold_presence_counts": model.fold_presence_counts,
            },
            "fold": model.fold_params,
            "artifacts": {"model_pickle": "model.pkl"},
        }
        if deepmaxent_metadata:
            package["deepmaxent"] = _json_safe(deepmaxent_metadata)
        if model.validation_scores is not None and len(model.validation_scores) > 0:
            validation_path = pkg_dir / "validation_scores.parquet"
            model.validation_scores.to_parquet(validation_path, index=False)
            package["artifacts"]["validation_scores"] = "validation_scores.parquet"

        # Per-model fold assignments match MaxEnt occurrence rows. Joint DeepMaxent
        # validation indices refer to the shared site table — skip here.
        if trainer == "maxent" and model.fold_method and model.fold_params:
            assignments = fold_assignments_frame(
                data.occurrence,
                FoldParams(
                    method=FoldMethod(model.fold_method),
                    n_folds=int(
                        model.cv_n_folds or model.fold_params.get("n_folds") or 3
                    ),
                    random_state=int(model.fold_params.get("seed", 0) or 0),
                    block_size_m=float(
                        model.fold_params.get("block_size_m", 10_000.0)
                    ),
                    block_n_tries=int(
                        model.fold_params.get("block_n_tries", 500)
                    ),
                    buffer_m=float(model.fold_params.get("buffer_m", 10_000.0)),
                    buffered_loo_warn_above=int(
                        model.fold_params.get(
                            "buffered_loo_warn_above",
                            DEFAULT_BUFFERED_LOO_WARN_ABOVE,
                        )
                    ),
                ),
            )
            fold_path = pkg_dir / "fold_assignments.parquet"
            assignments.to_parquet(fold_path, index=False)
            package["artifacts"]["fold_assignments"] = "fold_assignments.parquet"
        with open(pkg_dir / "package.json", "w", encoding="utf-8") as jf:
            json.dump(package, jf, indent=2, allow_nan=False)

        model_paths[mid] = model_path
        logger.info("Saved model package for %s under %s", mid, pkg_dir)

    return model_paths



def save_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save results DataFrame to disk.
    
    Args:
        results_df: DataFrame containing model results
        output_dir: Path to output directory
        
    Returns:
        None
    """
    results_df.to_csv(output_dir / "model_results.csv", index=False)


def save_run_thresholds(
    models: List[TrainingResults],
    training_data: List[TrainingData],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Write pooled OOF-percentile thresholds into the run folder (METHOD.md).

    ``models`` / ``training_data`` supply per-model ``model_limited`` flags;
    cut-offs are computed from on-disk ``validation_scores.parquet``.
    """
    del models
    limited_by_id = {
        get_model_id([data.latin_name, data.activity_type]): bool(data.model_limited)
        for data in training_data
    }
    rows = write_run_thresholds(
        output_dir,
        model_limited_by_id=limited_by_id,
    )
    if not rows:
        logger.warning("No models available to write run thresholds under %s", output_dir)
        return []
    logger.info(
        "Wrote suitability thresholds for %d model(s) under %s",
        len(rows),
        output_dir,
    )
    return rows


def save_run_metadata(
    output_dir: Path,
    *,
    seed: int,
    model_config_path: str,
    fold_params: Optional[FoldParams] = None,
    trainer: str = "maxent",
    device: Optional[str] = None,
    deepmaxent: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write run-level metadata (including the training seed) into the run folder."""
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "model_config_path": model_config_path,
        "trainer": trainer,
    }
    if device is not None:
        metadata["device"] = device
    if deepmaxent is not None:
        metadata["deepmaxent"] = _json_safe(deepmaxent)
    if fold_params is not None:
        metadata["cv"] = fold_params.to_metadata()
    path = output_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path



# ============================================================================
# Main training function
# ============================================================================


@dataclass
class TrainingSetup:
    """Container for shared training data setup."""
    project_config: ProjectConfig
    annotated_bats: gpd.GeoDataFrame
    annotated_background: gpd.GeoDataFrame
    background_density: Any  # xr.DataArray
    grid_points: Optional[gpd.GeoDataFrame]
    ev_columns: List[str]
    all_ev_columns: List[str]
    ev_raster_path: Path
    latin_names: List[str]
    activity_types: List[str]


def setup_training_data(
    project_config_path: Path,
    variables_config_path: Optional[Path],
    bats_file: Optional[Path] = None,
    background_file: Optional[Path] = None,
    ev_file: Optional[Path] = None,
    grid_points_file: Optional[Path] = None,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> TrainingSetup:
    """Set up shared training data that can be reused across multiple training runs.
    
    This function loads and prepares all the data that doesn't change between trials,
    such as annotated occurrence data, environmental variables, and grid points.
    
    Args:
        project_config_path: Path to project-level config
        variables_config_path: Path to variables config (roster + per-activity features)
        bats_file: Path to bat data file (optional, uses config default if not provided)
        background_file: Path to background points file (optional)
        ev_file: Path to environmental variables file (optional)
        grid_points_file: Path to grid points file (optional)
        species: List of species to filter to (optional)
        activity_types: List of activity types to filter to (optional)
        verbose: Enable verbose logging
        
    Returns:
        TrainingSetup object containing all shared data
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.debug("=== Setting up shared training data ===")
    
    # Load project config
    project_config = load_project_config(project_config_path)
    
    # Resolve variables config path
    if variables_config_path is None:
        variables_config_path = Path(project_config.paths.variables_config_path)
    variables_cfg = load_variables_config(
        resolve_sdm_config_root(variables_config_path)
    )
    
    # Resolve data paths from config if not explicitly provided
    bats_uri = resolve_data_path(
        bats_file if bats_file is not None else project_config.paths.occurence_data
    )
    background_uri = resolve_data_path(
        background_file
        if background_file is not None
        else project_config.paths.background_points
    )
    ev_uri = resolve_data_path(
        ev_file if ev_file is not None else project_config.paths.ev_tiff
    )
    grid_points_uri = resolve_data_path(
        grid_points_file
        if grid_points_file is not None
        else project_config.paths.grid_points
    )

    # Load data
    logger.debug("Loading bat occurrence data...")
    with local_materialized_path(bats_uri) as bats_path:
        bats_ant = load_bat_data(bats_path)

    logger.debug("Loading background points...")
    with local_materialized_path(background_uri) as background_path:
        background, background_density = load_background_points(background_path)

    logger.debug("Loading environmental variables...")
    ev_data, ev_raster_path = load_environmental_variables(ev_uri)
    all_ev_columns = list(ev_data.data_vars.keys())
    
    # Filter environmental variables based on roster
    if variables_cfg.roster:
        roster_matches = [col for col in all_ev_columns if col in variables_cfg.roster]
        if roster_matches:
            ev_columns = roster_matches
            logger.debug(
                "Filtered environmental variables to %d columns based on roster",
                len(ev_columns),
            )
        else:
            ev_columns = all_ev_columns
            logger.warning(
                "No roster variables matched available environmental variables; "
                "using full set (%d columns).",
                len(ev_columns),
            )
    else:
        ev_columns = all_ev_columns
        logger.debug(f"Found {len(ev_columns)} environmental variables")
    
    # Load grid points
    logger.debug("Loading grid points...")
    if not grid_points_uri or not uri_exists(grid_points_uri):
        grid_points = extract_grid_points(ev_data)
    else:
        with local_materialized_path(grid_points_uri) as grid_points_path:
            grid_points = gpd.read_parquet(grid_points_path)
    
    # Annotate points with environmental variables (skip if already annotated)
    logger.debug("Annotating points with environmental variables...")
    if all(col in bats_ant.columns for col in all_ev_columns[:5]):  # Check first 5 as a sample
        logger.debug("Data appears to be already annotated, skipping annotation")
        annotated_bats_gdf = bats_ant
        annotated_background_gdf = background
    else:
        annotated_bats_gdf, annotated_background_gdf = annotate_points(
            bats_ant, background, ev_raster_path, all_ev_columns
        )
    
    # Filter feature columns
    def _filter_feature_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        base_cols = [col for col in gdf.columns if col not in all_ev_columns]
        selected_feature_cols = [col for col in ev_columns if col in gdf.columns]
        return gdf[base_cols + selected_feature_cols]
    
    annotated_bats_gdf = _filter_feature_columns(annotated_bats_gdf)
    annotated_background_gdf = _filter_feature_columns(annotated_background_gdf)
    
    # Filter species and activity types if specified
    if species is not None:
        logger.debug(f"Filtering to species: {', '.join(species)}")
        annotated_bats_gdf = annotated_bats_gdf[annotated_bats_gdf.latin_name.isin(species)]
    
    if activity_types is not None:
        logger.debug(f"Filtering to activity types: {', '.join(activity_types)}")
        annotated_bats_gdf = annotated_bats_gdf[annotated_bats_gdf.activity_type.isin(activity_types)]
    
    latin_names = cast(List[str], annotated_bats_gdf.latin_name.unique().tolist())
    activity_types_list = cast(List[str], annotated_bats_gdf.activity_type.unique().tolist())
    
    logger.debug(
        f"Setup complete: {len(latin_names)} species × {len(activity_types_list)} activity types"
    )
    
    return TrainingSetup(
        project_config=project_config,
        annotated_bats=annotated_bats_gdf,
        annotated_background=annotated_background_gdf,
        background_density=background_density,
        grid_points=grid_points,
        ev_columns=ev_columns,
        all_ev_columns=all_ev_columns,
        ev_raster_path=ev_raster_path,
        latin_names=latin_names,
        activity_types=activity_types_list,
    )


def train_models_with_setup(
    setup: TrainingSetup,
    model_config: MaxentConfig,
    feature_selection: Dict[ActivityType, List[str]],
    sampling_params: Dict[str, Any],
    min_presence: int = 15,
    max_threads_per_model: int = 2,
    n_jobs: Optional[int] = None,
    n_cv_folds: int = 3,
    verbose: bool = False,
    *,
    seed: int,
    fold_params: Optional[FoldParams] = None,
) -> tuple[List[TrainingResults], List[TrainingData]]:
    """Train models using pre-setup shared data.
    
    This function generates training data and trains models using the shared setup
    data and trial-specific parameters (hyperparameters, sampling, features).
    
    Args:
        setup: TrainingSetup object with shared data
        model_config: MaxEnt model configuration
        feature_selection: Dictionary mapping activity types to feature lists
        sampling_params: Dictionary of sampling parameters (subset, background, etc.)
        min_presence: Minimum number of presence records required
        max_threads_per_model: Maximum threads per model
        n_jobs: Number of parallel jobs
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (List of TrainingResults, List of TrainingData)
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.debug("=== Training models with shared setup ===")
    
    # Filter feature selection to only include available features
    filtered_feature_selection: Dict[ActivityType, List[str]] = {}
    for activity, features in feature_selection.items():
        filtered = [feat for feat in features if feat in setup.ev_columns]
        if not filtered:
            logger.warning(
                "No valid features remain for %s; defaulting to %d available variables",
                activity.value,
                len(setup.ev_columns),
            )
            filtered = setup.ev_columns
        filtered_feature_selection[activity] = filtered

    logger.debug("Generating training data...")
    training_data = generate_training_data(
        bats_ant=cast(gpd.GeoDataFrame, setup.annotated_bats),
        background_points_gdf=cast(gpd.GeoDataFrame, setup.annotated_background),
        background_density_series=setup.background_density,
        grid_points=setup.grid_points,
        latin_names=setup.latin_names,
        activity_types=setup.activity_types,
        ev_columns=setup.ev_columns,
        min_presence=min_presence,
        subset=sampling_params.get("subset_occurrence"),
        subset_background=sampling_params.get("subset_background", True),
        order_by_density_for_subset=sampling_params.get("order_by_density_for_subset", True),
        sample_weight_n_neighbors=sampling_params.get("sample_weight_n_neighbors", 5),
        background_min_bg=sampling_params.get("background_min_bg", 1000),
        background_max_bg=sampling_params.get("background_max_bg", 10000),
        background_factor=sampling_params.get("background_factor", 10),
        model_config=model_config,
        feature_selection=filtered_feature_selection,
        grid_size_m=float(sampling_params.get("grid_size_m", 2000)),
        seed=seed,
    )

    logger.debug("Training %d models...", len(training_data))
    models = train_models_parallel(
        training_data,
        max_threads_per_model=max_threads_per_model,
        n_jobs=n_jobs,
        seed=seed,
        fold_params=fold_params,
    )
    
    return models, training_data


def train_sdm_models(
    project_config_path: Path = CONFIG_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    variables_config_path: Optional[Path] = None,
    config_dir: Optional[Path] = None,
    bats_file: Optional[Path] = None,
    ev_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_presence: Optional[int] = None,
    n_jobs: Optional[int] = None,
    max_threads_per_model: int = 1,  # Use 1 thread per model (parallelization handled at higher level)
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False,
    # Species-specific processing parameters
    d_min: float = 500,
    d_max: float = np.inf,
    sample_weight_n_neighbors: int = 10,
    allow_config_fallback: bool = False,
    cv_method: Optional[str] = None,
    cv_n_folds: Optional[int] = None,
    cv_block_size_m: Optional[float] = None,
    cv_block_n_folds: Optional[int] = None,
    cv_buffer_m: Optional[float] = None,
    cv_buffered_loo_warn_above: Optional[int] = None,
) -> pd.DataFrame:
    """Run the MaxEnt model training pipeline using the new modular approach.
    
    This function uses a modular approach where:
    - Background points are generated once and shared across all species-activity combinations
    - Environmental variables are converted to GeoDataFrame and joined using spatial joins
    - Each species-activity combination is processed independently
    - Per-species-activity configs are loaded from config_dir if provided, with fallback to base configs
    
    Args:
        project_config_path: Path to project-level config (paths, spatial, etc.)
        model_config_path: Path to base model hyperparameter config (used as fallback)
        variables_config_path: Path to base variables config (used as fallback)
        config_dir: Optional directory containing per-species-activity tuning configs
            (if provided, configs are loaded from {config_dir}/{latin_name}_{activity_type}/)
        bats_file: Path to bat data file
        ev_file: Path to environmental variables file
        output_dir: Model output root (each run writes under ``runs/<timestamp-id>/``)
        min_presence: Minimum number of presence records required
        n_jobs: Number of parallel jobs
        max_threads_per_model: Maximum threads per model
        species: List of species to model
        activity_types: List of activity types to model
        verbose: Enable verbose logging
        d_min: Minimum distance from presence for background (meters)
        d_max: Maximum distance from presence for background (meters)
        sample_weight_n_neighbors: Number of neighbors for sample weighting
        allow_config_fallback: If true, missing per-model configs use the base
            ``variables_config.yml`` roster instead of failing (default: false).
        
    Note:
        Background point generation parameters are loaded from model_config.yml
        under the 'background' key. See BackgroundConfig for available options.
        
    Returns:
        DataFrame containing model results
        
    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If no valid models can be trained
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("=== Starting SDM Model Training Pipeline (Modular Approach) ===")
    
    # Load configs
    project_config = load_project_config(project_config_path)
    base_model_cfg = load_model_config(resolve_sdm_config_root(model_config_path))

    if variables_config_path is None:
        variables_config_path = Path(project_config.paths.variables_config_path)
    base_variables_cfg = load_variables_config(
        resolve_sdm_config_root(variables_config_path)
    )
    
    # Get background config from model config (with defaults if not present)
    background_config = base_model_cfg.background or BackgroundConfig()
    # BackgroundConfig has default enum values, but add assertions for type checker
    assert background_config.background_method is not None
    assert background_config.transform_method is not None
    logger.info(
        f"Background point config: n={background_config.n_background_points}, "
        f"method={background_config.background_method.value}, "
        f"value={background_config.background_value}, sigma={background_config.sigma}, "
        f"transform={background_config.transform_method.value}"
    )
    
    # Get sampling config from model config (with defaults if not present)
    sampling_config = base_model_cfg.sampling or SamplingConfig()
    grid_size_m = sampling_config.grid_size_m
    training_seed = base_model_cfg.seed
    fold_params = resolve_fold_params(
        base_model_cfg.cv,
        model_seed=training_seed,
        method_override=cv_method,
        n_folds_override=cv_n_folds,
        block_size_m_override=cv_block_size_m,
        block_n_folds_override=cv_block_n_folds,
        buffer_m_override=cv_buffer_m,
        buffered_loo_warn_above_override=cv_buffered_loo_warn_above,
    )
    logger.info(f"Sampling config: grid_size_m={grid_size_m}")
    logger.info("Training seed (from model_config.yml): %s", training_seed)
    logger.info(
        "CV folds: method=%s params=%s",
        fold_params.method.value,
        fold_params.to_metadata(),
    )
    
    mlflow_cfg = resolve_mlflow_config(project_config)
    mlflow_enabled = configure_mlflow(mlflow_cfg)
    
    # Load data
    logger.info("Loading input data...")
    
    bats_uri = resolve_data_path(
        bats_file if bats_file is not None else project_config.paths.occurence_data
    )
    ev_uri = resolve_data_path(
        ev_file if ev_file is not None else project_config.paths.ev_tiff
    )
    boundary_uri = resolve_data_path(project_config.paths.boundary)
    output_root = resolve_output_root(
        cli_output_dir=output_dir,
        config_models_path=project_config.paths.models,
    )

    logger.debug("Loading bat occurrence data...")
    with local_materialized_path(bats_uri) as bats_path:
        bats_ant = load_bat_data(bats_path)
    bats_ant = bats_ant[bats_ant.accuracy <= 100]  # Filter by accuracy

    logger.debug("Loading boundary...")
    with local_materialized_path(boundary_uri) as boundary_path:
        boundary = load_boundary(
            filepath=boundary_path, buffer_distance=0, target_crs=27700
        )
    boundary = simplify_boundary(boundary, tolerance=100)

    logger.debug("Loading environmental variables...")
    evs_to_model, _ = load_environmental_variables(ev_uri)
    
    # Keep every bat record for background sampling, so filtered runs match full runs
    all_bats_gdf = bats_ant

    # Filter species and activity types if specified
    if species is not None:
        logger.debug(f"Filtering to species: {', '.join(species)}")
        bats_ant = bats_ant[bats_ant.latin_name.isin(species)]
    
    if activity_types is not None:
        logger.debug(f"Filtering to activity types: {', '.join(activity_types)}")
        bats_ant = bats_ant[bats_ant.activity_type.isin(activity_types)]
    
    # Get unique combinations (after any filtering)
    latin_names = cast(List[str], bats_ant.latin_name.unique().tolist())
    activity_types_list = cast(List[str], bats_ant.activity_type.unique().tolist())
    logger.info(
        f"Training models for {len(latin_names)} species × {len(activity_types_list)} activity types = {len(latin_names) * len(activity_types_list)} combinations"
    )
    
    # Prepare shared training data (presence/background annotated with EVs)
    # Unpack background config for prepare_training_data (keeps interface simple for testing)
    assert background_config.background_method is not None
    assert background_config.transform_method is not None
    (
        presence_with_evs_gdf,
        background_with_evs_gdf,
        ev_columns,
    ) = prepare_training_data(
        occurrence_gdf=bats_ant,
        background_occurrence_gdf=all_bats_gdf,
        boundary=boundary,
        evs_to_model=evs_to_model,
        n_background_points=background_config.n_background_points,
        background_method=background_config.background_method,
        background_value=background_config.background_value,
        sigma=background_config.sigma,
        transform_method=background_config.transform_method,
        seed=training_seed,
    )
    
    # Create default model config from base config (used as fallback)
    default_model_config = DefaultMaxentConfig.from_config(
        config=base_model_cfg
    )
    
    # Generate training data for each species-activity combination
    logger.info("Generating training data for each species-activity combination...")
    training_data: List[TrainingData] = []
    filter_combinations = list(product(latin_names, activity_types_list))
    
    effective_min_presence = (
        min_presence if min_presence is not None else base_model_cfg.sampling.min_presence
    )
    
    for latin_name, activity_type in tqdm(
        filter_combinations, desc="Preparing training data"
    ):
        # Prepare training data (using shared presence/background data)
        species_data = prepare_species_training_data(
            presence_data=presence_with_evs_gdf,
            background_data=background_with_evs_gdf,
            latin_name=latin_name,
            activity_type=activity_type,
            ev_columns=ev_columns,
            grid_size_m=grid_size_m,
            d_min=d_min,
            d_max=d_max,
            sample_weight_n_neighbors=sample_weight_n_neighbors,
            random_state=training_seed,
        )
        
        # Check if training data was generated
        if len(species_data) == 0:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: No training data generated"
            )
            continue
        
        # Check if "class" column exists (should always exist if data was generated)
        if "class" not in species_data.columns:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Missing 'class' column in training data"
            )
            continue
        
        # Check minimum presence requirement
        n_presence = len(species_data[species_data["class"] == 1])
        if n_presence < effective_min_presence:
            logger.warning(
                f"Skipping {latin_name} - {activity_type}: Only {n_presence} presence records (minimum {effective_min_presence} required)"
            )
            continue
        
        identifier = get_model_id([latin_name, activity_type])
        maxent_model_config: DefaultMaxentConfig = default_model_config
        model_features_config: VariablesConfig
        model_limited = False

        def _apply_base_variables_fallback(reason: str) -> None:
            nonlocal maxent_model_config, model_features_config, model_limited
            logger.warning(
                "%s: %s; using base variables config from %s",
                identifier,
                reason,
                resolve_sdm_config_root(variables_config_path),
            )
            maxent_model_config = default_model_config
            model_features_config = base_variables_cfg.validate_features(ev_columns)
            model_limited = False

        if config_dir is None:
            model_features_config = base_variables_cfg.validate_features(ev_columns)
            logger.info(
                "%s: no config_dir; using base variables (%d features)",
                identifier,
                len(model_features_config.variables),
            )
        else:
            try:
                model_subdir = resolve_model_config_dir(
                    config_dir, identifier, must_exist=True
                )
                per_model_config_path, per_model_variables_path = (
                    model_config_file_paths(model_subdir)
                )
                model_config = load_model_config(
                    per_model_config_path,
                    fallback_seed=base_model_cfg.seed,
                )
                maxent_model_config = DefaultMaxentConfig.from_config(
                    config=model_config
                )
                model_limited = bool(model_config.model_limited)
                variables_config = load_variables_config(per_model_variables_path)
                model_features_config = variables_config.validate_features(
                    ev_columns
                )
            except FileNotFoundError as exc:
                if allow_config_fallback:
                    _apply_base_variables_fallback(str(exc))
                else:
                    raise
            except Exception as exc:
                if allow_config_fallback:
                    _apply_base_variables_fallback(str(exc))
                else:
                    raise RuntimeError(
                        f"Error loading per-model configs for {identifier}: {exc}"
                    ) from exc

            logger.info(
                "%s: training with %d features",
                identifier,
                len(model_features_config.variables),
            )
            if model_limited:
                logger.info(
                    "%s: model_limited=true (no suitability cut-off will be published)",
                    identifier,
                )

        training_data.append(
            TrainingData(
                latin_name=latin_name,
                activity_type=activity_type,
                occurrence=species_data,
                maxent_config=maxent_model_config,
                model_features=model_features_config.variables,
                model_limited=model_limited,
            )
        )
    
    logger.info(
        f"Generated training data for {len(training_data)} species-activity combinations"
    )

    trainer = str(getattr(base_model_cfg, "type", "maxent") or "maxent").lower()
    deepmaxent_meta: Optional[Dict[str, Any]] = None
    run_device: Optional[str] = None

    if trainer == "deepmaxent":
        from sdm.models.deepmaxent.config import resolve_deepmaxent_params
        from sdm.models.deepmaxent.data import shared_feature_names
        from sdm.models.deepmaxent.training import train_deepmaxent_joint

        logger.info(
            "Training experimental joint DeepMaxent model for %d outputs "
            "(elapid MaxEnt remains the default when model.type is maxent)...",
            len(training_data),
        )
        dm_params = resolve_deepmaxent_params(
            base_model_cfg, seed_override=training_seed
        )
        joint_features = shared_feature_names(training_data)
        # Packages / predict use the shared roster the trunk was trained on.
        training_data = [
            TrainingData(
                latin_name=d.latin_name,
                activity_type=d.activity_type,
                occurrence=d.occurrence,
                maxent_config=d.maxent_config,
                model_features=joint_features,
                model_limited=d.model_limited,
            )
            for d in training_data
        ]
        dm_result = train_deepmaxent_joint(
            training_data, dm_params, fold_params=fold_params
        )
        models = dm_result.results
        run_device = dm_result.device
        deepmaxent_meta = {
            **dm_params.to_metadata(),
            "device_resolved": dm_result.device,
            "n_outputs": len(dm_result.output_names),
            "output_names": list(dm_result.output_names),
            "n_features": len(dm_result.feature_names),
            "n_sites": len(dm_result.joint_table.features),
            "n_seed_states": len(dm_result.seed_states),
        }
        logger.info(
            "DeepMaxent joint training complete on device=%s (%d outputs)",
            run_device,
            len(models),
        )
    else:
        # Train elapid MaxEnt models (default)
        logger.info(f"Training {len(training_data)} MaxEnt models...")
        models = train_models_parallel(
            training_data=training_data,
            max_threads_per_model=max_threads_per_model,
            n_jobs=n_jobs,
            seed=training_seed,
            fold_params=fold_params,
        )
    
    # Prepare and save results
    logger.info("Saving results...")
    results_df = prepare_results_dataframe(models, training_data)
    run_uri = build_run_output_uri(output_root)
    logger.info("Writing model run to %s", run_uri)

    with model_run_write_context(run_uri) as (models_output_dir, written_uri):
        model_paths = save_models(
            models,
            training_data,
            models_output_dir,
            trainer=trainer,
            deepmaxent_metadata=deepmaxent_meta,
        )

        published_paths = {
            identifier: join_run_artifact(
                written_uri,
                Path(path).relative_to(models_output_dir),
            )
            for identifier, path in model_paths.items()
        }
        results_df["model_path"] = [
            published_paths.get(identifier, "")
            for identifier in results_df["identifier"]
        ]
        results_df["model_package_dir"] = [
            artifact_parent_uri(p) if p else ""
            for p in results_df["model_path"]
        ]

        save_results(results_df, models_output_dir)
        save_training_data(training_data, models_output_dir)
        save_run_thresholds(models, training_data, models_output_dir)
        save_run_metadata(
            models_output_dir,
            seed=training_seed,
            model_config_path=repo_relative_config_path(model_config_path),
            fold_params=fold_params,
            trainer=trainer,
            device=run_device,
            deepmaxent=deepmaxent_meta,
        )

    logger.info("Model run outputs: %s", written_uri)
    run_id = run_segment_from_uri(str(written_uri))
    logger.info("To predict with this run: sdm predict --run %s", run_id)

    if mlflow_enabled:
        logger.info("Logging to MLflow...")
        session_metadata = TrainingSessionMetadata(
            occurrence_data=str(bats_uri),
            ev_data=str(ev_uri),
            model_config_path=str(model_config_path),
            variables_config_path=(
                str(variables_config_path) if variables_config_path else None
            ),
            model_run_uri=str(written_uri),
        )
        log_models_to_mlflow(
            models,
            training_data,
            results_df,
            session_metadata=session_metadata,
            maxent_config_to_dict=_maxent_config_to_dict,
        )
    else:
        logger.info("Skipping MLflow logging (disabled or unavailable).")
    
    logger.info("✓ Training pipeline complete")

    # print a summary of the results (model id, mean cv score, std cv score) in a table format
    logger.info("Summary of results:")
    logger.info(results_df[["identifier", "mean_cv_score", "std_cv_score"]].to_string(index=False))
    return results_df
