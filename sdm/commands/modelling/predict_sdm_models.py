"""
MaxEnt Species Distribution Model Inference for Sheffield Bats.

This module loads trained MaxEnt models and applies them to generate
predictions across the study area using the new modular structure.
"""

import logging
from pathlib import Path
from typing import Optional, List, Any, Dict
import pickle

import pandas as pd
import rioxarray as rxr

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary
from sdm.raster.io import load_environmental_variables
from sdm.models.maxent.maxent_model import apply_models_to_raster
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

def make_predictions(
    filtered_index: pd.DataFrame,
    models_dir: Path,
    ev_raster: Path,
    output_dir: Path,
    boundary_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Apply trained models to make predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    logger.debug("Loading models...")
    models: Dict[str, Any] = {}
    feature_names = None
    
    for _, row in filtered_index.iterrows():
        model_path = Path(row.model_path)
        latin_name = row.latin_name
        activity_type = row.activity_type
        model_id = f"{latin_name}_{activity_type}"
        
        try:
            # Load model
            model = load_model(model_path)
            
            # Get feature names from the first model's FeatureSubsetter
            if feature_names is None and hasattr(model, 'steps'):
                feature_subsetter = next((step[1] for step in model.steps if isinstance(step[1], FeatureSubsetter)), None)
                if feature_subsetter:
                    feature_names = feature_subsetter.feature_names
                    logger.debug(f"Using feature subset: {feature_names}")
            
            models[model_id] = model
            logger.debug(f"Loaded model for {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model for {model_id}: {e}")
    
    if not models:
        raise ValueError("No models were successfully loaded")
    
    # Generate predictions
    logger.info(f"Generating predictions for {len(models)} models...")
    output_path = output_dir / "all_predictions.tif"
    
    try:
        apply_models_to_raster(
            models=models,
            raster_path=ev_raster,
            output_path=output_path,
            window_size=128,
        )
        logger.debug(f"Successfully generated predictions for {len(models)} models")
        
        # Clip to boundary if provided
        if boundary_path and boundary_path.exists():
            logger.info(f"Clipping predictions to boundary: {boundary_path}")
            try:
                # Load boundary
                boundary_gdf = load_boundary(boundary_path, buffer_distance=0)
                
                # Load the prediction raster
                pred_raster = rxr.open_rasterio(output_path)
                
                # Store original transform and CRS to preserve them
                original_transform = pred_raster.rio.transform()
                original_crs = pred_raster.rio.crs
                original_nodata = pred_raster.rio.nodata
                
                # Ensure CRS matches
                if pred_raster.rio.crs != boundary_gdf.crs:
                    boundary_gdf = boundary_gdf.to_crs(pred_raster.rio.crs)
                
                # Get union of all boundary geometries
                # Use unary_union if available, otherwise try union_all() (geopandas extension)
                if hasattr(boundary_gdf, 'union_all'):
                    boundary_union = boundary_gdf.union_all()
                else:
                    boundary_union = boundary_gdf.geometry.unary_union
                
                # Clip the raster with crop=False to maintain original extent and transform
                pred_raster_clipped = pred_raster.rio.clip(
                    [boundary_union],
                    crs=boundary_gdf.crs,
                    all_touched=True,
                    crop=False  # Maintain original raster extent and transform
                )
                
                # Ensure the transform and CRS are preserved
                pred_raster_clipped = pred_raster_clipped.rio.write_transform(original_transform)
                pred_raster_clipped = pred_raster_clipped.rio.write_crs(original_crs)
                if original_nodata is not None:
                    pred_raster_clipped = pred_raster_clipped.rio.write_nodata(original_nodata)
                
                # Save the clipped raster (overwrite the original)
                pred_raster_clipped.rio.to_raster(output_path)
                logger.info(f"Clipped predictions saved to: {output_path}")
                
            except Exception as e:
                logger.warning(f"Failed to clip predictions to boundary: {e}. Output saved without clipping.")
        
        # Update results with success status
        filtered_index["success"] = True
        filtered_index["prediction_path"] = str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        filtered_index["success"] = False
        filtered_index["error"] = str(e)
    
    # Save results summary
    results_path = output_dir / "prediction_results.csv"
    filtered_index.to_csv(results_path, index=False)
    logger.debug(f"Prediction results saved to {results_path}")
    
    return filtered_index

def predict_sdm_models(
    ev_path: Path = Path("data/evs/evs-to-model.tif"),
    models_dir: Path = Path("data/sdm_models"),
    output_dir: Path = Path("data/sdm_predictions"),
    boundary_path: Optional[Path] = None,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Run the model inference pipeline.

    Args:
        ev_path: Path to environmental variables raster.
        models_dir: Directory containing trained models.
        output_dir: Directory for output prediction files.
        boundary_path: Optional path to boundary file for clipping output raster.
        species: Optional: Specific species to generate predictions for (Latin names).
        activity_types: Optional: Specific activity types to generate predictions for.
        verbose: Enable verbose logging.

    Returns:
        DataFrame containing prediction results.

    Raises:
        FileNotFoundError: If model index or input files are not found.
        ValueError: If no models match the specified criteria or if no models are successfully loaded.
    """
    setup_logging(level=logging.INFO, verbose=verbose)
    
    logger.info("Starting prediction pipeline...")
    
    # Load model index
    logger.debug("Loading model index...")
    model_index = load_model_index(models_dir)
    logger.info(f"Found {len(model_index)} models in index")
    
    # Filter models
    filtered_index = filter_models(model_index, species, activity_types)
    logger.info(f"Selected {len(filtered_index)} models for prediction")
    
    if len(filtered_index) == 0:
        logger.warning("No models match the specified criteria")
        raise ValueError("No models match the specified criteria")
    
    # Load environmental variables
    logger.debug("Loading environmental variables...")
    _, ev_raster = load_environmental_variables(ev_path)
    
    # Generate predictions
    results_df = make_predictions(
        filtered_index,
        models_dir,
        ev_raster,
        output_dir,
        boundary_path=boundary_path,
    )
    
    logger.info("✓ Prediction pipeline complete")
    return results_df 