"""
MaxEnt Species Distribution Model Inference for Sheffield Bats.

This module loads trained MaxEnt models and applies them to generate
predictions across the study area using the new modular structure.
"""

import logging
from pathlib import Path
from typing import Optional, List, Any, Dict
import pickle

import typer
import pandas as pd
import rioxarray as rxr

from sdm.utils.logging_utils import setup_logging
from sdm.data.loaders.vector import load_environmental_variables
from sdm.models.maxent.maxent_model import apply_models_to_raster
from sdm.models.core.feature_subsetter import FeatureSubsetter

app = typer.Typer()
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
) -> pd.DataFrame:
    """Apply trained models to make predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    logger.info("Loading models...")
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
            logger.info(f"Loaded model for {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model for {model_id}: {e}")
    
    if not models:
        raise ValueError("No models were successfully loaded")
    
    # Generate predictions
    logger.info("=== Generating Predictions ===")
    output_path = output_dir / "all_predictions.tif"
    
    try:
        apply_models_to_raster(
            models=models,
            raster_path=ev_raster,
            output_path=output_path,
            feature_names=feature_names,
            window_size=128,
        )
        logger.info(f"Successfully generated predictions for {len(models)} models")
        
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
    logger.info(f"Prediction results saved to {results_path}")
    
    return filtered_index

@app.command()
def main(
    ev_path: Path = typer.Option(
        "data/evs/evs-to-model.tif",
        help="Path to environmental variables raster."
    ),
    models_dir: Path = typer.Option(
        "data/sdm_models",
        help="Directory containing trained models."
    ),
    output_dir: Path = typer.Option(
        "data/sdm_predictions",
        help="Directory for output prediction files.",
        writable=True
    ),
    species: Optional[List[str]] = typer.Option(
        None,
        help="Optional: Specific species to generate predictions for (Latin names)."
    ),
    activity_types: Optional[List[str]] = typer.Option(
        None,
        help="Optional: Specific activity types to generate predictions for."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging."
    )
):
    """Run the model inference pipeline."""
    setup_logging(level=logging.INFO, verbose=verbose)
    
    logger.info("=== Starting SDM Model Inference Pipeline ===")
    
    # Load model index
    logger.info("Loading model index...")
    model_index = load_model_index(models_dir)
    logger.info(f"Found {len(model_index)} models in index")
    
    # Filter models
    filtered_index = filter_models(model_index, species, activity_types)
    logger.info(f"Selected {len(filtered_index)} models for prediction")
    
    if len(filtered_index) == 0:
        logger.warning("No models match the specified criteria")
        return
    
    # Load environmental variables
    logger.info("Loading environmental variables...")
    _, ev_raster = load_environmental_variables(ev_path)
    
    # Generate predictions
    logger.info("=== Starting Prediction Generation ===")
    results_df = make_predictions(
        filtered_index,
        models_dir,
        ev_raster,
        output_dir
    )
    
    logger.info("=== SDM Model Inference Pipeline Complete ===")

if __name__ == "__main__":
    app() 