"""
MaxEnt Species Distribution Model Inference for Sheffield Bats.

This module loads trained MaxEnt models and applies them to generate
predictions across the study area using the new modular structure.
"""

import logging
from pathlib import Path
from typing import Optional, List, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import pickle

import typer
import pandas as pd
import rioxarray as rxr
from tqdm import tqdm

from sdm.utils.logging_utils import setup_logging
from sdm.data.loaders.vector import load_environmental_variables
from sdm.models.maxent.maxent_model import apply_model_to_rasters
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

def predict_species(
    model_path: Path,
    ev_raster: Path,
    output_path: Path,
    latin_name: str,
    activity_type: str
) -> dict:
    """Apply a model to the environmental variables raster."""
    metadata = {
        "latin_name": latin_name,
        "activity_type": activity_type,
        "model_file": model_path.name,
        "prediction_path": str(output_path),
        "success": False
    }
    
    try:
        # Load model
        logger.info(f"Loading model for {latin_name} ({activity_type})...")
        model = load_model(model_path)
        
        # Get feature names from the FeatureSubsetter step in the pipeline
        feature_names = None
        if hasattr(model, 'steps'):
            feature_subsetter = next((step[1] for step in model.steps if isinstance(step[1], FeatureSubsetter)), None)
            if feature_subsetter:
                feature_names = feature_subsetter.feature_names
                logger.debug(f"Using feature subset: {feature_names}")
        
        # Apply the model to the raster
        logger.info(f"Generating prediction raster for {latin_name} ({activity_type})...")
        apply_model_to_rasters(
            model=model,
            raster_paths=[str(ev_raster)],
            output_path=str(output_path),
            feature_names=feature_names,
            window_size=4096
        )
        metadata["success"] = True
        logger.info(f"Successfully generated prediction for {latin_name} ({activity_type})")

    except Exception as e:
        logger.error(f"Failed to generate prediction for {latin_name} ({activity_type}): {e}")
        metadata["error"] = str(e)
        metadata["success"] = False

    return metadata

def make_predictions(
    filtered_index: pd.DataFrame,
    models_dir: Path,
    ev_raster: Path,
    output_dir: Path,
    n_workers: int = -1
) -> pd.DataFrame:
    """Apply trained models to make predictions."""
    
    if n_workers == -1:
        n_workers = cpu_count() - 1
    logger.info(f"Using {n_workers} workers for parallel processing")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    futures = []
    tasks = []
    
    # Prepare tasks
    logger.info("Preparing prediction tasks...")
    for _, row in tqdm(filtered_index.iterrows(), desc="Preparing tasks"):
        model_path = Path(row.model_path)
        latin_name = row.latin_name
        activity_type = row.activity_type
        
        output_path = output_dir / f"{latin_name}_{activity_type}.tif"
        tasks.append((
            model_path,
            ev_raster,
            output_path,
            latin_name,
            activity_type
        ))
    
    # Execute predictions
    logger.info("Starting parallel prediction processing...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(predict_species, *task) for task in tasks]
        
        for future in tqdm(futures, desc="Processing predictions"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
    
    # Compile results
    results_df = pd.DataFrame(results)
    success_count = results_df["success"].sum()
    logger.info(f"Completed {len(results)} predictions: {success_count} successful, {len(results) - success_count} failed")
    
    # Save results summary
    results_path = output_dir / "prediction_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Prediction results saved to {results_path}")
    
    return results_df

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
    n_workers: int = typer.Option(
        -1,
        help="Number of workers for parallel processing. Default is -1 (all available cores)."
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
        output_dir,
        n_workers
    )
    
    logger.info("=== SDM Model Inference Pipeline Complete ===")

if __name__ == "__main__":
    app() 