"""
MaxEnt Species Distribution Model Inference for Sheffield Bats.

This module loads trained MaxEnt models and applies them to generate
predictions across the study area using the new modular structure.
"""

import logging
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import pickle

import typer
import pandas as pd
import rioxarray as rxr
from tqdm import tqdm

from sdm.utils.logging_utils import setup_logging
from sdm.data.loaders.vector import load_environmental_variables
from sdm.models.maxent.maxent_model import predict_rasters_with_elapid_model

app = typer.Typer()
logger = logging.getLogger(__name__)

def load_model_index(models_dir: Path) -> pd.DataFrame:
    """Load the index of available models."""
    index_path = models_dir / "model_index.csv"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Model index not found at {index_path}")
    
    return pd.read_csv(index_path)

def load_model(model_path: Path):
    """Load a trained model from disk."""
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def filter_models(model_index: pd.DataFrame, species: Optional[List[str]] = None, activity_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter models based on species and activity types."""
    filtered_index = model_index.copy()
    
    if species:
        filtered_index = filtered_index[filtered_index.latin_name.isin(species)]
    
    if activity_types:
        filtered_index = filtered_index[filtered_index.activity_type.isin(activity_types)]
    
    return filtered_index

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
        model = load_model(model_path)
        
        # Apply the model to the raster using the new modular function
        predict_rasters_with_elapid_model(
            model=model,
            raster_paths=[str(ev_raster)],
            output_path=str(output_path)
        )
        metadata["success"] = True

    except Exception as e:
        logger.error(f"Error applying model {model_path.name}: {e}")
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
    logger.info(f"Using {n_workers} workers for parallel processing.")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    futures = []
    tasks = []
    
    for _, row in tqdm(filtered_index.iterrows(), total=len(filtered_index)):
        model_path = models_dir / row.model_file
        latin_name = row.latin_name
        activity_type = row.activity_type
        
        # Create unique output path for this prediction
        output_path = output_dir / f"{latin_name}_{activity_type}.tif"
        tasks.append((
            model_path,
            ev_raster,
            output_path,
            latin_name,
            activity_type
        ))

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for task in tasks:
                model_path, ev_raster, output_path, latin_name, activity_type = task
                futures.append(
                    executor.submit(
                        predict_species,
                        model_path,
                        ev_raster,
                        output_path,
                        latin_name,
                        activity_type
                    ))

            for future in tqdm(futures, total=len(futures)):
                result = future.result()
                results.append(result)
    else:
        for task in tqdm(tasks, total=len(tasks)):
            model_path, ev_raster, output_path, latin_name, activity_type = task
            result = predict_species(
                model_path,
                ev_raster,
                output_path,
                latin_name,
                activity_type
            )
            results.append(result)

    for result in results:
        if result["success"]:
            logger.info(f"Prediction successful for {result['latin_name']} - {result['activity_type']}")
        else:
            logger.error(f"Prediction failed for {result['latin_name']} - {result['activity_type']}: {result['error']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "prediction_results.csv", index=False)
    
    return results_df

@app.command()
def main(
    ev_path: Path = typer.Option(
        "data/evs/evs-to-model.tif",
        help="Path to environmental variables raster."
    ),
    models_dir: Path = typer.Option(
        "outputs/sdm_runs/models",
        help="Directory containing trained models."
    ),
    output_dir: Path = typer.Option(
        "outputs/sdm_predictions",
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
    
    # Load model index
    logger.info(f"Loading models from {models_dir}...")
    model_index = load_model_index(models_dir)
    logger.info(f"Found {len(model_index)} models")
    
    # Filter models based on species/activity type if specified
    filtered_index = filter_models(model_index, species, activity_types)
    logger.info(f"Generating predictions for {len(filtered_index)} models")
    
    if len(filtered_index) == 0:
        logger.warning("No models match the specified criteria.")
        return
    
    # Load environmental variables
    logger.info("Loading environmental variables...")
    _, ev_raster = load_environmental_variables(ev_path)
    
    # Generate predictions
    logger.info("Generating predictions...")
    results_df = make_predictions(
        filtered_index,
        models_dir,
        ev_raster,
        output_dir,
        n_workers
    )
    
    logger.info(f"Prediction results saved to {output_dir / 'prediction_results.csv'}")

if __name__ == "__main__":
    app() 