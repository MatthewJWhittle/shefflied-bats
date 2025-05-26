import logging
from pathlib import Path
from typing import Optional, List, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import typer
import pandas as pd
from tqdm import tqdm
# import rioxarray as rxr # Not directly loading EVs here, path is passed to model prediction func

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_model_run_summary, load_pickled_model # Shared helpers
from sdm.models.maxent_model import predict_rasters_with_elapid_model # Core prediction function
# from species_sdm.viz.plots import sanitize_filename # If needed for output paths

app = typer.Typer()
logger = logging.getLogger(__name__)

def _sanitize_filename_for_prediction(name: str) -> str:
    """Convert a string to a valid filename component for prediction outputs."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("\"", "")


def _predict_for_single_model_task(
    model_path_str: str, 
    ev_raster_path_str: str, # Path to the EV raster stack
    output_raster_path_str: str,
    species_name: str,
    activity_type: str
) -> dict:
    """Worker function to apply a single model to the EV raster."""
    metadata = {
        "species": species_name,
        "activity": activity_type,
        "model_path": model_path_str,
        "prediction_output_path": output_raster_path_str,
        "success": False
    }
    try:
        model_object = load_pickled_model(model_path_str)
        if model_object is None: # load_pickled_model now raises error, so this check might be redundant
            raise ValueError("Failed to load model object.")

        predict_rasters_with_elapid_model(
            model=model_object,
            raster_paths=[ev_raster_path_str], # Assuming ev_raster is a single stack file
            output_path=output_raster_path_str,
            # Defaults for predict_rasters_with_elapid_model are generally fine
            # template_raster_idx=0,
            # windowed_prediction=True,
            # predict_proba=True,
        )
        metadata["success"] = True
        logger.info(f"Prediction successful for {species_name} - {activity_type} -> {output_raster_path_str}")

    except Exception as e:
        logger.error(f"Error during prediction for {species_name} - {activity_type} using model {model_path_str}: {e}", exc_info=True)
        metadata["error"] = str(e)
    
    return metadata


@app.command()
def main(
    run_summary_path: Path = typer.Option(
        "outputs/sdm_runs/sdm_run_summary.csv",
        help="Path to the SDM run summary CSV (contains model paths).",
        exists=True, readable=True, resolve_path=True
    ),
    ev_raster_path: Path = typer.Option(
        "data/evs/evs-to-model.tif", 
        help="Path to environmental variables raster stack (GeoTIFF).",
        exists=True, readable=True, resolve_path=True
    ),
    predictions_output_dir: Path = typer.Option(
        "outputs/sdm_predictions/rasters", 
        help="Directory to save output prediction rasters.",
        writable=True, resolve_path=True, file_okay=False, dir_okay=True
    ),
    species_filter: Optional[List[str]] = typer.Option(None, "--species", help="Optional: Specific species (Latin names) to predict for."),
    activity_filter: Optional[List[str]] = typer.Option(None, "--activity", help="Optional: Specific activity types to predict for."),
    num_workers: int = typer.Option(-1, help="Number of workers for parallel processing (-1 for all available cores - 1)."),
    overwrite_existing_predictions: bool = typer.Option(False, "--overwrite", help="Overwrite existing prediction raster files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """
    Applies trained SDM models to environmental rasters to generate prediction maps.
    """
    setup_logging(verbose)
    predictions_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting model inference. Output directory for predictions: {predictions_output_dir}")

    # 1. Load model run summary
    logger.info(f"Loading model run summary from: {run_summary_path}")
    try:
        model_runs_df = load_model_run_summary(run_summary_path)
    except FileNotFoundError:
        logger.error(f"Model run summary CSV not found at {run_summary_path}. Exiting.")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"Error loading model run summary: {e}. Exiting.", exc_info=True)
        raise typer.Exit(code=1)

    # 2. Filter models based on CLI options and success status
    if species_filter:
        model_runs_df = model_runs_df[model_runs_df["species"].isin(species_filter)]
    if activity_filter:
        model_runs_df = model_runs_df[model_runs_df["activity"].isin(activity_filter)]

    # Only consider models that were successfully trained and have a path
    # The `run_sdm_model.py` script saves model path even if prediction raster failed.
    # We need models that have `status == "Success"` from the training run.
    models_to_predict_df = model_runs_df[
        model_runs_df["model_path"].notna() &
        (model_runs_df["status"] == "Success") # Key filter for successfully trained models
    ].copy()

    if models_to_predict_df.empty:
        logger.warning("No suitable models found in the summary to generate predictions for. Exiting.")
        raise typer.Exit()
    
    logger.info(f"Found {len(models_to_predict_df)} models to process for prediction.")

    # 3. Prepare tasks for parallel execution
    tasks = []
    for _, row in models_to_predict_df.iterrows():
        model_p = str(row["model_path"])
        species = str(row["species"])
        activity = str(row.get("activity", "all")) # Handle potential NaN for activity
        
        # Define output path for the prediction raster
        sane_species = _sanitize_filename_for_prediction(species)
        sane_activity = _sanitize_filename_for_prediction(activity)
        output_fname = f"prediction_{sane_species}_{sane_activity}.tif"
        current_output_raster_path = predictions_output_dir / output_fname

        if not overwrite_existing_predictions and current_output_raster_path.exists():
            logger.info(f"Skipping prediction for {species} - {activity} as output already exists: {current_output_raster_path}")
            continue

        tasks.append({
            "model_path_str": model_p,
            "ev_raster_path_str": str(ev_raster_path),
            "output_raster_path_str": str(current_output_raster_path),
            "species_name": species,
            "activity_type": activity
        })

    if not tasks:
        logger.info("No new predictions to generate (all existing or no models selected). Exiting.")
        raise typer.Exit()

    # 4. Execute predictions in parallel
    if num_workers <= 0:
        num_workers = max(1, cpu_count() -1 if cpu_count() else 1)
    logger.info(f"Generating predictions using up to {num_workers} workers...")

    all_prediction_results = []
    # Using ProcessPoolExecutor for CPU-bound tasks (model prediction can be)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_predict_for_single_model_task, **task_args) for task_args in tasks]
        
        for future in tqdm(futures, total=len(tasks), desc="Generating Predictions"):
            try:
                result = future.result()
                all_prediction_results.append(result)
            except Exception as e:
                # This catch is a fallback; _predict_for_single_model_task should handle its errors
                logger.error(f"A prediction task failed in the pool: {e}", exc_info=True)
                # We don't have task details here to add a specific failed result, relies on worker logging
    
    # 5. Save summary of prediction generation
    if all_prediction_results:
        prediction_summary_df = pd.DataFrame(all_prediction_results)
        summary_output_path = predictions_output_dir.parent / "predictions_generation_summary.csv" # Save one level up
        try:
            prediction_summary_df.to_csv(summary_output_path, index=False)
            logger.info(f"Prediction generation summary saved to: {summary_output_path}")
        except Exception as e_save:
            logger.error(f"Failed to save prediction summary: {e_save}", exc_info=True)
    else:
        logger.warning("No prediction results were generated.")

    logger.info("Model inference script finished.")

if __name__ == "__main__":
    app() 