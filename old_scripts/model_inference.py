"""
MaxEnt Species Distribution Model Inference for Sheffield Bats.

This module loads trained MaxEnt models and applies them to generate
predictions across the study area.
"""

import argparse
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pandas as pd
import rioxarray as rxr
from tqdm import tqdm


from modelling.maxent_utils import apply_model_to_rasters

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply trained MaxEnt models to generate predictions."
    )
    
    parser.add_argument(
        "--ev-path", 
        type=str,
        default="data/evs/evs-to-model.tif",
        help="Path to environmental variables raster."
    )
    
    parser.add_argument(
        "--models-dir", 
        type=str,
        default="data/sdm_models",
        help="Directory containing trained models."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="data/sdm_predictions",
        help="Directory for output prediction files."
    )
    
    parser.add_argument(
        "--species",
        type=str,
        nargs="*",
        help="Optional: Specific species to generate predictions for (Latin names)."
    )
    
    parser.add_argument(
        "--activity-types",
        type=str,
        nargs="*",
        help="Optional: Specific activity types to generate predictions for."
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers for parallel processing. Default is -1 (all available cores)."
    )
    
    return parser.parse_args()


def load_model_index(models_dir):
    """Load the index of available models."""
    models_dir = Path(models_dir)
    index_path = models_dir / "model_index.csv"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Model index not found at {index_path}")
    
    return pd.read_csv(index_path)


def load_environmental_variables(ev_path):
    """Load environmental variables for predictions."""
    ev_raster = Path(ev_path)
    if not ev_raster.exists():
        raise FileNotFoundError(f"Environmental variables file not found at {ev_path}")
    
    evs = rxr.open_rasterio(ev_raster, masked=True, band_as_variable=True).squeeze()
    # rename the variables by their long name
    for var in evs.data_vars:
        evs = evs.rename({var: evs[var].attrs["long_name"]})
    
    return evs, ev_raster


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def filter_models(model_index, species=None, activity_types=None):
    """Filter models based on species and activity types."""
    filtered_index = model_index.copy()
    
    if species:
        filtered_index = filtered_index[filtered_index.latin_name.isin(species)]
    
    if activity_types:
        filtered_index = filtered_index[filtered_index.activity_type.isin(activity_types)]
    
    return filtered_index


def predict_species(
        model_path, 
        ev_raster, 
        output_path,
        latin_name, 
        activity_type
        ) -> dict:
    
    """Apply a model to the environmental variables raster."""
    metadata = {
        "latin_name": latin_name,
        "activity_type": activity_type,
        "model_file": model_path.name,
        "prediction_path": output_path,
        "success": False
    }
    try:
        model = load_model(model_path)
        
        # Apply the model to the raster
        apply_model_to_rasters(
            model=model,
            raster_paths=[ev_raster],
            output_path=output_path
        )
        metadata["success"] = True

    except Exception as e:
        print(f"Error applying model {model_path.name}: {e}")
        metadata["error"] = str(e)
        metadata["success"] = False

    return metadata


def make_predictions(
        filtered_index, 
        models_dir, 
        ev_raster, 
        output_dir,
        n_workers= - 1
        ):
    """Apply trained models to make predictions."""
    
    if n_workers == -1:
        n_workers = cpu_count() - 1
    print(f"Using {n_workers} workers for parallel processing.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    
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
            print(f"Prediction successful for {result['latin_name']} - {result['activity_type']}")
        else:
            print(f"Prediction failed for {result['latin_name']} - {result['activity_type']}: {result['error']}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "prediction_results.csv", index=False)
    
    return results_df


def main(ev_path, models_dir, output_dir, species=None, activity_types=None, n_workers=-1):
    """Run the model inference pipeline."""
    
    # Load model index
    print(f"Loading models from {models_dir}...")
    model_index = load_model_index(models_dir)
    print(f"Found {len(model_index)} models")
    
    # Filter models based on species/activity type if specified
    filtered_index = filter_models(model_index, species, activity_types)
    print(f"Generating predictions for {len(filtered_index)} models")
    
    if len(filtered_index) == 0:
        print("No models match the specified criteria.")
        return
    
    # Load environmental variables
    print("Loading environmental variables...")
    _, ev_raster = load_environmental_variables(ev_path)
    
    # Generate predictions
    print("Generating predictions...")
    results = make_predictions(filtered_index, models_dir, ev_raster, output_dir, n_workers)
    
    # Summarize results
    successful = results[results.success].shape[0]
    failed = results[~results.success].shape[0]
    print(f"Predictions complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("Failed predictions:")
        for _, row in results[~results.success].iterrows():
            print(f"  {row.latin_name} - {row.activity_type}: {row.error}")
    
    return results


if __name__ == "__main__":
    args = parse_arguments()
    main(
        ev_path=args.ev_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        species=args.species,
        activity_types=args.activity_types,
        n_workers=args.n_workers
    )
