#!/usr/bin/env python
"""
Visualize MaxEnt species distribution models with various plots and metrics.

This script provides a modular framework for generating visualizations from trained models,
starting with partial dependence plots but designed for easy extension.
"""

import os
import pickle
import argparse
import traceback
from pathlib import Path
from typing import List, Union, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from sklearn.inspection import PartialDependenceDisplay
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for MaxEnt models"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/sdm_models",
        help="Directory containing trained models and model index",
    )
    parser.add_argument(
        "--evs-path",
        type=str,
        default="data/evs/evs-to-model.tif",
        help="Path to environmental variables raster",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sdm_predictions/visualization",
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="*",
        help="Optional: Specific species to generate visualizations for (Latin names).",
    )
    parser.add_argument(
        "--activity-types",
        type=str,
        nargs="*",
        help="Optional: Specific activity types to generate visualizations for.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to use for calculations",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=30,
        help="Resolution for partial dependence grid",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output plots",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers for parallel processing. Default is -1 (all available cores - 1)."
    )
    return parser.parse_args()


def load_model_index(models_dir):
    """Load the index of available models."""
    models_dir = Path(models_dir)
    index_path = models_dir / "model_index.csv"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Model index not found at {index_path}")
    
    return pd.read_csv(index_path)


def filter_models(model_index, species=None, activity_types=None):
    """Filter models based on species and activity types."""
    filtered_index = model_index.copy()
    
    if species:
        filtered_index = filtered_index[filtered_index.latin_name.isin(species)]
    
    if activity_types:
        filtered_index = filtered_index[filtered_index.activity_type.isin(activity_types)]
    
    return filtered_index


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_ev_df(evs_path: str) -> pd.DataFrame:
    """Load environmental variables as a DataFrame."""
    print(f"Loading environmental variables from {evs_path}")
    evs = rxr.open_rasterio(evs_path, masked=True, band_as_variable=True).squeeze()
    
    # Rename the variables by their long name
    for var in evs.data_vars:
        evs = evs.rename({var: evs[var].attrs["long_name"]})
    
    # Convert to DataFrame
    ev_df = evs.to_dataframe()
    if 'spatial_ref' in ev_df.columns:
        ev_df.drop("spatial_ref", axis=1, inplace=True)
    ev_df.dropna(inplace=True)

    return ev_df


def sample_ev_data(ev_df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """Sample the environmental data for analysis."""
    print(f"Sampling {sample_size} points from the environmental data")
    return ev_df.sample(min(sample_size, len(ev_df)), random_state=42)


def sanitize_filename(name: str) -> str:
    """Convert a species or feature name to a valid filename."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _generate_pdp_plot_for_feature(
    model_info: Dict[str, Any], 
    feature: str,
    ev_sample: pd.DataFrame,
    output_path: Path,
    grid_resolution: int = 30,
    dpi: int = 150,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Generate a single PDP plot for a specific feature and model.
    
    This function contains robust error handling to skip plots that fail
    without crashing the entire pipeline.
    """
    result = {
        "feature": feature,
        "latin_name": model_info["latin_name"],
        "activity_type": model_info["activity_type"],
        "output_path": str(output_path),
        "success": False
    }
    
    try:
        # Skip if output already exists
        if output_path.exists() and (not overwrite):
            result["success"] = True
            result["skipped"] = "Output file already exists"
            return result
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        display = PartialDependenceDisplay.from_estimator(
            model_info["model"], 
            ev_sample, 
            [feature], 
            ax=ax,
            grid_resolution=grid_resolution,
            n_jobs=1,  # Use 1 job here since we're already parallelizing at a higher level
            line_kw={"linewidth": 2.5, "color": "darkblue"},
            centered=True
        )
        
        ax.set_title(f"Partial Dependence: {feature}", fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Partial Dependence", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        
        result["success"] = True
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"Error generating plot for {model_info['latin_name']}, {feature}: {str(e)}")
    
    return result


def generate_pdp_individual_plots(
    models_df: pd.DataFrame, 
    ev_sample: pd.DataFrame, 
    output_dir: Union[str, Path],
    feature_names: List[str],
    grid_resolution: int = 30, 
    dpi: int = 150,
    n_workers: int = -1
) -> pd.DataFrame:
    """Generate individual partial dependence plots for all models and features."""
    # Create output directory
    pdp_dir = Path(output_dir) / "pdp_individual"
    pdp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up workers
    if n_workers <= 0:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Generating individual partial dependence plots using {n_workers} workers...")
    
    # Prepare tasks
    tasks = []
    for _, row in models_df.iterrows():
        model = row['model']
        latin_name = row.latin_name
        activity_type = row.activity_type if not pd.isna(row.activity_type) else "All"
        
        # Create directory structure
        species_dir = pdp_dir / sanitize_filename(latin_name) / sanitize_filename(activity_type)
        species_dir.mkdir(parents=True, exist_ok=True)
        
        for feature in feature_names:
            output_path = species_dir / f"{sanitize_filename(feature)}.png"
            
            tasks.append({
                "model_info": {
                    "model": model,
                    "latin_name": latin_name,
                    "activity_type": activity_type
                },
                "feature": feature,
                "output_path": output_path
            })
    
    results = []
    
    # Use parallel processing
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for task in tasks:
                futures.append(
                    executor.submit(
                        _generate_pdp_plot_for_feature,
                        model_info=task["model_info"],
                        feature=task["feature"],
                        ev_sample=ev_sample,
                        output_path=task["output_path"],
                        grid_resolution=grid_resolution,
                        dpi=dpi
                    )
                )
            
            # Track progress
            for future in tqdm(futures, total=len(futures), desc="Generating individual PDP plots"):
                results.append(future.result())
    else:
        # Sequential processing with progress bar
        for task in tqdm(tasks, total=len(tasks), desc="Generating individual PDP plots"):
            result = _generate_pdp_plot_for_feature(
                model_info=task["model_info"],
                feature=task["feature"],
                ev_sample=ev_sample,
                output_path=task["output_path"],
                grid_resolution=grid_resolution,
                dpi=dpi
            )
            results.append(result)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Report on success/failure
    successful = len(results_df[results_df.success])
    failed = len(results_df[~results_df.success])
    print(f"PDP individual plots: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("Sample of errors:")
        for _, row in results_df[~results_df.success].head(5).iterrows():
            print(f"  {row.latin_name} - {row.feature}: {row.error}")
    
    # Save results
    results_df.to_csv(pdp_dir / "plot_results.csv", index=False)
    
    return results_df


def load_models_from_index(models_dir, filtered_index):
    """Load models from disk based on the filtered index."""
    print(f"Loading {len(filtered_index)} models from disk...")
    models_dir = Path(models_dir)
    
    # Create a new DataFrame to hold both metadata and model objects
    models_df = filtered_index.copy()
    models_df['model'] = None
    models_df['load_success'] = False
    models_df['error'] = None
    
    # Load each model
    for idx, row in tqdm(models_df.iterrows(), total=len(models_df)):
        model_path = models_dir / row.model_file
        try:
            model = load_model(model_path)
            if model is not None:
                models_df.at[idx, 'model'] = model
                models_df.at[idx, 'load_success'] = True
            else:
                models_df.at[idx, 'error'] = "Model loading returned None"
        except Exception as e:
            models_df.at[idx, 'error'] = str(e)
    
    # Filter out failed models
    successful_loads = models_df[models_df.load_success]
    failed_loads = models_df[~models_df.load_success]
    
    print(f"Successfully loaded {len(successful_loads)} models, {len(failed_loads)} failed")
    
    if len(failed_loads) > 0:
        print("Sample of model loading errors:")
        for _, row in failed_loads.head(5).iterrows():
            print(f"  {row.latin_name} - {row.activity_type}: {row.error}")
    
    return successful_loads


def main(
    models_dir: str,
    evs_path: str, 
    output_dir: Union[str, Path],
    species=None,
    activity_types=None,
    sample_size: int = 1000,
    grid_resolution: int = 30,
    dpi: int = 150,
    n_workers: int = -1
):
    """
    Main function to generate visualizations for MaxEnt models.
    """
    # Determine number of workers
    if n_workers <= 0:
        n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} worker processes")
    
    # Create main output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and filter model index
    print(f"Loading model index from {models_dir}...")
    model_index = load_model_index(models_dir)
    filtered_index = filter_models(model_index, species, activity_types)
    print(f"Found {len(filtered_index)} models matching criteria.")
    
    if len(filtered_index) == 0:
        print("No models match the specified criteria.")
        return
    
    # Load the actual model objects
    models_df = load_models_from_index(models_dir, filtered_index)
    
    if len(models_df) == 0:
        print("No models could be loaded. Check for errors above.")
        return
    
    # Load environmental data
    ev_df = load_ev_df(evs_path)
    feature_names = ev_df.columns.tolist()
    ev_sample = sample_ev_data(ev_df, sample_size=sample_size)
    
    # Generate individual PDP plots
    generate_pdp_individual_plots(
        models_df=models_df,
        ev_sample=ev_sample,
        output_dir=output_dir,
        feature_names=feature_names,
        grid_resolution=grid_resolution,
        dpi=dpi,
        n_workers=n_workers
    )
    
    print("Visualization generation complete!")


if __name__ == "__main__":
    args = parse_args()
    main(
        models_dir=args.models_dir,
        evs_path=args.evs_path,
        output_dir=args.output_dir,
        species=args.species,
        activity_types=args.activity_types,
        sample_size=args.sample_size,
        grid_resolution=args.grid_resolution,
        dpi=args.dpi,
        n_workers=args.n_workers
    )