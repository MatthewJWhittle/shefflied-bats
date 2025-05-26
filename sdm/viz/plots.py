import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor # For generate_pdp_individual_plots
from multiprocessing import cpu_count # For generate_pdp_individual_plots
import traceback # For _generate_pdp_plot_for_feature

import numpy as np # Required by sklearn.inspection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from tqdm import tqdm # For generate_pdp_individual_plots
import matplotlib.cm as cm # For write_tif_to_pngs
import xarray as xr # For write_tif_to_pngs

# Placeholder for sanitize_filename, will define it or import from utils later
# from ..utils.text_utils import sanitize_filename 

logger = logging.getLogger(__name__)

# Moved from modelling/vis_models.py
# Helper function, consider moving to a more general utils if used elsewhere
def sanitize_filename(name: str) -> str:
    """Convert a string to a valid filename component."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("\"", "")

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
        
        # Ensure the model object is the actual estimator, not a path or string
        model_estimator = model_info["model"] # Assuming this is the fitted model object
        if not hasattr(model_estimator, "predict_proba") and not hasattr(model_estimator, "decision_function") and not hasattr(model_estimator, "predict"):
             raise ValueError("Model object does not have a recognized prediction method for PDP.")

        display = PartialDependenceDisplay.from_estimator(
            model_estimator, 
            ev_sample, 
            [feature], 
            ax=ax,
            grid_resolution=grid_resolution,
            # n_jobs=1, # Prefer not to set n_jobs here if outer loop parallelizes
            line_kw={"linewidth": 2.5, "color": "darkblue"},
            centered=True # Elapid Maxent usually provides probabilities, centering is fine.
        )
        
        ax.set_title(f"Partial Dependence: {feature}", fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Partial Dependence", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig) # Close figure to free memory
        
        result["success"] = True
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"Error generating PDP plot for {model_info['latin_name']}, {feature}: {str(e)}", exc_info=True)
    
    return result

def generate_pdp_individual_plots(
    models_df: pd.DataFrame, # DataFrame containing model objects and info
    ev_sample: pd.DataFrame, 
    output_dir: Union[str, Path],
    # feature_names: List[str], # This should be derived from model_info or ev_sample for safety
    grid_resolution: int = 30, 
    dpi: int = 150,
    n_workers: int = -1,
    overwrite_plots: bool = True
) -> pd.DataFrame:
    """Generate individual partial dependence plots for all models and features."""
    pdp_base_dir = Path(output_dir) / "pdp_individual"
    pdp_base_dir.mkdir(parents=True, exist_ok=True)
    
    if n_workers <= 0:
        n_workers = max(1, cpu_count() - 1 if cpu_count() else 1)
    
    logger.info(f"Generating individual partial dependence plots using up to {n_workers} workers...")
    
    tasks = []
    for _, row in models_df.iterrows():
        model_object = row['model_object'] # Assuming model object is in this column
        latin_name = row["species"] # Assuming column names from run_sdm_model.py results
        activity_type = row["activity"] if pd.notna(row.get("activity")) else "all"
        features_used_in_model = row["feature_names_used"] # From run_sdm_model.py results
        
        if not features_used_in_model or not isinstance(features_used_in_model, list):
            logger.warning(f"No valid feature list for {latin_name} - {activity_type}. Skipping PDP.")
            continue
        if model_object is None:
            logger.warning(f"Model object is None for {latin_name} - {activity_type}. Skipping PDP.")
            continue

        # Ensure ev_sample contains all features used by the model
        # This is crucial as PartialDependenceDisplay will use these columns from ev_sample
        missing_ev_cols = [f for f in features_used_in_model if f not in ev_sample.columns]
        if missing_ev_cols:
            logger.error(f"Sample EV data is missing columns required by model {latin_name} - {activity_type}: {missing_ev_cols}. Skipping PDPs for this model.")
            continue

        species_activity_subdir_name = f"{sanitize_filename(latin_name)}_{sanitize_filename(activity_type)}"
        model_pdp_dir = pdp_base_dir / species_activity_subdir_name
        model_pdp_dir.mkdir(parents=True, exist_ok=True)
        
        model_info_dict = {
            "model": model_object,
            "latin_name": latin_name,
            "activity_type": activity_type
        }

        for feature in features_used_in_model:
            plot_output_path = model_pdp_dir / f"pdp_{sanitize_filename(feature)}.png"
            tasks.append({
                "model_info": model_info_dict,
                "feature": feature,
                "ev_sample": ev_sample[features_used_in_model], # Pass only relevant EV columns
                "output_path": plot_output_path,
                "grid_resolution": grid_resolution,
                "dpi": dpi,
                "overwrite": overwrite_plots
            })

    if not tasks:
        logger.warning("No PDP tasks were generated. Check model list and feature configurations.")
        return pd.DataFrame()

    plot_results = []
    # Using ProcessPoolExecutor for CPU-bound tasks like PDP generation
    # Might need to ensure model objects and large data are efficiently passed or loaded in worker
    # For simplicity here, passing them. For very large models/data, consider alternatives.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future for each task
        futures = [executor.submit(_generate_pdp_plot_for_feature, **task) for task in tasks]
        
        for future in tqdm(futures, total=len(tasks), desc="Generating PDPs"):
            try:
                plot_results.append(future.result())
            except Exception as e:
                # This should ideally be caught within _generate_pdp_plot_for_feature
                # but as a fallback:
                logger.error(f"A PDP generation task failed in the pool: {e}", exc_info=True)
                # Append a failure result if possible, based on task info
                # This part is tricky as the original task details are not directly available here
                # Best to ensure _generate_pdp_plot_for_feature is robust.

    return pd.DataFrame(plot_results) 

# Moved from pipelines_old/GenerateAppData/main.py
def normalize(array: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Normalize an array to 0-1 range."""
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    
    # Handle cases where vmin and vmax are the same to avoid division by zero
    if vmin == vmax:
        # If all values are the same, return an array of 0.5 (or 0, or 1, depending on desired behavior)
        # Or, if they are all NaN, nanmin/nanmax might lead here, so check for that.
        if np.isnan(vmin):
             return np.full_like(array, np.nan, dtype=np.float32)
        return np.full_like(array, 0.5, dtype=np.float32) 

    normalized = (array - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1) # Ensure values are strictly within 0-1

# Moved from pipelines_old/GenerateAppData/main.py
def write_tif_to_pngs(
    dataset: xr.Dataset, 
    out_dir: Path, 
    colormap_name: str = "viridis", # Renamed from colormap to avoid conflict with cm module
    vmin: Optional[float] = None, 
    vmax: Optional[float] = None, 
    overwrite: bool = False
) -> Tuple[Dict[str, Path], Tuple[float, float], int]:
    """Write each band of a TIF to a PNG file, using a consistent colormap and normalization.

    Args:
        dataset: Input xarray Dataset, where each data variable is a band.
        out_dir: Directory to save PNG files.
        colormap_name: Name of the Matplotlib colormap to use.
        vmin: Optional minimum value for normalization. If None, calculated from data.
        vmax: Optional maximum value for normalization. If None, calculated from data.
        overwrite: Whether to overwrite existing PNG files.

    Returns:
        A tuple containing: 
            - Dictionary mapping band names to output PNG paths.
            - Tuple of (actual_vmin, actual_vmax) used for normalization.
            - Number of bands processed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths = {}
    count = 0

    # Determine global vmin and vmax across all bands if not provided
    if vmin is None:
        vmin = min(float(arr.min()) for arr in dataset.data_vars.values() if arr.size > 0)
    if vmax is None:
        vmax = max(float(arr.max()) for arr in dataset.data_vars.values() if arr.size > 0)

    colormap_obj = cm.get_cmap(colormap_name)

    for band_name, data_array in dataset.data_vars.items():
        if not isinstance(band_name, str): # Ensure band_name is a string for sanitize_filename
            band_name = str(band_name)
            
        out_path = out_dir / f"{sanitize_filename(band_name)}.png"
        png_paths[band_name] = out_path
        count += 1

        if out_path.exists() and not overwrite:
            logger.info(f"Skipping {out_path}, already exists.")
            continue
        
        # Ensure data_array is 2D numpy array
        img_data = data_array.squeeze().to_numpy()
        
        if img_data.size == 0:
            logger.warning(f"Band {band_name} is empty or all NaN, skipping PNG generation.")
            continue

        # Normalize array using the determined vmin and vmax
        normalized_img = normalize(img_data, vmin=vmin, vmax=vmax)
        
        # Apply colormap
        # Matplotlib colormaps expect values in [0, 1] for non-RGBA output
        # They return RGBA arrays by default
        colored_img = colormap_obj(normalized_img)
        
        plt.imsave(out_path, colored_img, dpi=150) # Save the RGBA image
        logger.info(f"Saved {out_path}")

    return png_paths, (vmin, vmax), count 