import logging
from pathlib import Path
from typing import Optional, List, Any # Added Any
import pickle # For loading model objects

import typer
import pandas as pd
import geopandas as gpd # May not be directly needed, but good for context
import rioxarray as rxr # For load_ev_df

from sdm.utils.logging_utils import setup_logging
# from species_sdm.utils.io import load_config # If using main config for paths
from sdm.viz.plots import generate_pdp_individual_plots, sanitize_filename # sanitize_filename might move

app = typer.Typer()
logger = logging.getLogger(__name__)

# --- Helper functions adapted or to be adapted from vis_models.py --- 

def load_model_run_summary(summary_csv_path: Path) -> pd.DataFrame:
    """Loads the SDM run summary CSV file."""
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Model run summary not found at {summary_csv_path}")
    return pd.read_csv(summary_csv_path)

def load_pickled_model(model_path_str: str) -> Any:
    """Loads a pickled model object from a given path string."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None

def load_ev_dataframe(ev_raster_path: Path, sample_size: Optional[int] = 1000) -> pd.DataFrame:
    """Loads environmental variables from a raster, converts to DataFrame, and samples."""
    logger.info(f"Loading environmental variables from: {ev_raster_path}")
    try:
        evs_dataset = rxr.open_rasterio(ev_raster_path, masked=True, band_as_variable=True).squeeze()
        
        # Rename variables by their long name if available (common practice from this project)
        # This assumes bands in ev_raster_path correspond to features used by models
        rename_map = {var: evs_dataset[var].attrs.get("long_name", str(var)) for var in evs_dataset.data_vars}
        evs_dataset = evs_dataset.rename(rename_map)
        
        ev_df = evs_dataset.to_dataframe()
        if 'spatial_ref' in ev_df.columns: # Common metadata column to drop
            ev_df = ev_df.drop(columns=["spatial_ref"])
        ev_df.dropna(inplace=True) # Drop rows with any NaNs after conversion
        
        if sample_size is not None and len(ev_df) > sample_size:
            logger.info(f"Sampling {sample_size} points from EV data (out of {len(ev_df)})...")
            ev_df_sampled = ev_df.sample(n=sample_size, random_state=42)
            return ev_df_sampled
        elif len(ev_df) == 0:
            logger.warning("EV DataFrame is empty after loading and NaN drop. Cannot sample.")
            return pd.DataFrame() # Return empty DF
        else:
            logger.info(f"Using all {len(ev_df)} available EV data points (sample_size={sample_size}).")
            return ev_df
    except Exception as e:
        logger.error(f"Failed to load or process EV raster {ev_raster_path}: {e}", exc_info=True)
        raise # Re-raise after logging

@app.command()
def main(
    run_summary_path: Path = typer.Option(
        "outputs/sdm_runs/sdm_run_summary.csv", # Default to output of run_sdm_model.py
        help="Path to the SDM run summary CSV file (contains paths to models and feature lists).",
        exists=True, readable=True, resolve_path=True
    ),
    ev_raster_path: Path = typer.Option(
        "data/evs/evs-to-model.tif", # Default EV stack path from other scripts
        help="Path to the environmental variables raster stack (GeoTIFF).",
        exists=True, readable=True, resolve_path=True
    ),
    visualisations_output_dir: Path = typer.Option(
        "outputs/sdm_visualisations", 
        help="Directory to save visualization outputs.",
        writable=True, resolve_path=True, file_okay=False, dir_okay=True
    ),
    species_filter: Optional[List[str]] = typer.Option(None, "--species", help="Optional: Specific species (Latin names) to visualize."),
    activity_filter: Optional[List[str]] = typer.Option(None, "--activity", help="Optional: Specific activity types to visualize."),
    ev_sample_size: int = typer.Option(1000, help="Number of samples from EV data to use for PDP calculations."),
    pdp_grid_resolution: int = typer.Option(30, help="Grid resolution for partial dependence calculations."),
    plot_dpi: int = typer.Option(150, help="DPI for saved plot images."),
    num_workers: int = typer.Option(-1, help="Number of workers for parallel processing (-1 for all available cores - 1)."),
    overwrite_existing_plots: bool = typer.Option(False, "--overwrite-plots", help="Overwrite existing plot files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """
    Generates visualizations (e.g., Partial Dependence Plots) for trained SDM models.
    Reads model information from the run summary CSV produced by run_sdm_model.py.
    """
    setup_logging(verbose)
    visualisations_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting model visualisation generation. Output directory: {visualisations_output_dir}")

    # 1. Load model run summary
    logger.info(f"Loading model run summary from: {run_summary_path}")
    try:
        model_runs_df = load_model_run_summary(run_summary_path)
    except FileNotFoundError:
        logger.error(f"Model run summary CSV not found at {run_summary_path}. Exiting.")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error loading model run summary: {e}. Exiting.", exc_info=True)
        raise typer.Exit(code=1)

    # 2. Filter models if species/activity filters are provided
    if species_filter:
        model_runs_df = model_runs_df[model_runs_df["species"].isin(species_filter)]
    if activity_filter:
        model_runs_df = model_runs_df[model_runs_df["activity"].isin(activity_filter)]

    # Filter for successful runs with a model path
    model_runs_df = model_runs_df[
        (model_runs_df["status"] == "Success") & 
        (model_runs_df["model_path"].notna())
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if model_runs_df.empty:
        logger.warning("No successful models found to visualize after filtering. Exiting.")
        raise typer.Exit()

    # 3. Load model objects into the DataFrame
    logger.info("Loading trained model objects...")
    model_runs_df["model_object"] = model_runs_df["model_path"].apply(load_pickled_model)
    # Drop rows where model loading failed
    model_runs_df.dropna(subset=["model_object"], inplace=True)
    
    if model_runs_df.empty:
        logger.warning("No models could be loaded successfully. Exiting.")
        raise typer.Exit()

    # 4. Load and sample Environmental Variable data
    try:
        ev_sample_df = load_ev_dataframe(ev_raster_path, sample_size=ev_sample_size)
    except Exception as e:
        logger.error(f"Failed to load or sample EV data from {ev_raster_path}. Exiting.")
        raise typer.Exit(code=1)
        
    if ev_sample_df.empty:
        logger.error(f"EV sample data is empty. Cannot generate PDPs. Exiting.")
        raise typer.Exit(code=1)

    # 5. Generate PDP plots
    logger.info("Generating Partial Dependence Plots...")
    pdp_results_df = generate_pdp_individual_plots(
        models_df=model_runs_df, # This df now contains 'model_object' and 'feature_names_used'
        ev_sample=ev_sample_df,
        output_dir=visualisations_output_dir,
        grid_resolution=pdp_grid_resolution,
        dpi=plot_dpi,
        n_workers=num_workers,
        overwrite_plots=overwrite_existing_plots
    )

    # Save PDP generation summary (optional)
    if not pdp_results_df.empty:
        pdp_summary_path = visualisations_output_dir / "pdp_generation_summary.csv"
        pdp_results_df.to_csv(pdp_summary_path, index=False)
        logger.info(f"PDP generation summary saved to: {pdp_summary_path}")
    else:
        logger.warning("PDP generation did not produce any results (summary DataFrame is empty).")

    logger.info("Model visualisation script finished.")

if __name__ == "__main__":
    app() 