import logging
from pathlib import Path
from typing import Optional, List, Any
import pickle

import pandas as pd
import geopandas as gpd
import rioxarray as rxr

from sdm.utils.logging_utils import setup_logging
from sdm.viz.plots import generate_pdp_individual_plots, sanitize_filename

logger = logging.getLogger(__name__)

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
        
        # Rename variables by their long name if available
        rename_map = {var: evs_dataset[var].attrs.get("long_name", str(var)) for var in evs_dataset.data_vars}
        evs_dataset = evs_dataset.rename(rename_map)
        
        ev_df = evs_dataset.to_dataframe()
        if 'spatial_ref' in ev_df.columns:
            ev_df = ev_df.drop(columns=["spatial_ref"])
        ev_df.dropna(inplace=True)
        
        if sample_size is not None and len(ev_df) > sample_size:
            logger.info(f"Sampling {sample_size} points from EV data (out of {len(ev_df)})...")
            ev_df_sampled = ev_df.sample(n=sample_size, random_state=42)
            return ev_df_sampled
        elif len(ev_df) == 0:
            logger.warning("EV DataFrame is empty after loading and NaN drop. Cannot sample.")
            return pd.DataFrame()
        else:
            logger.info(f"Using all {len(ev_df)} available EV data points (sample_size={sample_size}).")
            return ev_df
    except Exception as e:
        logger.error(f"Failed to load or process EV raster {ev_raster_path}: {e}", exc_info=True)
        raise

def generate_model_visualisations(
    run_summary_path: Path,
    ev_raster_path: Path,
    visualisations_output_dir: Path,
    species_filter: Optional[List[str]] = None,
    activity_filter: Optional[List[str]] = None,
    ev_sample_size: int = 1000,
    pdp_grid_resolution: int = 30,
    plot_dpi: int = 150,
    num_workers: int = -1,
    overwrite_existing_plots: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Core function to generate visualizations for trained SDM models.
    Can be called from other scripts or notebooks.

    Args:
        run_summary_path: Path to the SDM run summary CSV file
        ev_raster_path: Path to the environmental variables raster stack
        visualisations_output_dir: Directory to save visualization outputs
        species_filter: Optional list of species to visualize
        activity_filter: Optional list of activity types to visualize
        ev_sample_size: Number of samples from EV data to use for PDP calculations
        pdp_grid_resolution: Grid resolution for partial dependence calculations
        plot_dpi: DPI for saved plot images
        num_workers: Number of workers for parallel processing (-1 for all available cores - 1)
        overwrite_existing_plots: Whether to overwrite existing plot files
        verbose: Enable verbose logging

    Returns:
        DataFrame containing PDP generation results
    """
    setup_logging(verbose)
    visualisations_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting model visualisation generation. Output directory: {visualisations_output_dir}")

    # 1. Load model run summary
    logger.info(f"Loading model run summary from: {run_summary_path}")
    try:
        model_runs_df = load_model_run_summary(run_summary_path)
    except FileNotFoundError:
        logger.error(f"Model run summary CSV not found at {run_summary_path}.")
        raise
    except Exception as e:
        logger.error(f"Error loading model run summary: {e}.", exc_info=True)
        raise

    # 2. Filter models if species/activity filters are provided
    if species_filter:
        model_runs_df = model_runs_df[model_runs_df["species"].isin(species_filter)]
    if activity_filter:
        model_runs_df = model_runs_df[model_runs_df["activity"].isin(activity_filter)]

    # Filter for successful runs with a model path
    model_runs_df = model_runs_df[
        (model_runs_df["status"] == "Success") & 
        (model_runs_df["model_path"].notna())
    ].copy()

    if model_runs_df.empty:
        logger.warning("No successful models found to visualize after filtering.")
        return pd.DataFrame()

    # 3. Load model objects into the DataFrame
    logger.info("Loading trained model objects...")
    model_runs_df["model_object"] = model_runs_df["model_path"].apply(load_pickled_model)
    model_runs_df.dropna(subset=["model_object"], inplace=True)
    
    if model_runs_df.empty:
        logger.warning("No models could be loaded successfully.")
        return pd.DataFrame()

    # 4. Load and sample Environmental Variable data
    try:
        ev_sample_df = load_ev_dataframe(ev_raster_path, sample_size=ev_sample_size)
    except Exception as e:
        logger.error(f"Failed to load or sample EV data from {ev_raster_path}.")
        raise
        
    if ev_sample_df.empty:
        logger.error(f"EV sample data is empty. Cannot generate PDPs.")
        return pd.DataFrame()

    # 5. Generate PDP plots
    logger.info("Generating Partial Dependence Plots...")
    pdp_results_df = generate_pdp_individual_plots(
        models_df=model_runs_df,
        ev_sample=ev_sample_df,
        output_dir=visualisations_output_dir,
        grid_resolution=pdp_grid_resolution,
        dpi=plot_dpi,
        n_workers=num_workers,
        overwrite_plots=overwrite_existing_plots
    )

    # Save PDP generation summary
    if not pdp_results_df.empty:
        pdp_summary_path = visualisations_output_dir / "pdp_generation_summary.csv"
        pdp_results_df.to_csv(pdp_summary_path, index=False)
        logger.info(f"PDP generation summary saved to: {pdp_summary_path}")

    logger.info("Model visualisation generation finished.")
    return pdp_results_df 