"""
Simplified CLI for SDM project using config defaults.
"""

import logging
from pathlib import Path
from typing import Optional, List

import typer

from sdm.utils.io import load_project_config
from sdm.utils.logging_utils import setup_logging
from sdm.types import ProjectConfig

# Load project-level config once for CLI defaults
PROJECT_CONFIG: ProjectConfig = load_project_config()

app = typer.Typer(
    name="sdm",
    help="Species Distribution Modelling CLI",
    no_args_is_help=True
)

@app.command()
def setup(
    counties_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """Set up the project by creating the study boundary."""
    from sdm.data.spatial import create_boundary
    
    setup_logging(verbose=verbose)
    
    boundary_path = Path(PROJECT_CONFIG.paths.boundary)
    counties_path = counties_file or Path("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson")
    
    boundary_gdf = create_boundary(
        counties_file=counties_path,
        county_names=None,  # Yorkshire default
        target_crs=PROJECT_CONFIG.crs,
        simplify_tolerance=100.0
    )
    
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_gdf.to_file(boundary_path, driver="GeoJSON")
    logging.info(f"Study boundary saved to: {boundary_path}")

@app.command()
def data(
    verbose: bool = False
) -> None:
    """Generate all environmental data layers."""
    from sdm.commands.data_preparation.environmental.generate_terrain_data import generate_terrain_data
    from sdm.commands.data_preparation.environmental.generate_climate_data import generate_climate_data
    from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import generate_ceh_lc_data
    from sdm.commands.data_preparation.environmental.generate_vom_data import generate_vom_data
    from sdm.commands.data_preparation.spatial.generate_coastal_distance import generate_coastal_distance
    from sdm.commands.data_preparation.processing.process_os_data import process_os_data
    from sdm.commands.data_preparation.environmental.generate_terrain_stats import generate_terrain_stats
    from sdm.commands.data_preparation.processing.merge_ev_layers import merge_ev_layers
    
    setup_logging(verbose=verbose)
    
    boundary_path = Path(PROJECT_CONFIG.paths.boundary)
    evs_dir = Path("data/evs")
    
    logging.info("Generating all environmental data layers...")
    
    # 1. Terrain data (DTM/DSM)
    logging.info("1/7: Downloading terrain data...")
    generate_terrain_data(
        output_dir=evs_dir / "terrain",
        boundary_path=boundary_path,
        buffer_distance_m=7000,
        verbose=verbose
    )
    
    # 2. Terrain statistics
    logging.info("2/7: Calculating terrain statistics...")
    generate_terrain_stats(
        input_dem_path=evs_dir / "terrain" / "dtm_dsm_100m.tif",
        output_path=evs_dir / "terrain_stats.tif",
        verbose=verbose
    )
    
    # 3. Climate data
    logging.info("3/7: Downloading climate data...")
    generate_climate_data(
        output_dir=evs_dir / "climate",
        boundary_path=boundary_path,
        worldclim_cache_dir=Path("data/raw/worldclim"),
        verbose=verbose
    )
    
    # 4. Land cover data
    logging.info("4/7: Processing land cover data...")
    generate_ceh_lc_data(
        output_dir=evs_dir / "landcover",
        boundary_path=boundary_path,
        ceh_data_path=Path("data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"),
        verbose=verbose
    )
    
    # 5. VOM data
    logging.info("5/7: Downloading VOM data...")
    generate_vom_data(
        output_dir=evs_dir / "vom",
        boundary_path=boundary_path,
        buffer_distance_m=7000,
        verbose=verbose
    )
    
    # 6. Coastal distance
    logging.info("6/7: Calculating coastal distance...")
    generate_coastal_distance(
        boundary_path=boundary_path,
        output_dir=evs_dir,
        bgs_geocoast_shp_path=Path("data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp"),
        verbose=verbose
    )
    
    # 7. OS data
    logging.info("7/7: Processing OS data...")
    process_os_data(
        output_dir=evs_dir,
        boundary_path=boundary_path,
        buffer_distance=7000,
        verbose=verbose
    )
    
    # Merge all layers
    logging.info("Merging all environmental layers...")
    merge_ev_layers(
        dataset_inputs=[
            "terrain_stats=evs/terrain_stats.tif",
            "climate=evs/climate",
            "landcover=evs/landcover",
            "vom=evs/vom",
            "coastal=evs/coastal_distance.tif",
            "os_cover=evs/os-feature-cover.tif",
            "os_distance=evs/os-distance-to-feature.tif"
        ],
        boundary_path=boundary_path,
        output_path=Path(PROJECT_CONFIG.paths.ev_tiff),
        verbose=verbose
    )
    
    logging.info("Environmental data generation complete!")

@app.command()
def background(
    occurrence_data_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.occurence_data),
        help="Path to bat occurrence data used to generate background points.",
    ),
    boundary_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.boundary),
        help="Path to study boundary for background generation.",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.processed_data) / "background_generation",
        help="Directory to write background generation outputs.",
    ),
    background_points_output_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.background_points),
        help="Path to write background points GeoJSON.",
    ),
    verbose: bool = False,
) -> None:
    """Generate background points for modeling."""
    from sdm.commands.data_preparation.spatial.generate_background_points import (
        generate_background_points_wrapper,
    )

    setup_logging(verbose=verbose)

    # Generate background points and write directly to expected location
    _bg_points_path, _density_raster_path = generate_background_points_wrapper(
        occurrence_data_path=occurrence_data_path,
        boundary_path=boundary_path,
        output_dir=output_dir,
        background_points_output_path=background_points_output_path,
        verbose=verbose,
    )

    logging.info("Background points generated!")

@app.command()
def train(
    bats_file: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.occurence_data),
        help="Path to bat occurrence data.",
    ),
    ev_file: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.ev_tiff),
        help="Path to environmental variables raster.",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.models),
        help="Directory to write trained models and results.",
    ),
    model_config_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.model_config_path),
        help="Path to model/training configuration YAML.",
    ),
    variables_config_path: Optional[Path] = typer.Option(
        None,
        help="Path to variables configuration YAML (defaults to config).",
    ),
    config_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.config_dir),
        help="Directory containing per-species-activity model + feature configs (optional). If provided, configs are loaded from {tuning_dir}/{latin_name}_{activity_type}/ with fallback to base configs.",
    ),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    min_presence: Optional[int] = typer.Option(
        None,
        help="Minimum number of presence records required (defaults to config).",
    ),
    n_jobs: Optional[int] = typer.Option(
        None,
        help="Number of parallel jobs (defaults to auto).",
    ),
    max_threads_per_model: int = typer.Option(
        2,
        help="Maximum threads per model.",
    ),
    # Species-specific processing parameters
    d_min: float = typer.Option(
        500,
        help="Minimum distance from presence for background (meters).",
    ),
    d_max: Optional[float] = typer.Option(
        None,
        help="Maximum distance from presence for background (meters, None = no limit).",
    ),
    sample_weight_n_neighbors: int = typer.Option(
        10,
        help="Number of neighbors for sample weighting.",
    ),
    verbose: bool = False,
) -> None:
    """Train SDM models using the new modular approach.
    
    Background point generation parameters are loaded from model_config.yml.
    """
    from sdm.commands.modelling.train_sdm_models import train_sdm_models
    import numpy as np
    
    setup_logging(verbose=verbose)
    
    # Convert d_max: None means np.inf
    d_max_value = d_max if d_max is not None else np.inf
    
    train_sdm_models(
        model_config_path=model_config_path,
        variables_config_path=variables_config_path,
        config_dir=config_dir,
        bats_file=bats_file,
        ev_file=ev_file,
        output_dir=output_dir,
        min_presence=min_presence,
        n_jobs=n_jobs,
        max_threads_per_model=max_threads_per_model,
        species=species,
        activity_types=activity_types,
        verbose=verbose,
        d_min=d_min,
        d_max=d_max_value,
        sample_weight_n_neighbors=sample_weight_n_neighbors,
    )
    
    logging.info("Model training complete!")

@app.command()
def predict(
    ev_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.ev_tiff),
        help="Path to environmental variables raster.",
    ),
    models_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.models),
        help="Directory containing trained models.",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.predictions),
        help="Directory to write prediction rasters.",
    ),
    boundary_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.boundary),
        help="Path to boundary file for clipping output raster.",
    ),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    split_files: bool = typer.Option(
        True,
        "--split-files/--no-split-files",
        help="If True, write each model prediction as a separate file. If False, write all predictions in one combined file.",
    ),
    verbose: bool = False,
) -> None:
    """Generate model predictions."""
    from sdm.commands.modelling.predict_sdm_models import predict_sdm_models
    
    setup_logging(verbose=verbose)
    
    predict_sdm_models(
        ev_path=ev_path,
        models_dir=models_dir,
        output_dir=output_dir,
        boundary_path=boundary_path,
        species=species,
        activity_types=activity_types,
        split_files=split_files,
        verbose=verbose,
    )
    
    logging.info("Predictions generated!")

@app.command()
def visualize(
    run_summary_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.predictions) / "results.csv",
        help="Path to model run summary CSV.",
    ),
    ev_raster_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.ev_tiff),
        help="Path to environmental variables raster.",
    ),
    visualisations_output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.predictions) / "visualization",
        help="Directory to write visualizations.",
    ),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """Generate model visualizations."""
    from sdm.commands.visualization.visualise_model_outputs import generate_model_visualisations
    
    setup_logging(verbose=verbose)
    
    generate_model_visualisations(
        run_summary_path=run_summary_path,
        ev_raster_path=ev_raster_path,
        visualisations_output_dir=visualisations_output_dir,
        species_filter=species,
        activity_filter=activity_types,
        verbose=verbose,
    )
    
    logging.info("Visualizations generated!")

@app.command()
def explain(
    ev_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.ev_tiff),
        help="Path to environmental variables raster.",
    ),
    models_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.models),
        help="Directory containing trained models.",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.predictions) / "visualization" / "shap",
        help="Directory to write SHAP plots.",
    ),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    n_explain: int = typer.Option(
        200,
        help="Number of points to explain (default: 200).",
    ),
    n_background: int = typer.Option(
        200,
        help="Number of background samples for SHAP (default: 100).",
    ),
    top_features: Optional[int] = typer.Option(
        None,
        help="Number of top features for dependence plots (None = all features, default: all).",
    ),
    n_jobs: int = typer.Option(
        -1,
        help="Number of parallel workers (default: all available cores).",
    ),
    verbose: bool = False,
) -> None:
    """Calculate SHAP values for SDM models and generate interpretability plots.
    
    This command loads trained SDM models, samples points from the environmental
    variables dataset, and computes SHAP values in parallel for each model.
    Generates feature importance bar plots and dependence plots for top features.
    """
    from sdm.commands.modelling.explain_sdm_models import explain_sdm_models
    
    setup_logging(verbose=verbose)
    
    explain_sdm_models(
        ev_path=ev_path,
        models_dir=models_dir,
        output_dir=output_dir,
        species=species,
        activity_types=activity_types,
        n_explain=n_explain,
        n_background=n_background,
        top_features=top_features,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
    logging.info("SHAP explanation complete!")

@app.command()
def pipeline(
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    generate_background: bool = typer.Option(
        False,
        help="Generate background points file (optional, training generates them on-the-fly).",
    ),
    verbose: bool = False
) -> None:
    """Run the complete SDM pipeline.
    
    Note: Background points are now generated on-the-fly during training,
    so the background step is optional unless you need the background points file
    for other purposes.
    """
    setup_logging(verbose=verbose)
    
    logging.info("Starting complete SDM pipeline...")
    
    # Run all steps
    if generate_background:
        background(verbose=verbose)
    data(verbose=verbose)
    train(species=species, activity_types=activity_types, verbose=verbose)
    predict(species=species, activity_types=activity_types, verbose=verbose)
    visualize(species=species, activity_types=activity_types, verbose=verbose)
    
    logging.info("SDM pipeline complete!")

if __name__ == "__main__":
    app()

@app.command()
def set_boundary(
    filepath: Path,
    verbose: bool = False
) -> None:
    """Set the boundary file and validate it."""
    from sdm.utils.io import update_config, validate_boundary_file
    import logging
    
    setup_logging(verbose=verbose)
    
    # Validate the file
    validate_boundary_file(filepath)
    
    # Update config
    config_path = Path("config.yml")
    update_config(config_path, {"paths.boundary": str(filepath.resolve())})
    
    logging.info(f"Boundary file set to: {filepath.resolve()}")

@app.command()
def set_occurrence(
    filepath: Path,
    verbose: bool = False
) -> None:
    """Set the occurrence data file and validate it."""
    from sdm.utils.io import update_config, validate_occurrence_file
    import logging
    
    setup_logging(verbose=verbose)
    
    # Validate the file
    validate_occurrence_file(filepath)
    
    # Update config
    config_path = Path("config.yml")
    update_config(config_path, {"paths.occurence_data": str(filepath.resolve())})
    
    logging.info(f"Occurrence data file set to: {filepath.resolve()}")

@app.command()
def config(
    verbose: bool = False
) -> None:
    """Show current configuration."""
    import yaml
    from sdm.utils.io import load_config
    
    setup_logging(verbose=verbose)
    
    config = load_config()
    
    print("Current Configuration:")
    print("=" * 50)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

@app.command()
def tune(
    bats_file: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.occurence_data),
        help="Path to bat occurrence data.",
    ),
    ev_file: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.ev_tiff),
        help="Path to environmental variables raster.",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.tuning_dir),
        help="Directory to write tuning results and best configs.",
    ),
    model_config_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.model_config_path),
        help="Path to base model/training configuration YAML.",
    ),
    variables_config_path: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.variables_config_path),
        help="Path to base variables configuration YAML.",
    ),
    n_trials: int = typer.Option(
        50,
        help="Number of Optuna trials to run.",
    ),
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    study_name: Optional[str] = typer.Option(
        None,
        help="Name for Optuna study (optional).",
    ),
    storage: Optional[str] = typer.Option(
        None,
        help="Optuna storage URL for distributed tuning (optional).",
    ),
    verbose: bool = False,
    n_cv_folds: int = typer.Option(
        3,
        help="Number of CV folds for evaluation (reduced from 3 for faster tuning).",
    ),
    n_jobs: int = typer.Option(
        -1,
        help="Number of parallel trials (None = sequential, >1 = parallel).",
    ),
) -> None:
    """Tune hyperparameters for SDM models using Optuna.
    
    Uses the new modular approach where expensive operations (EV conversion,
    annotation, background generation) are done once before tuning. Only model
    parameters and feature selection are tuned (background points are fixed).
    """
    from sdm.commands.modelling.tune_hyperparameters import tune_hyperparameters
    
    setup_logging(verbose=verbose)
    
    tune_hyperparameters(
        project_config_path=Path("config.yml"),
        model_config_path=model_config_path,
        variables_config_path=variables_config_path,
        bats_file=bats_file,
        ev_file=ev_file,
        output_dir=output_dir,
        n_trials=n_trials,
        species=species,
        activity_types=activity_types,
        study_name=study_name,
        storage=storage,
        verbose=verbose,
        n_cv_folds=n_cv_folds,
        n_jobs=n_jobs,
    )
    
    logging.info(f"Tuning complete! Best configs written to {output_dir}")
