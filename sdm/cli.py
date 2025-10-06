"""
Simplified CLI for SDM project using config defaults.
"""

import typer
from typing import Optional, List
from pathlib import Path
import logging

from sdm.utils.io import load_config
from sdm.utils.logging_utils import setup_logging

# Load default config
CONFIG = load_config()

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
    
    boundary_path = Path(CONFIG["paths"]["boundary"])
    counties_path = counties_file or Path("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson")
    
    boundary_gdf = create_boundary(
        counties_file=counties_path,
        county_names=None,  # Yorkshire default
        target_crs=CONFIG["crs"],
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
    
    boundary_path = Path(CONFIG["paths"]["boundary"])
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
        output_path=Path(CONFIG["paths"]["ev_tiff"]),
        verbose=verbose
    )
    
    logging.info("Environmental data generation complete!")

@app.command()
def background(
    verbose: bool = False
) -> None:
    """Generate background points for modeling."""
    from sdm.commands.data_preparation.spatial.generate_background_points import generate_background_points_wrapper
    
    setup_logging(verbose=verbose)
    
    generate_background_points_wrapper(
        occurrence_data_path=Path(CONFIG["paths"]["occurence_data"]),
        boundary_path=Path(CONFIG["paths"]["boundary"]),
        output_dir=Path(CONFIG["paths"]["processed_data"]) / "background_generation",
        verbose=verbose
    )
    
    logging.info("Background points generated!")

@app.command()
def train(
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """Train SDM models."""
    from sdm.commands.modelling.train_sdm_models import train_sdm_models
    
    setup_logging(verbose=verbose)
    
    train_sdm_models(
        bats_file=Path(CONFIG["paths"]["occurence_data"]),
        background_file=Path(CONFIG["paths"]["background_points"]),
        ev_file=Path(CONFIG["paths"]["ev_tiff"]),
        output_dir=Path(CONFIG["paths"]["models"]),
        species=species,
        activity_types=activity_types,
        verbose=verbose
    )
    
    logging.info("Model training complete!")

@app.command()
def predict(
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """Generate model predictions."""
    from sdm.commands.modelling.predict_sdm_models import predict_sdm_models
    
    setup_logging(verbose=verbose)
    
    predict_sdm_models(
        ev_path=Path(CONFIG["paths"]["ev_tiff"]),
        models_dir=Path(CONFIG["paths"]["models"]),
        output_dir=Path(CONFIG["paths"]["predictions"]),
        species=species,
        activity_types=activity_types,
        verbose=verbose
    )
    
    logging.info("Predictions generated!")

@app.command()
def visualize(
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """Generate model visualizations."""
    from sdm.commands.visualization.visualise_model_outputs import generate_model_visualisations
    
    setup_logging(verbose=verbose)
    
    generate_model_visualisations(
        run_summary_path=Path(CONFIG["paths"]["predictions"]) / "results.csv",
        ev_raster_path=Path(CONFIG["paths"]["ev_tiff"]),
        visualisations_output_dir=Path(CONFIG["paths"]["predictions"]) / "visualization",
        species_filter=species,
        activity_filter=activity_types,
        verbose=verbose
    )
    
    logging.info("Visualizations generated!")

@app.command()
def pipeline(
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """Run the complete SDM pipeline."""
    setup_logging(verbose=verbose)
    
    logging.info("Starting complete SDM pipeline...")
    
    # Run all steps
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
    config_path = Path("config/default.yaml")
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
    config_path = Path("config/default.yaml")
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
