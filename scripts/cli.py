import typer
from pathlib import Path
from typing_extensions import Annotated
from typing import Optional, List, Literal

app = typer.Typer(
    name="sheffield-bats",
    help="CLI tools for Sheffield Bats data processing and modelling",
    add_completion=False,
)

# Import core functions from their respective modules
from .modelling.extract_model_features import extract_features
from .visualization.visualise_model_outputs import generate_model_visualisations
from .data_preparation.spatial.generate_background_points import generate_background_points_wrapper
from .data_preparation.processing.merge_ev_layers import merge_ev_layers
from .data_preparation.processing.process_os_data import process_os_data
from .data_preparation.spatial.generate_coastal_distance import generate_coastal_distance
from .data_preparation.spatial.create_study_boundary import create_study_boundary_wrapper
from .modelling.predict_sdm_models import predict_sdm_models
from .modelling.train_sdm_models import train_sdm_models
from .data_preparation.environmental.generate_terrain_data import generate_terrain_data
from .data_preparation.environmental.generate_climate_data import generate_climate_data
from .data_preparation.environmental.generate_ceh_lc_data import generate_ceh_lc_data
from .data_preparation.environmental.generate_vom_data import generate_vom_data
from .data_preparation.environmental.generate_terrain_stats import generate_terrain_stats

@app.command()
def extract(
    input_points_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to combined occurrence and background points data (e.g., Parquet from generate_background_points.py).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    ev_raster_stack_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to the multi-band environmental variable raster stack (e.g., GeoTIFF from merge_ev_layers.py).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    output_path: Annotated[
        Path, 
        typer.Option(
            help="Path to save the final model-ready feature data (e.g., Parquet).",
            writable=True, resolve_path=True
        )
    ] = Path("data/processed/model_input_features.parquet"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False
) -> None:
    """
    Extracts environmental variable (EV) data for occurrence/background points,
    applies any necessary preprocessing (e.g., scaling), and saves the 
    model-ready feature dataset.
    """
    extract_features(
        input_points_path=input_points_path,
        ev_raster_stack_path=ev_raster_stack_path,
        output_path=output_path,
        verbose=verbose
    )

@app.command()
def visualize(
    run_summary_path: Annotated[
        Path, 
        typer.Option(
            "outputs/sdm_runs/sdm_run_summary.csv",
            help="Path to the SDM run summary CSV file (contains paths to models and feature lists).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    ev_raster_path: Annotated[
        Path, 
        typer.Option(
            "data/evs/evs-to-model.tif",
            help="Path to the environmental variables raster stack (GeoTIFF).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    visualisations_output_dir: Annotated[
        Path, 
        typer.Option(
            "outputs/sdm_visualisations",
            help="Directory to save visualization outputs.",
            writable=True, resolve_path=True, file_okay=False, dir_okay=True
        )
    ],
    species_filter: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--species", help="Optional: Specific species (Latin names) to visualize.")
    ] = None,
    activity_filter: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--activity", help="Optional: Specific activity types to visualize.")
    ] = None,
    ev_sample_size: Annotated[
        int, 
        typer.Option(1000, help="Number of samples from EV data to use for PDP calculations.")
    ] = 1000,
    pdp_grid_resolution: Annotated[
        int, 
        typer.Option(30, help="Grid resolution for partial dependence calculations.")
    ] = 30,
    plot_dpi: Annotated[
        int, 
        typer.Option(150, help="DPI for saved plot images.")
    ] = 150,
    num_workers: Annotated[
        int, 
        typer.Option(-1, help="Number of workers for parallel processing (-1 for all available cores - 1).")
    ] = -1,
    overwrite_existing_plots: Annotated[
        bool, 
        typer.Option(False, "--overwrite-plots", help="Overwrite existing plot files.")
    ] = False,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Generates visualizations (e.g., Partial Dependence Plots) for trained SDM models.
    Reads model information from the run summary CSV produced by run_sdm_model.py.
    """
    generate_model_visualisations(
        run_summary_path=run_summary_path,
        ev_raster_path=ev_raster_path,
        visualisations_output_dir=visualisations_output_dir,
        species_filter=species_filter,
        activity_filter=activity_filter,
        ev_sample_size=ev_sample_size,
        pdp_grid_resolution=pdp_grid_resolution,
        plot_dpi=plot_dpi,
        num_workers=num_workers,
        overwrite_existing_plots=overwrite_existing_plots,
        verbose=verbose
    )

@app.command()
def background(
    occurrence_data_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to occurrence data (GeoJSON, GPKG or Parquet).", 
            exists=True, 
            readable=True, 
            resolve_path=True
        )
    ],
    boundary_path: Annotated[
        Path, 
        typer.Option(
            help="Path to boundary data (e.g., GeoJSON).", 
            exists=True, 
            readable=True, 
            resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    output_dir: Annotated[
        Path, 
        typer.Option(
            help="Base directory to save outputs (density raster and background points).", 
            writable=True, 
            resolve_path=True, 
            file_okay=False, 
            dir_okay=True
        )
    ] = Path("data/processed/background_generation"),
    n_background_points: Annotated[
        int, 
        typer.Option(help="Number of background points to generate.")
    ] = 10000,
    background_method: Annotated[
        Literal["contrast", "percentile", "scale", "fixed", "binary"],
        typer.Option(case_sensitive=False, help="Method for setting minimum background probability.")
    ] = "contrast",
    background_value: Annotated[
        float, 
        typer.Option(help="Value for background_method (e.g., contrast ratio, percentile).")
    ] = 0.3,
    grid_resolution: Annotated[
        Optional[int], 
        typer.Option(help="Resolution of the model grid in CRS units (e.g., meters). Defaults to project setting.")
    ] = None,
    transform_method: Annotated[
        Literal["log", "sqrt", "presence", "cap", "rank"],
        typer.Option(case_sensitive=False, help="Method to transform occurrence counts for density estimation.")
    ] = "log",
    cap_percentile: Annotated[
        float, 
        typer.Option(help="Percentile for 'cap' transform_method (0-100).")
    ] = 90.0,
    sigma: Annotated[
        float, 
        typer.Option(help="Sigma value for Gaussian smoothing of occurrence density.")
    ] = 1.5,
    verbose: Annotated[
        bool, 
        typer.Option("--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Generates background points for species distribution modeling based on 
    density-smoothed occurrence data.
    Saves an intermediate occurrence density raster and the final background points (Parquet).
    """
    generate_background_points_wrapper(
        occurrence_data_path=occurrence_data_path,
        boundary_path=boundary_path,
        output_dir=output_dir,
        n_background_points=n_background_points,
        background_method=background_method,
        background_value=background_value,
        grid_resolution=grid_resolution,
        transform_method=transform_method,
        cap_percentile=cap_percentile,
        sigma=sigma,
        verbose=verbose
    )

@app.command()
def merge(
    dataset_inputs: Annotated[
        List[str], 
        typer.Option(
            ..., # Required
            help="List of datasets to merge, each in 'name=path/to/file.tif' format. "
                 "Example: --dataset-inputs 'Layer1=./ev1.tif' --dataset-inputs 'Layer2=./ev2.tif'"
        )
    ],
    output_path: Annotated[
        Path, 
        typer.Option(
            "data/evs/merged_environmental_variables.tif", 
            help="Path to save the merged multi-band GeoTIFF.",
            writable=True, resolve_path=True
        )
    ] = Path("data/evs/merged_environmental_variables.tif"),
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file for clipping the final merged layer.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    boundary_buffer_m: Annotated[
        float, 
        typer.Option(help="Buffer (meters) for the boundary before final clipping. Set to 0 for no buffer.")
    ] = 0,
    verbose: Annotated[
        bool, 
        typer.Option("--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Merges multiple raster datasets into a single multi-band GeoTIFF file.
    
    Each input dataset is reprojected to match the project's spatial configuration 
    (derived from the boundary and its associated spatial config, e.g. resolution, CRS).
    Variable names are tidied, and the final merged raster is clipped to the specified boundary.
    """
    merge_ev_layers(
        dataset_inputs=dataset_inputs,
        output_path=output_path,
        boundary_path=boundary_path,
        boundary_buffer_m=boundary_buffer_m,
        verbose=verbose
    )

@app.command()
def os(
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/evs", 
            help="Directory to save output EV files.",
            writable=True, resolve_path=True, file_okay=False, dir_okay=True
        )
    ] = Path("data/evs"),
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    buffer_distance: Annotated[
        float, 
        typer.Option(7000, help="Buffer distance for the boundary.")
    ] = 7000,
    load_from_shp: Annotated[
        bool, 
        typer.Option(False, help="Load OS data directly from SHP instead of expecting Parquet files (slower).")
    ] = False,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Process Ordnance Survey data to generate environmental variables.
    
    Creates two main outputs:
    1. Feature coverage density (percentage of area covered by each feature type)
    2. Distance to nearest feature (for each feature type)
    """
    process_os_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        buffer_distance=buffer_distance,
        load_from_shp=load_from_shp,
        verbose=verbose
    )

@app.command()
def coastal(
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file (e.g., GeoJSON) for the study area.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/evs", 
            help="Directory to save the output coastal_distance.tif file.",
            writable=True, resolve_path=True, file_okay=False, dir_okay=True
        )
    ] = Path("data/evs"),
    bgs_geocoast_shp_path: Annotated[
        Path, 
        typer.Option(
            "data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp",
            help="Path to the BGS GeoCoast Authority Area Inundation shapefile.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp"),
    buffer_dist_km_for_sea: Annotated[
        float, 
        typer.Option(10.0, help="Buffer distance in km to create the initial 'sea' zone from coastline.")
    ] = 10.0,
    simplify_tolerance_m: Annotated[
        float, 
        typer.Option(1000.0, help="Simplification tolerance in meters for coastal geometry processing.")
    ] = 1000.0,
    min_sea_area_km2: Annotated[
        float, 
        typer.Option(1.0, help="Minimum area in km^2 for a sea polygon to be retained after explosion.")
    ] = 1.0,
    distance_calc_resolution_factor: Annotated[
        int, 
        typer.Option(10, help="Factor to multiply base model resolution for distance calculation.")
    ] = 10,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Generates a coastal distance raster layer using BGS GeoCoast data.
    
    The process involves:
    1. Loading and buffering the study area boundary.
    2. Loading BGS GeoCoast inundation data.
    3. Processing the coastal data to define a 'sea' polygon.
    4. Calculating distance from a grid to this 'sea' polygon.
    5. Reprojecting, squeezing, and clipping the resulting distance raster.
    6. Saving the final coastal_distance.tif.
    """
    generate_coastal_distance(
        boundary_path=boundary_path,
        output_dir=output_dir,
        bgs_geocoast_shp_path=bgs_geocoast_shp_path,
        buffer_dist_km_for_sea=buffer_dist_km_for_sea,
        simplify_tolerance_m=simplify_tolerance_m,
        min_sea_area_km2=min_sea_area_km2,
        distance_calc_resolution_factor=distance_calc_resolution_factor,
        verbose=verbose
    )

@app.command()
def boundary(
    raw_counties_file: Annotated[
        Path, 
        typer.Option(
            "data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson", 
            help="Path to the raw UK counties GeoJSON file.",
            exists=True, readable=True, file_okay=True, resolve_path=True
        )
    ] = Path("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"),
    output_geojson: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to save the processed study area boundary GeoJSON.",
            writable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    target_crs: Annotated[
        str, 
        typer.Option("EPSG:27700", help="Target CRS for the output boundary.")
    ] = "EPSG:27700",
    simplify_tolerance: Annotated[
        Optional[float], 
        typer.Option(100.0, help="Simplification tolerance for TopoJSON (meters). Set to 0 or negative for no simplification.")
    ] = 100.0,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Creates the study area boundary GeoJSON file by loading UK counties, 
    filtering for a defined Yorkshire region, reprojecting, simplifying, 
    and dissolving.
    """
    create_study_boundary_wrapper(
        raw_counties_file=raw_counties_file,
        output_geojson=output_geojson,
        target_crs=target_crs,
        simplify_tolerance=simplify_tolerance,
        verbose=verbose
    )

@app.command()
def predict(
    ev_path: Annotated[
        Path, 
        typer.Option(
            "data/evs/evs-to-model.tif",
            help="Path to environmental variables raster.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/evs/evs-to-model.tif"),
    models_dir: Annotated[
        Path, 
        typer.Option(
            "data/sdm_models",
            help="Directory containing trained models.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/sdm_models"),
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/sdm_predictions",
            help="Directory for output prediction files.",
            writable=True, resolve_path=True
        )
    ] = Path("data/sdm_predictions"),
    species: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--species", help="Optional: Specific species to generate predictions for (Latin names).")
    ] = None,
    activity_types: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--activity", help="Optional: Specific activity types to generate predictions for.")
    ] = None,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Run the model inference pipeline to generate predictions using trained SDM models.
    
    The pipeline:
    1. Loads the model index and filters models based on species/activity criteria
    2. Loads environmental variables
    3. Generates predictions for each model
    4. Saves results as a multi-band GeoTIFF and summary CSV
    """
    predict_sdm_models(
        ev_path=ev_path,
        models_dir=models_dir,
        output_dir=output_dir,
        species=species,
        activity_types=activity_types,
        verbose=verbose
    )

@app.command()
def train(
    bats_file: Annotated[
        Path, 
        typer.Option(
            "data/processed/bats-tidy.geojson",
            help="Path to bat data file",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/bats-tidy.geojson"),
    background_file: Annotated[
        Path, 
        typer.Option(
            "data/processed/background-points.geojson",
            help="Path to background points file",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/background-points.geojson"),
    ev_file: Annotated[
        Path, 
        typer.Option(
            "data/evs/evs-to-model.tif",
            help="Path to environmental variables file",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/evs/evs-to-model.tif"),
    grid_points_file: Annotated[
        Optional[Path], 
        typer.Option(
            "data/evs/grid-points.parquet",
            help="Path to grid points file (for training data generation). If not provided, grid points will be extracted from the environmental variables file.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/evs/grid-points.parquet"),
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/sdm_models",
            help="Output directory for models and results",
            writable=True, resolve_path=True
        )
    ] = Path("data/sdm_models"),
    min_presence: Annotated[
        int, 
        typer.Option(15, help="Minimum number of presence records required")
    ] = 15,
    n_jobs: Annotated[
        Optional[int], 
        typer.Option(None, help="Number of parallel jobs")
    ] = None,
    max_threads_per_model: Annotated[
        int, 
        typer.Option(2, help="Maximum threads per model")
    ] = 2,
    species: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--species", help="List of species to model. If not provided, all species will be used.")
    ] = None,
    activity_types: Annotated[
        Optional[List[str]], 
        typer.Option(None, "--activity", help="List of activity types to model. If not provided, all activity types will be used.")
    ] = None,
    subset_occurrence: Annotated[
        Optional[int], 
        typer.Option(None, help="If provided, randomly sample this many presence records for each species-activity type combination.")
    ] = None,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Run the MaxEnt model training pipeline.
    
    The pipeline:
    1. Loads and processes bat occurrence data
    2. Loads background points and environmental variables
    3. Generates training data for each species-activity combination
    4. Trains MaxEnt models in parallel
    5. Saves models, results, and logs to MLflow
    """
    train_sdm_models(
        bats_file=bats_file,
        background_file=background_file,
        ev_file=ev_file,
        grid_points_file=grid_points_file,
        output_dir=output_dir,
        min_presence=min_presence,
        n_jobs=n_jobs,
        max_threads_per_model=max_threads_per_model,
        species=species,
        activity_types=activity_types,
        subset_occurrence=subset_occurrence,
        verbose=verbose
    )

@app.command()
def terrain(
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/evs/terrain", 
            help="Directory to save the output terrain TIFF file.",
            writable=True, resolve_path=True
        )
    ] = Path("data/evs/terrain"),
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file for defining processing extent.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    buffer_distance_m: Annotated[
        float, 
        typer.Option(7000, help="Buffer distance in meters for the boundary.")
    ] = 7000,
    wcs_tile_width_px: Annotated[
        int, 
        typer.Option(1024, help="Width of WCS request tiles in pixels.")
    ] = 1024,
    wcs_tile_height_px: Annotated[
        int, 
        typer.Option(1024, help="Height of WCS request tiles in pixels.")
    ] = 1024,
    wcs_temp_storage: Annotated[
        bool, 
        typer.Option(True, help="Use temporary disk storage for WCS tiles.")
    ] = True,
    wcs_download_resolution_m: Annotated[
        int, 
        typer.Option(10, help="Target resolution in meters for initial WCS GetCoverage download.")
    ] = 10,
    max_concurrent_downloads: Annotated[
        int, 
        typer.Option(5, help="Maximum concurrent WCS download requests.")
    ] = 5,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Downloads and processes DTM and DSM terrain data from EA WCS services.
    
    The process involves:
    1. Loading the study boundary and spatial configuration.
    2. Initializing WCS downloaders for DTM and DSM.
    3. Fetching data for DTM and DSM layers via WCS GetCoverage requests (optionally tiled).
    4. Reprojecting each layer to the model's reference CRS and resolution.
    5. Merging the DTM and DSM layers into a single multi-band GeoTIFF.
    6. Saving the final raster.
    """
    generate_terrain_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        buffer_distance_m=buffer_distance_m,
        wcs_tile_width_px=wcs_tile_width_px,
        wcs_tile_height_px=wcs_tile_height_px,
        wcs_temp_storage=wcs_temp_storage,
        wcs_download_resolution_m=wcs_download_resolution_m,
        max_concurrent_downloads=max_concurrent_downloads,
        verbose=verbose
    )

@app.command()
def landcover(
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/evs/landcover", 
            help="Directory to save the output CEH land cover GeoTIFF.",
            writable=True, resolve_path=True
        )
    ] = Path("data/evs/landcover"),
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file for clipping and context.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    ceh_data_path: Annotated[
        Path, 
        typer.Option(
            "data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif",
            help="Path to the raw CEH land cover GeoTIFF file (e.g., gblcm2023_10m.tif).",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"),
    buffer_distance_m: Annotated[
        float, 
        typer.Option(1000, help="Buffer distance in meters for the boundary when clipping raw data.")
    ] = 1000,
    output_resolution_m: Annotated[
        int, 
        typer.Option(100, help="Target output resolution in meters for the processed land cover EV.")
    ] = 100,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Process CEH land cover data based on a given boundary.
    
    Steps include:
    1. Loading the boundary from the specified path
    2. Loading the CEH land cover data
    3. Clipping the data to the boundary with buffer
    4. Converting the land cover data into category layers
    5. Coarsening the data to a lower resolution (e.g., 100m)
    6. Performing feature engineering (aggregates categories)
    7. Reprojecting to the model CRS and writing the output
    """
    generate_ceh_lc_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        ceh_data_path=ceh_data_path,
        buffer_distance_m=buffer_distance_m,
        output_resolution_m=output_resolution_m,
        verbose=verbose
    )

@app.command()
def vom(
    output_dir: Annotated[
        Path, 
        typer.Option(
            "data/evs/terrain", # VOM is often with terrain, adjust if different EV category
            help="Directory to save the output VOM summary TIFF file.",
            writable=True, resolve_path=True
        )
    ] = Path("data/evs/terrain"),
    boundary_path: Annotated[
        Path, 
        typer.Option(
            "data/processed/boundary.geojson", 
            help="Path to the boundary file for defining processing extent.",
            exists=True, readable=True, resolve_path=True
        )
    ] = Path("data/processed/boundary.geojson"),
    buffer_distance_m: Annotated[
        float, 
        typer.Option(7000, help="Buffer distance in meters for the boundary.")
    ] = 7000,
    wcs_tile_width_px: Annotated[
        int, 
        typer.Option(1024, help="Width of WCS request tiles in pixels.")
    ] = 1024,
    wcs_tile_height_px: Annotated[
        int, 
        typer.Option(1024, help="Height of WCS request tiles in pixels.")
    ] = 1024,
    wcs_temp_storage: Annotated[
        bool, 
        typer.Option(True, help="Use temporary disk storage for WCS tiles.")
    ] = True,
    wcs_download_resolution_m: Annotated[
        int, 
        typer.Option(10, help="Target resolution (meters) for initial VOM WCS GetCoverage download.")
    ] = 10,
    max_concurrent_downloads: Annotated[
        int, 
        typer.Option(5, help="Maximum concurrent WCS download requests.")
    ] = 5,
    summary_target_resolution_m: Annotated[
        int, 
        typer.Option(100, help="Target resolution (meters) for VOM summary metrics (e.g., mean, min, max, std).")
    ] = 100,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Downloads Vegetation Object Model (VOM) data, summarises it to various metrics
    (mean, min, max, std) at a specified resolution, and saves as a multi-band GeoTIFF.
    """
    generate_vom_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        buffer_distance_m=buffer_distance_m,
        wcs_tile_width_px=wcs_tile_width_px,
        wcs_tile_height_px=wcs_tile_height_px,
        wcs_temp_storage=wcs_temp_storage,
        wcs_download_resolution_m=wcs_download_resolution_m,
        max_concurrent_downloads=max_concurrent_downloads,
        summary_target_resolution_m=summary_target_resolution_m,
        verbose=verbose
    )

@app.command()
def terrain_stats(
    input_dem_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to the input Digital Elevation Model (DEM) GeoTIFF file.",
            exists=True, readable=True, resolve_path=True
        )
    ],
    output_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to save the output multi-band terrain statistics GeoTIFF.",
            writable=True, resolve_path=True
        )
    ],
    dem_band_index: Annotated[
        int, 
        typer.Option(0, help="0-indexed band number in the DEM file to process (e.g., 0 for first band).")
    ] = 0,
    slope_window_size: Annotated[
        int, 
        typer.Option(3, help="Window size (pixels) for calculating roughness (std dev of slope).")
    ] = 3,
    tpi_window_size: Annotated[
        int, 
        typer.Option(3, help="Window size (pixels) for calculating Topographic Position Index (TPI).")
    ] = 3,
    output_slope_units: Annotated[
        str, 
        typer.Option("radians", help="Units for the output slope layer ('degrees', 'percent', or 'radians').")
    ] = "radians",
    drop_dem_from_output: Annotated[
        bool, 
        typer.Option(True, help="Whether to drop the original DEM layer from the output terrain statistics file.")
    ] = True,
    verbose: Annotated[
        bool, 
        typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
    ] = False
) -> None:
    """
    Calculates various terrain statistics from an input Digital Elevation Model (DEM)
    and saves them as a multi-band GeoTIFF file.

    Terrain attributes calculated include:
    - Slope
    - Aspect (Eastness, Northness)
    - Topographic Wetness Index (TWI)
    - Planform Curvature
    - Roughness (std dev of slope)
    - Topographic Position Index (TPI)
    - Slope-weighted aspect components
    """
    generate_terrain_stats(
        input_dem_path=input_dem_path,
        output_path=output_path,
        dem_band_index=dem_band_index,
        slope_window_size=slope_window_size,
        tpi_window_size=tpi_window_size,
        output_slope_units=output_slope_units,
        drop_dem_from_output=drop_dem_from_output,
        verbose=verbose
    )

if __name__ == "__main__":
    app() 