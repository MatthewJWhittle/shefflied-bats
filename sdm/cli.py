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
    live_download: bool = True,
    verbose: bool = False
) -> None:
    """Set up the project by creating the study boundary."""
    from sdm.data.spatial import create_boundary
    
    setup_logging(verbose=verbose)
    
    boundary_path = Path(PROJECT_CONFIG.paths.boundary)
    
    boundary_gdf = create_boundary(
        counties_file=counties_file,
        county_names=None,  # Yorkshire default
        target_crs=PROJECT_CONFIG.crs,
        simplify_tolerance=100.0,
        live_download=live_download,
    )
    
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_gdf.to_file(boundary_path, driver="GeoJSON")
    logging.info(f"Study boundary saved to: {boundary_path}")

def run_data_rebuild(
    output_dir: str,
    *,
    cache_dir: Optional[str] = None,
    fresh: bool = False,
    live_download: bool = True,
    skip: Optional[str] = None,
    only: Optional[str] = None,
    ceh_path: Optional[Path] = None,
    coastline_path: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """Core EV rebuild used by ``sdm data`` and ``sdm pipeline``.

    Plain defaults (not ``typer.Option``) so callers that invoke this as a
    Python function cannot accidentally treat OptionInfo as truthy and skip
    steps. ``skip`` / ``only`` are comma-separated registry step names.
    """
    from sdm.commands.data_preparation.rebuild_ev_layers import (
        ForbiddenEvOutputError,
        rebuild_environmental_layers,
    )
    from sdm.commands.data_preparation.rebuild_steps import (
        UnknownRebuildStepError,
        parse_step_name_list,
        resolve_active_steps,
    )
    from sdm.utils.rebuild_provenance import DEFAULT_CEH_PATH

    try:
        skip_steps = parse_step_name_list(skip)
        only_steps = parse_step_name_list(only)
        resolve_active_steps(skip=skip_steps, only=only_steps)
    except (UnknownRebuildStepError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        return rebuild_environmental_layers(
            output_dir,
            boundary_path=Path(PROJECT_CONFIG.paths.boundary),
            resolution=PROJECT_CONFIG.spatial.resolution,
            live_download=live_download,
            skip=skip_steps,
            only=only_steps,
            cache_dir=Path(cache_dir) if cache_dir else None,
            fresh=fresh,
            ceh_data_path=Path(ceh_path) if ceh_path is not None else DEFAULT_CEH_PATH,
            coastline_path=Path(coastline_path) if coastline_path is not None else None,
            config_evs=PROJECT_CONFIG.paths.evs,
            config_ev_tiff=PROJECT_CONFIG.paths.ev_tiff,
            verbose=verbose,
        )
    except ForbiddenEvOutputError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc


@app.command()
def data(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=(
            "Local directory for rebuilt EV layers (required unless --plan; taken "
            "as a string so gs:// URIs are not collapsed by pathlib). "
            "Refuses data/evs, paths under inputs/, INPUT_DATA_ROOT, "
            "and gs://…/inputs URIs so the live layer set is not overwritten."
        ),
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help=(
            "Directory for download caches (WorldClim, OS tiles/parquets, "
            "Boundary-Line). Defaults to <output-dir>/cache."
        ),
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help=(
            "Ignore stale OS parquet and Boundary-Line caches and rebuild them "
            "from sources under --cache-dir."
        ),
    ),
    plan: bool = typer.Option(
        False,
        "--plan",
        help=(
            "Print each rebuild step, its source URL or local input path, whether "
            "that input is present, and the OS 100 km tiles needed. No downloads "
            "and no writes. Exits non-zero if a required manual input "
            "(CEH LCM 2023 10 m) is missing."
        ),
    ),
    live_download: bool = True,
    skip: Optional[str] = typer.Option(
        None,
        "--skip",
        help=(
            "Comma-separated rebuild step names to skip "
            "(registry: terrain,terrain_stats,climate,ceh,vom,coastal,os,merge). "
            "Example: --skip ceh,coastal"
        ),
    ),
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help=(
            "Comma-separated rebuild step names to run exclusively "
            "(same registry as --skip). Mutually exclusive with --skip."
        ),
    ),
    ceh_path: Optional[Path] = typer.Option(
        None,
        "--ceh-path",
        help=(
            "Path to the manual CEH LCM 2023 10 m GeoTIFF. "
            "Defaults to the git-ignored raw CEH path under data/raw/big-files/CEH/."
        ),
    ),
    coastline_path: Optional[Path] = typer.Option(
        None,
        "--coastline-path",
        help=(
            "Explicit coastline file for the coastal step (e.g. BGS GeoCoast). "
            "When set, skips the Boundary-Line download and uses this path."
        ),
    ),
    verbose: bool = False,
) -> None:
    """Rebuild all environmental layers into a caller-chosen output folder."""
    from sdm.commands.data_preparation.plan_data_rebuild import run_data_plan
    from sdm.utils.rebuild_provenance import DEFAULT_CEH_PATH

    setup_logging(verbose=verbose)
    resolved_ceh = Path(ceh_path) if ceh_path is not None else DEFAULT_CEH_PATH

    if plan:
        from sdm.commands.data_preparation.rebuild_steps import (
            UnknownRebuildStepError,
        )

        try:
            exit_code = run_data_plan(
                boundary_path=Path(PROJECT_CONFIG.paths.boundary),
                buffer_m=float(PROJECT_CONFIG.spatial.study_area_buffer),
                ceh_data_path=resolved_ceh,
                cache_dir=Path(cache_dir) if cache_dir else None,
                skip=skip,
                only=only,
            )
        except (UnknownRebuildStepError, ValueError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc
        raise typer.Exit(code=exit_code)

    if not output_dir:
        typer.secho(
            "Error: --output-dir / -o is required unless --plan is set.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    run_data_rebuild(
        output_dir,
        cache_dir=cache_dir,
        fresh=fresh,
        live_download=live_download,
        skip=skip,
        only=only,
        ceh_path=ceh_path,
        coastline_path=coastline_path,
        verbose=verbose,
    )

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
    bats_file: str = typer.Option(
        PROJECT_CONFIG.paths.occurence_data,
        help="Path to bat occurrence data (local path or URI such as file:// or gs://).",
    ),
    ev_file: str = typer.Option(
        PROJECT_CONFIG.paths.ev_tiff,
        help="Path to environmental variables raster (local path or URI such as file:// or gs://).",
    ),
    output_dir: Path = typer.Option(
        Path(PROJECT_CONFIG.paths.models),
        help="Model output root; each train run writes under runs/<timestamp-id>/ here.",
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
    allow_config_fallback: bool = typer.Option(
        False,
        help=(
            "Allow missing per-model config folders to fall back to the base "
            "variables_config.yml roster (default: fail with a clear error)."
        ),
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
        allow_config_fallback=allow_config_fallback,
    )
    
    logging.info("Model training complete!")


@app.command()
def predict(
    ev_path: str = typer.Option(
        PROJECT_CONFIG.paths.ev_tiff,
        help="Path to environmental variables raster (local path or URI such as file:// or gs://).",
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
    prediction_crs: Optional[str] = typer.Option(
        None,
        "--prediction-crs",
        help="CRS for prediction rasters (default: project CRS from config.yml).",
    ),
    write_cog: bool = typer.Option(
        True,
        "--cog/--no-cog",
        help="Emit Cloud Optimized GeoTIFF outputs (default: on).",
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
        prediction_crs=prediction_crs,
        write_cog=write_cog,
        verbose=verbose,
    )
    
    logging.info("Predictions generated!")


@app.command("export-rasters")
def export_rasters_cmd(
    raster_paths: List[Path] = typer.Argument(
        ...,
        help="One or more GeoTIFF paths to export.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory for exported files.",
    ),
    output_crs: Optional[str] = typer.Option(
        None,
        "--output-crs",
        help='Target CRS (e.g. EPSG:3857). Omit to keep source CRS/grid.',
    ),
    as_cog: bool = typer.Option(
        False,
        "--cog/--no-cog",
        help="Cloud Optimized GeoTIFF encoding (orthogonal to --output-crs).",
    ),
    resampling: str = typer.Option(
        "bilinear",
        "--resampling",
        help="Warp resampling when using --output-crs: nearest, bilinear, cubic, average.",
    ),
    verbose: bool = False,
) -> None:
    """Reproject and/or COG-wrap rasters (secondary bundles; ``sdm predict`` already emits COG by default)."""
    from sdm.commands.modelling.export_rasters import export_raster_paths

    setup_logging(verbose=verbose)
    try:
        written = export_raster_paths(
            raster_paths,
            output_dir,
            output_crs=output_crs,
            as_cog=as_cog,
            resampling=resampling,
            quiet_cog=not verbose,
        )
    except ValueError as e:
        typer.secho(str(e), err=True)
        raise typer.Exit(code=1) from e

    for path in written:
        logging.info("Wrote %s", path)


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


@app.command("compare-layers")
def compare_layers(
    rebuilt_dir: Path = typer.Argument(
        ...,
        help="Folder of rebuilt environmental layers (GeoTIFF files or multi-band stacks).",
    ),
    current_dir: Path = typer.Argument(
        ...,
        help="Folder of the current / reference environmental layers.",
    ),
    output_csv: Path = typer.Option(
        Path("layer_comparison/layer_comparison.csv"),
        "--output-csv",
        "-o",
        help=(
            "CSV table with one row per layer. Must not be under data/evs or inputs/ "
            "(same write guard as the rebuild command)."
        ),
    ),
    priority_config: Optional[List[Path]] = typer.Option(
        None,
        "--priority-config",
        help=(
            "Repeatable path to a variables_config.yml. Priority order is the union "
            "of variables across these configs (first-seen order). Default: all "
            "data/sdm_config/*/variables_config.yml."
        ),
    ),
    difference_maps_dir: Optional[Path] = typer.Option(
        None,
        "--difference-maps-dir",
        help=(
            "Directory for per-layer difference GeoTIFFs and patch tables "
            "(default: alongside the CSV). Must not be under data/evs or inputs/."
        ),
    ),
    source_versions: Optional[Path] = typer.Option(
        None,
        "--source-versions",
        help="Optional YAML mapping layer → {current, rebuilt} source versions.",
    ),
    model_hooks_csv: Optional[Path] = typer.Option(
        None,
        "--model-hooks-csv",
        help="Optional stub CSV for later model AUC / cut-off checks (not run here).",
    ),
    stack_only: bool = typer.Option(
        False,
        "--stack-only",
        help="Only read **/evs-to-model.tif stacks (ignore intermediate layer files).",
    ),
    file_glob: Optional[str] = typer.Option(
        None,
        "--glob",
        help="Only read raster files matching this glob (mutually exclusive with --stack-only).",
    ),
    align: bool = typer.Option(
        False,
        "--align",
        help=(
            "Crop / snap the rebuilt stack onto the current reference grid "
            "(rioxarray reproject_match) when shapes differ. Grid alignment is "
            "otherwise a hard precondition."
        ),
    ),
    flag_layer: Optional[List[str]] = typer.Option(
        None,
        "--flag-layer",
        help=(
            "Repeatable layer name treated as an intended method change: report "
            "stats with verdict 'flagged' (not fail). Example: OS distance layers "
            "after the #42 edge-distance change."
        ),
    ),
    flag_prefix: Optional[List[str]] = typer.Option(
        None,
        "--flag-prefix",
        help=(
            "Repeatable name prefix for intended method-change layers "
            "(e.g. os_distance). Matching layers are flagged, not failed."
        ),
    ),
    verbose: bool = False,
) -> None:
    """Compare rebuilt environmental layers to the current set (issue #38).

    Closeness, not exactness: grid must align; continuous layers pass when ≥95%
    of cells are within 1% of the layer range; categorical layers when ≥95%
    match. Nodata mismatches count as not-close and are reported separately.
    Patch compactness is report-only. Intended method changes use --flag-layer /
    --flag-prefix (flagged, not fail).
    """
    from sdm.commands.data_preparation.processing.compare_ev_layers import (
        compare_ev_layers,
    )
    from sdm.raster.layer_compare import DuplicateLayerError
    from sdm.utils.ev_output_guards import ForbiddenEvOutputError

    setup_logging(verbose=verbose)
    if stack_only and file_glob:
        typer.secho("Use only one of --stack-only and --glob", err=True)
        raise typer.Exit(code=2)
    try:
        summary = compare_ev_layers(
            rebuilt_dir=rebuilt_dir,
            current_dir=current_dir,
            output_csv=output_csv,
            priority_config_paths=priority_config,
            difference_maps_dir=difference_maps_dir,
            source_versions_path=source_versions,
            write_model_hooks_csv=model_hooks_csv,
            stack_only=stack_only,
            file_glob=file_glob,
            align=align,
            flag_names=flag_layer,
            flag_prefixes=flag_prefix,
            verbose=verbose,
        )
    except ForbiddenEvOutputError as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except DuplicateLayerError as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    n_pass = sum(1 for r in summary.layers if r.verdict.value == "pass")
    n_fail = sum(1 for r in summary.layers if r.verdict.value == "fail")
    n_flag = sum(
        1
        for r in summary.layers
        if r.verdict.value in {"flagged", "flagged_source_change"}
    )
    n_missing = sum(1 for r in summary.layers if r.verdict.value == "missing")
    logging.info(
        "Layer comparison complete: %s pass, %s fail, %s flagged, %s missing → %s",
        n_pass,
        n_fail,
        n_flag,
        n_missing,
        output_csv,
    )
    if summary.duration_seconds is not None:
        logging.info(
            "Stack compare timing: %.3fs, peak RSS=%s bytes",
            summary.duration_seconds,
            summary.peak_rss_bytes,
        )


@app.command()
def explain(
    ev_path: str = typer.Option(
        PROJECT_CONFIG.paths.ev_tiff,
        help="Path to environmental variables raster (local path or URI such as file:// or gs://).",
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
    n_ev_pool: int = typer.Option(
        10_000,
        help="Number of points to sample from EV raster as SHAP pool.",
    ),
    n_explain: int = typer.Option(
        200,
        help="Number of points to explain (more points = more detailed plots).",
    ),
    n_background: int = typer.Option(
        1000,
        help="Number of background samples for the SHAP explainer.",
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
        n_ev_pool=n_ev_pool,
        n_explain=n_explain,
        n_background=n_background,
        top_features=top_features,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
    logging.info("SHAP explanation complete!")

@app.command()
def pipeline(
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help=(
            "Local directory for rebuilt EV layers (passed to `sdm data`). "
            "Must not be data/evs, inputs/, or a bucket inputs URI."
        ),
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Download cache root for `sdm data` (default: <output-dir>/cache).",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Rebuild OS parquet / Boundary-Line caches under --cache-dir.",
    ),
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
    
    # Run all steps via core helpers (not Typer command callables) so OptionInfo
    # defaults cannot be treated as true and silently skip layers.
    if generate_background:
        background(verbose=verbose)
    run_data_rebuild(
        output_dir=output_dir,
        cache_dir=cache_dir,
        fresh=fresh,
        live_download=True,
        skip=None,
        only=None,
        verbose=verbose,
    )
    rebuilt_ev = str(Path(output_dir) / "evs-to-model.tif")
    train(
        species=species,
        activity_types=activity_types,
        ev_file=rebuilt_ev,
        verbose=verbose,
    )
    predict(
        species=species,
        activity_types=activity_types,
        ev_path=rebuilt_ev,
        verbose=verbose,
    )
    visualize(
        species=species,
        activity_types=activity_types,
        ev_raster_path=Path(rebuilt_ev),
        verbose=verbose,
    )
    
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
    bats_file: str = typer.Option(
        PROJECT_CONFIG.paths.occurence_data,
        help="Path to bat occurrence data (local path or URI such as file:// or gs://).",
    ),
    ev_file: str = typer.Option(
        PROJECT_CONFIG.paths.ev_tiff,
        help="Path to environmental variables raster (local path or URI such as file:// or gs://).",
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
