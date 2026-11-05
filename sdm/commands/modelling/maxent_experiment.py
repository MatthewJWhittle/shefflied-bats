"""Interactive MaxEnt experiment helpers for notebooks.

Same library path as ``sdm train`` / ``train_sdm_models``: shared background
draw, elapid annotate, per-model YAML under ``data/sdm_config/<id>/``, spatial
block CV, seed, and ``pooled_oof_percentile`` cut-offs. No duplicated pipeline.

Workflow for fast iteration:

1. ``cache = prepare_maxent_experiment(...)`` — load stack, draw background,
   annotate (expensive; run once).
2. Edit the overrides cell (variables / MaxEnt / background prep knobs).
3. ``result = train_from_cache(cache, ...)`` — species prep + fit only when
   background surface settings are unchanged.

With empty overrides, ``train_from_cache`` matches ``train_sdm_models`` for the
same species × activity, seed, and inputs. Predictions stay under a local temp
directory — never write to the prediction publish bucket from these helpers.
"""

from __future__ import annotations

import logging
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import geopandas as gpd
import pandas as pd
import xarray as xr

from sdm.commands.modelling.train_sdm_models import (
    prepare_results_dataframe,
    prepare_species_training_data,
    prepare_training_data,
    save_models,
    save_results,
    save_run_metadata,
    save_run_thresholds,
    save_training_data,
    simplify_boundary,
)
from sdm.commands.modelling.utils import get_model_id
from sdm.data.loaders.vector import load_bat_data
from sdm.models.adapters import get_adapter
from sdm.models.maxent.folds import FoldParams, resolve_fold_params
from sdm.models.maxent.maxent_model import DefaultMaxentConfig
from sdm.models.maxent.training import summarize_cv_scores
from sdm.occurrence.background_settings import (
    BackgroundSurface,
    ResolvedBackgroundSettings,
    resolve_background_settings,
)
from sdm.occurrence.sampling import BackgroundMethod, TransformMethod
from sdm.raster.io import load_environmental_variables
from sdm.types import TrainingData, TrainingResults, VariablesConfig
from sdm.utils.fs import local_materialized_path
from sdm.utils.io import (
    CONFIG_PATH,
    MODEL_CONFIG_PATH,
    load_boundary,
    load_model_config,
    load_project_config,
    load_variables_config,
)
from sdm.utils.logging_utils import setup_logging
from sdm.utils.model_runs import (
    artifact_parent_uri,
    build_run_output_uri,
    join_run_artifact,
    model_run_write_context,
    resolve_data_path,
    resolve_output_root,
    run_segment_from_uri,
)
from sdm.utils.sdm_config import (
    model_config_file_paths,
    repo_relative_config_path,
    resolve_model_config_dir,
    resolve_sdm_config_root,
)

logger = logging.getLogger(__name__)

# Background knobs that force a fresh shared background draw + annotate.
_SURFACE_BACKGROUND_KEYS = frozenset(
    {
        "n_background_points",
        "surface",
        "background_method",
        "background_value",
        "sigma",
        "transform_method",
        "surface_resolution_m",
        "seed",
    }
)

# Per-model prep knobs that force ``prepare_species_training_data`` again.
_SPECIES_PREP_KEYS = frozenset(
    {
        "presence_exclusion_m",
        "presence_exclusion_max_m",
        "background_thin_m",
        "presence_thin_m",
        "cap_ratio",
        "min_background",
        "max_background",
        "distance_weights",
        "background_distance_weights",
        "sample_weight_n_neighbors",
        "class_balance",
        "seed",
    }
)

_MAXENT_OVERRIDE_KEYS = frozenset(
    {
        "feature_types",
        "beta_multiplier",
        "beta_lqp",
        "beta_hinge",
        "beta_threshold",
        "beta_categorical",
        "n_hinge_features",
        "n_threshold_features",
        "clamp",
        "convergence_tolerance",
        "use_lambdas",
        "n_lambdas",
        "class_weights",
        "tau",
        "transform",
    }
)


@dataclass
class MaxentExperimentCache:
    """Cached expensive steps shared across override retrains."""

    latin_name: str
    activity_type: str
    presence_with_evs: gpd.GeoDataFrame
    background_with_evs: gpd.GeoDataFrame
    background_occurrence_gdf: gpd.GeoDataFrame
    filtered_occurrence_gdf: gpd.GeoDataFrame
    boundary: gpd.GeoDataFrame
    evs_to_model: xr.Dataset
    ev_columns: List[str]
    background_settings: ResolvedBackgroundSettings
    fold_params: FoldParams
    training_seed: int
    maxent_config: DefaultMaxentConfig
    model_features: List[str]
    model_limited: bool
    model_config_path: Path
    project_config_path: Path
    config_dir: Optional[Path]
    trainer: str
    baseline_mean_cv: Optional[float] = None
    baseline_std_cv: Optional[float] = None
    baseline_features: Optional[List[str]] = None
    _species_data: Optional[gpd.GeoDataFrame] = field(default=None, repr=False)


@dataclass
class MaxentExperimentResult:
    """One experiment retrain: metrics, models, and local run URI."""

    results_df: pd.DataFrame
    models: List[TrainingResults]
    training_data: List[TrainingData]
    run_uri: str
    mean_cv_score: float
    std_cv_score: float
    features: List[str]
    baseline_mean_cv: Optional[float]
    baseline_std_cv: Optional[float]
    wall_seconds: float

    @property
    def final_model(self) -> Any:
        if not self.models or self.models[0].final_model is None:
            raise RuntimeError("No fitted model on this experiment result")
        return self.models[0].final_model

    @property
    def package_dir(self) -> str:
        paths = self.results_df["model_package_dir"].tolist()
        if not paths or not paths[0]:
            raise RuntimeError("No model package directory on this result")
        return str(paths[0])


def train_maxent_experiment(
    *,
    latin_name: str,
    activity_type: str,
    project_config_path: Path = CONFIG_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    variables_config_path: Optional[Path] = None,
    config_dir: Optional[Path] = None,
    bats_file: Optional[Path] = None,
    ev_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_jobs: Optional[int] = None,
    max_threads_per_model: int = 1,
    verbose: bool = False,
    allow_config_fallback: bool = False,
    cv_method: Optional[str] = None,
    cv_n_folds: Optional[int] = None,
    cv_block_size_m: Optional[float] = None,
    cv_block_n_folds: Optional[int] = None,
    cv_buffer_m: Optional[float] = None,
    cv_buffered_loo_warn_above: Optional[int] = None,
    n_background_points: Optional[int] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    sample_weight_n_neighbors: Optional[int] = None,
) -> pd.DataFrame:
    """Train one species × activity via the same path as ``sdm train``.

    Thin wrapper around ``train_sdm_models`` so notebooks do not invent a second
    pipeline. Prefer ``prepare_maxent_experiment`` + ``train_from_cache`` when
    iterating on overrides.
    """
    from sdm.commands.modelling.train_sdm_models import train_sdm_models

    return train_sdm_models(
        project_config_path=project_config_path,
        model_config_path=model_config_path,
        variables_config_path=variables_config_path,
        config_dir=config_dir,
        bats_file=bats_file,
        ev_file=ev_file,
        output_dir=output_dir,
        n_jobs=n_jobs,
        max_threads_per_model=max_threads_per_model,
        species=[latin_name],
        activity_types=[activity_type],
        verbose=verbose,
        allow_config_fallback=allow_config_fallback,
        cv_method=cv_method,
        cv_n_folds=cv_n_folds,
        cv_block_size_m=cv_block_size_m,
        cv_block_n_folds=cv_block_n_folds,
        cv_buffer_m=cv_buffer_m,
        cv_buffered_loo_warn_above=cv_buffered_loo_warn_above,
        n_background_points=n_background_points,
        d_min=d_min,
        d_max=d_max,
        sample_weight_n_neighbors=sample_weight_n_neighbors,
    )


def prepare_maxent_experiment(
    *,
    latin_name: str = "Plecotus auritus",
    activity_type: str = "Roost",
    project_config_path: Path = CONFIG_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    variables_config_path: Optional[Path] = None,
    config_dir: Optional[Path] = None,
    bats_file: Optional[Path] = None,
    ev_file: Optional[Path] = None,
    verbose: bool = False,
    allow_config_fallback: bool = False,
    cv_method: Optional[str] = None,
    cv_n_folds: Optional[int] = None,
    cv_block_size_m: Optional[float] = None,
    cv_block_n_folds: Optional[int] = None,
    cv_buffer_m: Optional[float] = None,
    cv_buffered_loo_warn_above: Optional[int] = None,
    n_background_points: Optional[int] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    sample_weight_n_neighbors: Optional[int] = None,
) -> MaxentExperimentCache:
    """Load data, draw background, annotate — same steps as ``train_sdm_models``.

    Does not fit a model. Call ``train_from_cache`` next.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    project_config = load_project_config(project_config_path)
    base_model_cfg = load_model_config(resolve_sdm_config_root(model_config_path))
    if variables_config_path is None:
        variables_config_path = Path(project_config.paths.variables_config_path)
    base_variables_cfg = load_variables_config(
        resolve_sdm_config_root(variables_config_path)
    )

    if config_dir is None:
        config_dir = Path(project_config.paths.config_dir)

    training_seed = base_model_cfg.seed
    background_settings = resolve_background_settings(
        project_config=project_config,
        model_config=base_model_cfg,
        seed=training_seed,
        n_background_points_override=n_background_points,
        presence_exclusion_m_override=d_min,
        presence_exclusion_max_m_override=d_max,
        sample_weight_n_neighbors_override=sample_weight_n_neighbors,
    )
    fold_params = resolve_fold_params(
        base_model_cfg.cv,
        model_seed=training_seed,
        method_override=cv_method,
        n_folds_override=cv_n_folds,
        block_size_m_override=cv_block_size_m,
        block_n_folds_override=cv_block_n_folds,
        buffer_m_override=cv_buffer_m,
        buffered_loo_warn_above_override=cv_buffered_loo_warn_above,
    )

    bats_uri = resolve_data_path(
        bats_file if bats_file is not None else project_config.paths.occurence_data
    )
    ev_uri = resolve_data_path(
        ev_file if ev_file is not None else project_config.paths.ev_tiff
    )
    boundary_uri = resolve_data_path(project_config.paths.boundary)

    with local_materialized_path(bats_uri) as bats_path:
        bats_ant = load_bat_data(bats_path)
    bats_ant = bats_ant[bats_ant.accuracy <= 100]

    with local_materialized_path(boundary_uri) as boundary_path:
        boundary = load_boundary(
            filepath=boundary_path, buffer_distance=0, target_crs=27700
        )
    boundary = simplify_boundary(boundary, tolerance=100)

    evs_to_model, _ = load_environmental_variables(ev_uri)
    all_bats_gdf = bats_ant
    filtered = bats_ant[
        (bats_ant.latin_name == latin_name)
        & (bats_ant.activity_type == activity_type)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No occurrence rows for {latin_name!r} / {activity_type!r} "
            f"after accuracy filter"
        )

    presence_with_evs, background_with_evs, ev_columns = prepare_training_data(
        occurrence_gdf=filtered,
        background_occurrence_gdf=all_bats_gdf,
        boundary=boundary,
        evs_to_model=evs_to_model,
        n_background_points=background_settings.n_background_points,
        background_method=background_settings.background_method,
        background_value=background_settings.background_value,
        sigma=background_settings.sigma,
        transform_method=background_settings.transform_method,
        seed=background_settings.seed,
        surface=background_settings.surface,
        grid_resolution=background_settings.surface_resolution_m,
    )

    maxent_config, model_features, model_limited = _load_per_model_configs(
        latin_name=latin_name,
        activity_type=activity_type,
        config_dir=config_dir,
        base_model_cfg=base_model_cfg,
        base_variables_cfg=base_variables_cfg,
        variables_config_path=variables_config_path,
        ev_columns=ev_columns,
        allow_config_fallback=allow_config_fallback,
        background_settings=background_settings,
    )

    return MaxentExperimentCache(
        latin_name=latin_name,
        activity_type=activity_type,
        presence_with_evs=presence_with_evs,
        background_with_evs=background_with_evs,
        background_occurrence_gdf=all_bats_gdf,
        filtered_occurrence_gdf=filtered,
        boundary=boundary,
        evs_to_model=evs_to_model,
        ev_columns=list(ev_columns),
        background_settings=background_settings,
        fold_params=fold_params,
        training_seed=training_seed,
        maxent_config=maxent_config,
        model_features=list(model_features),
        model_limited=model_limited,
        model_config_path=Path(model_config_path),
        project_config_path=Path(project_config_path),
        config_dir=Path(config_dir) if config_dir is not None else None,
        trainer=base_model_cfg.type,
    )


def train_from_cache(
    cache: MaxentExperimentCache,
    *,
    variables: Optional[Sequence[str]] = None,
    maxent_overrides: Optional[Mapping[str, Any]] = None,
    background_overrides: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[Path] = None,
    n_jobs: Optional[int] = None,
    max_threads_per_model: int = 1,
    set_as_baseline: bool = False,
    verbose: bool = False,
) -> MaxentExperimentResult:
    """Species prep + MaxEnt fit from a prepared cache.

    Override maps:

    - ``variables``: feature list (must exist on the EV stack).
    - ``maxent_overrides``: fields of ``DefaultMaxentConfig`` (e.g. ``beta_multiplier``).
    - ``background_overrides``: fields of ``ResolvedBackgroundSettings``. Surface
      knobs (``n_background_points``, ``surface``, ``seed``, …) redraw background
      and re-annotate; prep-only knobs (thin / exclusion / weights) only rebuild
      the species table.
    """
    import time

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    started = time.perf_counter()

    bg_overrides = _coerce_background_overrides(dict(background_overrides or {}))
    _validate_override_keys(bg_overrides, _SURFACE_BACKGROUND_KEYS | _SPECIES_PREP_KEYS, "background")
    maxent_overrides = dict(maxent_overrides or {})
    _validate_override_keys(maxent_overrides, _MAXENT_OVERRIDE_KEYS, "maxent")

    settings = cache.background_settings
    if bg_overrides:
        settings = replace(settings, **bg_overrides)

    need_surface_refresh = bool(_SURFACE_BACKGROUND_KEYS & set(bg_overrides))
    need_species_refresh = (
        need_surface_refresh
        or bool(_SPECIES_PREP_KEYS & set(bg_overrides))
        or cache._species_data is None
    )

    presence_with_evs = cache.presence_with_evs
    background_with_evs = cache.background_with_evs
    ev_columns = cache.ev_columns

    if need_surface_refresh:
        logger.info("Background surface overrides set — redrawing + annotating")
        presence_with_evs, background_with_evs, ev_columns = prepare_training_data(
            occurrence_gdf=cache.filtered_occurrence_gdf,
            background_occurrence_gdf=cache.background_occurrence_gdf,
            boundary=cache.boundary,
            evs_to_model=cache.evs_to_model,
            n_background_points=settings.n_background_points,
            background_method=settings.background_method,
            background_value=settings.background_value,
            sigma=settings.sigma,
            transform_method=settings.transform_method,
            seed=settings.seed,
            surface=settings.surface,
            grid_resolution=settings.surface_resolution_m,
        )
        cache.presence_with_evs = presence_with_evs
        cache.background_with_evs = background_with_evs
        cache.ev_columns = list(ev_columns)
        cache.background_settings = settings
        cache._species_data = None
        need_species_refresh = True

    if need_species_refresh:
        species_data = prepare_species_training_data(
            presence_data=presence_with_evs,
            background_data=background_with_evs,
            latin_name=cache.latin_name,
            activity_type=cache.activity_type,
            ev_columns=ev_columns,
            grid_size_m=settings.presence_thin_m,
            d_min=settings.presence_exclusion_m,
            d_max=settings.presence_exclusion_max_m,
            sample_weight_n_neighbors=settings.sample_weight_n_neighbors,
            random_state=settings.seed,
            presence_thin_m=settings.presence_thin_m,
            background_thin_m=settings.background_thin_m,
            distance_weights=settings.distance_weights,
            background_distance_weights=settings.background_distance_weights,
            cap_ratio=settings.cap_ratio,
            min_background=settings.min_background,
            max_background=settings.max_background,
        )
        if species_data.empty or "class" not in species_data.columns:
            raise ValueError(
                f"No training rows for {cache.latin_name} / {cache.activity_type}"
            )
        cache._species_data = species_data
        cache.background_settings = settings
    else:
        species_data = cache._species_data
        assert species_data is not None

    features = list(variables) if variables is not None else list(cache.model_features)
    missing = set(features) - set(ev_columns)
    if missing:
        raise ValueError(f"Override variables not in EV stack: {sorted(missing)}")
    VariablesConfig(variables=features).validate_features(ev_columns)

    maxent_config = deepcopy(cache.maxent_config)
    for key, value in maxent_overrides.items():
        setattr(maxent_config, key, value)
    maxent_config.class_weights = settings.maxent_class_weights()

    training_data = [
        TrainingData(
            latin_name=cache.latin_name,
            activity_type=cache.activity_type,
            occurrence=species_data,
            maxent_config=maxent_config,
            model_features=features,
            model_limited=cache.model_limited,
        )
    ]

    adapter = get_adapter(cache.trainer)
    bundle = adapter.train(
        training_data,
        max_threads_per_model=max_threads_per_model,
        n_jobs=n_jobs,
        seed=cache.training_seed,
        fold_params=cache.fold_params,
    )
    models = bundle.results
    if not models or not models[0].success:
        err = models[0].error if models else "no model returned"
        raise RuntimeError(f"Training failed: {err}")

    results_df = prepare_results_dataframe(models, training_data)
    project_config = load_project_config(cache.project_config_path)
    output_root = resolve_output_root(
        cli_output_dir=output_dir,
        config_models_path=project_config.paths.models,
    )
    run_uri = build_run_output_uri(output_root)

    with model_run_write_context(run_uri) as (models_output_dir, written_uri):
        model_paths = save_models(
            models,
            training_data,
            models_output_dir,
            trainer=adapter.name,
        )
        published_paths = {
            identifier: join_run_artifact(
                written_uri,
                Path(path).relative_to(models_output_dir),
            )
            for identifier, path in model_paths.items()
        }
        results_df["model_path"] = [
            published_paths.get(identifier, "")
            for identifier in results_df["identifier"]
        ]
        results_df["model_package_dir"] = [
            artifact_parent_uri(p) if p else ""
            for p in results_df["model_path"]
        ]
        save_results(results_df, models_output_dir)
        save_training_data(training_data, models_output_dir)
        save_run_thresholds(models, training_data, models_output_dir)
        save_run_metadata(
            models_output_dir,
            seed=cache.training_seed,
            model_config_path=repo_relative_config_path(cache.model_config_path),
            fold_params=cache.fold_params,
            trainer=adapter.name,
            background_settings=settings,
        )

    mean_cv, std_cv, _, _ = summarize_cv_scores(models[0].cv_scores)
    wall = time.perf_counter() - started

    if set_as_baseline or cache.baseline_mean_cv is None:
        cache.baseline_mean_cv = mean_cv
        cache.baseline_std_cv = std_cv
        cache.baseline_features = list(features)

    logger.info(
        "Experiment %s: mean CV AUC=%.4f ± %.4f (baseline %.4f ± %.4f) in %.1fs → %s",
        get_model_id([cache.latin_name, cache.activity_type]),
        mean_cv,
        std_cv,
        cache.baseline_mean_cv if cache.baseline_mean_cv is not None else float("nan"),
        cache.baseline_std_cv if cache.baseline_std_cv is not None else float("nan"),
        wall,
        written_uri,
    )
    logger.info("Run id for predict: %s", run_segment_from_uri(str(written_uri)))

    return MaxentExperimentResult(
        results_df=results_df,
        models=models,
        training_data=training_data,
        run_uri=str(written_uri),
        mean_cv_score=float(mean_cv),
        std_cv_score=float(std_cv),
        features=list(features),
        baseline_mean_cv=cache.baseline_mean_cv,
        baseline_std_cv=cache.baseline_std_cv,
        wall_seconds=float(wall),
    )


def default_scratch_output_dir() -> Path:
    """Local-only output root for notebook experiments (never the publish bucket)."""
    root = Path(tempfile.gettempdir()) / "sdm-maxent-experiment"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_per_model_configs(
    *,
    latin_name: str,
    activity_type: str,
    config_dir: Optional[Path],
    base_model_cfg: Any,
    base_variables_cfg: VariablesConfig,
    variables_config_path: Path,
    ev_columns: List[str],
    allow_config_fallback: bool,
    background_settings: ResolvedBackgroundSettings,
) -> tuple[DefaultMaxentConfig, List[str], bool]:
    default_model_config = DefaultMaxentConfig.from_config(config=base_model_cfg)
    default_model_config.class_weights = background_settings.maxent_class_weights()
    identifier = get_model_id([latin_name, activity_type])

    if config_dir is None:
        features = base_variables_cfg.validate_features(ev_columns).variables
        return default_model_config, list(features), False

    try:
        model_subdir = resolve_model_config_dir(config_dir, identifier, must_exist=True)
        per_model_config_path, per_model_variables_path = model_config_file_paths(
            model_subdir
        )
        model_config = load_model_config(
            per_model_config_path,
            fallback_seed=base_model_cfg.seed,
        )
        maxent_config = DefaultMaxentConfig.from_config(config=model_config)
        model_limited = bool(model_config.model_limited)
        variables_config = load_variables_config(per_model_variables_path)
        features = variables_config.validate_features(ev_columns).variables
        maxent_config.class_weights = background_settings.maxent_class_weights()
        return maxent_config, list(features), model_limited
    except FileNotFoundError:
        if not allow_config_fallback:
            raise
        logger.warning(
            "%s: missing per-model config; falling back to %s",
            identifier,
            resolve_sdm_config_root(variables_config_path),
        )
        features = base_variables_cfg.validate_features(ev_columns).variables
        return default_model_config, list(features), False


def _validate_override_keys(
    overrides: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} override keys: {unknown}")


def _coerce_background_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Accept string enum names in notebook overrides."""
    out = dict(overrides)
    if "surface" in out and not isinstance(out["surface"], BackgroundSurface):
        out["surface"] = BackgroundSurface(str(out["surface"]).lower())
    if "background_method" in out and not isinstance(
        out["background_method"], BackgroundMethod
    ):
        out["background_method"] = BackgroundMethod(
            str(out["background_method"]).lower()
        )
    if "transform_method" in out and not isinstance(
        out["transform_method"], TransformMethod
    ):
        out["transform_method"] = TransformMethod(
            str(out["transform_method"]).lower()
        )
    return out
