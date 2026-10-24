"""Orchestrate a full environmental-layer rebuild into a caller-chosen folder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from sdm.commands.data_preparation.environmental.generate_ceh_lc_data import (
    generate_ceh_lc_data,
)
from sdm.commands.data_preparation.environmental.generate_climate_data import (
    generate_climate_data,
)
from sdm.commands.data_preparation.environmental.generate_terrain_data import (
    generate_terrain_data,
)
from sdm.commands.data_preparation.environmental.generate_terrain_stats import (
    generate_terrain_stats,
)
from sdm.commands.data_preparation.environmental.generate_vom_data import (
    generate_vom_data,
)
from sdm.commands.data_preparation.processing.build_model_ev_stack import (
    build_model_ev_stack,
    build_settings_from_config,
)
from sdm.commands.data_preparation.processing.merge_ev_layers import (
    build_ev_dataset_inputs,
    merge_ev_layers,
)
from sdm.commands.data_preparation.processing.process_os_data import process_os_data
from sdm.commands.data_preparation.rebuild_steps import (
    REBUILD_STEP_NAMES,
    UnknownRebuildStepError,
    resolve_active_steps,
)
from sdm.commands.data_preparation.spatial.generate_coastal_distance import (
    generate_coastal_distance,
)
from sdm.utils.ev_output_guards import (
    ForbiddenEvOutputError,
    forbidden_output_summary,
    validate_ev_output_dir,
)
from sdm.utils.io import load_config
from sdm.utils.rebuild_provenance import (
    DEFAULT_CEH_PATH,
    collect_rebuild_provenance,
)
from sdm.utils.step_metrics import StepMetricsRecorder


def rebuild_cache_layout(cache_root: Path) -> dict[str, Path]:
    """Standard cache subfolders under ``cache_root``."""
    layout = {
        "root": cache_root,
        "worldclim": cache_root / "worldclim",
        "os_raw": cache_root / "os-vector-map",
        "os_parquets": cache_root / "os-parquets",
        "boundary_line": cache_root / "boundary-line",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def rebuild_environmental_layers(
    output_dir: Path | str,
    *,
    boundary_path: Path,
    resolution: int,
    live_download: bool = True,
    skip: Iterable[str] = (),
    only: Iterable[str] = (),
    cache_dir: Path | str | None = None,
    fresh: bool = False,
    ceh_data_path: Path = DEFAULT_CEH_PATH,
    coastline_path: Path | None = None,
    buffer_distance_m: float = 0,
    config_evs: str | Path | None = None,
    config_ev_tiff: str | Path | None = None,
    metrics_stem: str = "rebuild_metrics",
    verbose: bool = False,
) -> Path:
    """Rebuild every EV layer into ``output_dir`` and write step metrics beside it.

    Download caches (WorldClim, OS tiles/parquets, Boundary-Line) go under
    ``cache_dir`` when set, otherwise ``<output_dir>/cache``. Pass ``fresh=True``
    to ignore/rebuild stale cached parquets and Boundary-Line extracts.

    ``skip`` / ``only`` take registry step names (see
    ``sdm.commands.data_preparation.rebuild_steps``); they are mutually
    exclusive.

    Raises
    ------
    ForbiddenEvOutputError
        If ``output_dir`` is the live EV set, under ``inputs/``, under
        ``INPUT_DATA_ROOT``, or a ``gs://…/inputs`` URI.
    UnknownRebuildStepError
        If ``skip`` or ``only`` names a step outside the registry.
    ValueError
        If both ``skip`` and ``only`` are non-empty.
    """
    active = resolve_active_steps(skip=skip, only=only)
    evs_dir = validate_ev_output_dir(
        output_dir,
        config_evs=config_evs,
        config_ev_tiff=config_ev_tiff,
    )
    evs_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_dir) if cache_dir is not None else evs_dir / "cache"
    caches = rebuild_cache_layout(cache_root)
    merged_output = evs_dir / "all-evs.tif"
    model_stack_output = evs_dir / "evs-to-model.tif"
    recorder = StepMetricsRecorder()
    try:
        config_ev_stack = load_config().get("ev_stack")
    except FileNotFoundError:
        config_ev_stack = None
    build_settings = build_settings_from_config(config_ev_stack)
    applied_build_settings: dict = build_settings

    logging.info("Rebuilding environmental layers into %s", evs_dir)
    logging.info("Cache directory: %s (fresh=%s)", cache_root, fresh)
    logging.info("Write guards: %s", forbidden_output_summary())
    logging.info(
        "Model-stack layer clipping enabled=%s (architecture review default ON)",
        build_settings["layer_clipping"]["enabled"],
    )
    if active != REBUILD_STEP_NAMES:
        logging.info("Active steps: %s", ", ".join(sorted(active)))

    def _run_or_skip(name: str, recorder_key: str, fn) -> None:
        if name not in active:
            logging.info("Skipping %s (--skip / --only)", name)
            return
        recorder.run(recorder_key, fn)

    try:
        _run_or_skip(
            "terrain",
            "generate_terrain_data",
            lambda: generate_terrain_data(
                output_dir=evs_dir / "terrain",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "terrain_stats",
            "generate_terrain_stats",
            lambda: generate_terrain_stats(
                input_dem_path=evs_dir / "terrain" / "dtm_dsm_100m.tif",
                output_path=evs_dir / "terrain_stats.tif",
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "climate",
            "generate_climate_data",
            lambda: generate_climate_data(
                output_dir=evs_dir / "climate",
                boundary_path=boundary_path,
                worldclim_cache_dir=caches["worldclim"],
                run_stats=True,
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "ceh",
            "generate_ceh_lc_data",
            lambda: generate_ceh_lc_data(
                output_dir=evs_dir / "landcover",
                boundary_path=boundary_path,
                ceh_data_path=ceh_data_path,
                buffer_distance_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "vom",
            "generate_vom_data",
            lambda: generate_vom_data(
                output_dir=evs_dir / "vom",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "coastal",
            "generate_coastal_distance",
            lambda: generate_coastal_distance(
                boundary_path=boundary_path,
                output_dir=evs_dir,
                coastline_path=coastline_path,
                live_download=live_download,
                cache_dir=caches["boundary_line"],
                overwrite_cache=fresh,
                verbose=verbose,
            ),
        )

        _run_or_skip(
            "os",
            "process_os_data",
            lambda: process_os_data(
                output_dir=evs_dir,
                boundary_path=boundary_path,
                buffer_distance=buffer_distance_m,
                parquet_dir=caches["os_parquets"],
                raw_dir=caches["os_raw"],
                overwrite_cache=fresh,
                live_download=live_download,
                verbose=verbose,
            ),
        )

        # Omit CEH/coastal from merge only when explicitly ``--skip``'d.
        # ``--only merge`` leaves those steps inactive for generation but must
        # still consume existing outputs in the folder (otherwise all-evs.tif
        # is rewritten without them and the model stack KeyErrors).
        explicit_skip = frozenset(skip)
        skip_ceh = "ceh" in explicit_skip
        skip_coastal = "coastal" in explicit_skip
        _run_or_skip(
            "merge",
            "merge_ev_layers",
            lambda: merge_ev_layers(
                dataset_inputs=build_ev_dataset_inputs(
                    evs_dir,
                    resolution,
                    skip_ceh=skip_ceh,
                    skip_coastal=skip_coastal,
                ),
                boundary_path=boundary_path,
                output_path=merged_output,
                boundary_buffer_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        # Model stack needs the merge product. Skip it for --only/--skip paths
        # that did not select merge so a successful single step exits 0.
        if "merge" in active:
            def _build_stack() -> Path:
                nonlocal applied_build_settings
                path, used_settings = build_model_ev_stack(
                    merged_output,
                    model_stack_output,
                    build_settings=build_settings,
                    verbose=verbose,
                )
                applied_build_settings = used_settings
                return path

            recorder.run("build_model_ev_stack", _build_stack)
        else:
            logging.info(
                "Skipping build_model_ev_stack (merge not in active steps)"
            )
    finally:
        if recorder.steps:
            metrics_paths = recorder.write(
                evs_dir,
                stem=metrics_stem,
                extra={
                    "build_settings": applied_build_settings,
                    "provenance": collect_rebuild_provenance(
                        ceh_data_path=ceh_data_path
                    ),
                },
            )
            logging.info(
                "Rebuild metrics written to %s and %s",
                metrics_paths["json"],
                metrics_paths["csv"],
            )

    if "merge" in active:
        logging.info("Environmental data generation complete: %s", model_stack_output)
    else:
        logging.info(
            "Environmental data generation complete (active steps: %s)",
            ", ".join(sorted(active)),
        )
    return evs_dir


__all__ = [
    "DEFAULT_CEH_PATH",
    "ForbiddenEvOutputError",
    "UnknownRebuildStepError",
    "forbidden_output_summary",
    "rebuild_cache_layout",
    "rebuild_environmental_layers",
    "validate_ev_output_dir",
]
