"""Orchestrate a full environmental-layer rebuild into a caller-chosen folder."""

from __future__ import annotations

import logging
from pathlib import Path

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
from sdm.commands.data_preparation.processing.merge_ev_layers import (
    build_ev_dataset_inputs,
    merge_ev_layers,
)
from sdm.commands.data_preparation.processing.process_os_data import process_os_data
from sdm.commands.data_preparation.spatial.generate_coastal_distance import (
    generate_coastal_distance,
)
from sdm.utils.ev_output_guards import (
    ForbiddenEvOutputError,
    forbidden_output_summary,
    validate_ev_output_dir,
)
from sdm.utils.step_metrics import StepMetricsRecorder


def rebuild_environmental_layers(
    output_dir: Path | str,
    *,
    boundary_path: Path,
    resolution: int,
    live_download: bool = True,
    skip_ceh: bool = False,
    skip_coastal: bool = False,
    worldclim_cache_dir: Path = Path("data/raw/worldclim"),
    ceh_data_path: Path = Path(
        "data/raw/big-files/CEH/data/"
        "7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"
    ),
    buffer_distance_m: float = 7000,
    config_evs: str | Path | None = None,
    config_ev_tiff: str | Path | None = None,
    metrics_stem: str = "rebuild_metrics",
    verbose: bool = False,
) -> Path:
    """Rebuild every EV layer into ``output_dir`` and write step metrics beside it.

    Raises
    ------
    ForbiddenEvOutputError
        If ``output_dir`` is the live EV set, under ``inputs/``, under
        ``INPUT_DATA_ROOT``, or a ``gs://…/inputs`` URI.
    """
    evs_dir = validate_ev_output_dir(
        output_dir,
        config_evs=config_evs,
        config_ev_tiff=config_ev_tiff,
    )
    evs_dir.mkdir(parents=True, exist_ok=True)
    merged_output = evs_dir / "evs-to-model.tif"
    recorder = StepMetricsRecorder()

    logging.info("Rebuilding environmental layers into %s", evs_dir)
    logging.info("Write guards: %s", forbidden_output_summary())

    try:
        recorder.run(
            "generate_terrain_data",
            lambda: generate_terrain_data(
                output_dir=evs_dir / "terrain",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        recorder.run(
            "generate_terrain_stats",
            lambda: generate_terrain_stats(
                input_dem_path=evs_dir / "terrain" / "dtm_dsm_100m.tif",
                output_path=evs_dir / "terrain_stats.tif",
                verbose=verbose,
            ),
        )

        recorder.run(
            "generate_climate_data",
            lambda: generate_climate_data(
                output_dir=evs_dir / "climate",
                boundary_path=boundary_path,
                worldclim_cache_dir=worldclim_cache_dir,
                verbose=verbose,
            ),
        )

        if skip_ceh:
            logging.info("Skipping CEH land cover (--skip-ceh)")
        else:
            recorder.run(
                "generate_ceh_lc_data",
                lambda: generate_ceh_lc_data(
                    output_dir=evs_dir / "landcover",
                    boundary_path=boundary_path,
                    ceh_data_path=ceh_data_path,
                    verbose=verbose,
                ),
            )

        recorder.run(
            "generate_vom_data",
            lambda: generate_vom_data(
                output_dir=evs_dir / "vom",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                verbose=verbose,
            ),
        )

        if skip_coastal:
            logging.info("Skipping coastal distance (--skip-coastal)")
        else:
            recorder.run(
                "generate_coastal_distance",
                lambda: generate_coastal_distance(
                    boundary_path=boundary_path,
                    output_dir=evs_dir,
                    live_download=live_download,
                    verbose=verbose,
                ),
            )

        recorder.run(
            "process_os_data",
            lambda: process_os_data(
                output_dir=evs_dir,
                boundary_path=boundary_path,
                buffer_distance=buffer_distance_m,
                verbose=verbose,
            ),
        )

        recorder.run(
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
                verbose=verbose,
            ),
        )
    finally:
        if recorder.steps:
            metrics_paths = recorder.write(evs_dir, stem=metrics_stem)
            logging.info(
                "Rebuild metrics written to %s and %s",
                metrics_paths["json"],
                metrics_paths["csv"],
            )

    logging.info("Environmental data generation complete: %s", merged_output)
    return evs_dir


__all__ = [
    "ForbiddenEvOutputError",
    "forbidden_output_summary",
    "rebuild_environmental_layers",
    "validate_ev_output_dir",
]
