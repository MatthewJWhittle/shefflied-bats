#!/usr/bin/env python3
"""Profile ``sdm data`` rebuild steps on public-data 10 km and 50 km squares.

Assessment helper for issue #56. Writes GeoJSON squares, synthetic CEH/coast
stand-ins, runs selected rebuild steps, and records wall time + peak RSS.

Outputs stay under a caller-chosen folder (default ``data/scaling-bench``) so
the live EV set is never overwritten. Network use is limited to public OS /
EA WCS / WorldClim / Boundary-Line sources; CEH and GeoCoast are synthetic.

Usage
-----
    uv run python scripts/profile_ev_scaling.py
    uv run python scripts/profile_ev_scaling.py --sizes 10,50 --skip-network
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

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
from sdm.commands.data_preparation.rebuild_ev_layers import rebuild_cache_layout
from sdm.commands.data_preparation.spatial.generate_coastal_distance import (
    generate_coastal_distance,
)
from sdm.utils.io import load_config
from sdm.utils.step_metrics import StepMetricsRecorder

LOGGER = logging.getLogger("profile_ev_scaling")

# Squares centred inside the current study boundary (EPSG:27700).
SQUARE_SPECS_M = {
    "10km": 10_000,
    "50km": 50_000,
}

# Provided full-rebuild baseline on the study area (one machine, ~18 min).
STUDY_AREA_BASELINE = {
    "process_os_data": {"duration_seconds": 422, "peak_rss_gb": 6.9},
    "generate_vom_data": {"duration_seconds": 235, "peak_rss_gb": 0.5},
    "generate_terrain_data": {"duration_seconds": 92, "peak_rss_gb": 0.5},
    "generate_climate_data": {"duration_seconds": 52, "peak_rss_gb": 2.5},
    "climate_stats": {"duration_seconds": 17, "peak_rss_gb": 3.0},
    "generate_coastal_distance": {
        "duration_seconds_lo": 32,
        "duration_seconds_hi": 52,
        "peak_rss_gb_lo": 0.9,
        "peak_rss_gb_hi": 1.3,
    },
    "generate_ceh_lc_data": {"duration_seconds": 1, "peak_rss_gb": 0.5},
    "merge_ev_layers": {"duration_seconds": 21, "peak_rss_gb": 5.6},
    "full_run_peak_rss_gb": 10.3,
}

# Approximate land areas used for linear-area extrapolations (km²).
AREA_KM2 = {
    "10km": 100.0,
    "50km": 2_500.0,
    "study": 14_369.0,  # measured from data/processed/boundary.geojson
    "england": 130_279.0,
    "great_britain": 209_331.0,
}


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 3)


def write_square_boundary(
    study_boundary: Path,
    side_m: int,
    output_path: Path,
) -> Path:
    """Write an axis-aligned square GeoJSON centred in the study boundary."""
    study = gpd.read_file(study_boundary)
    minx, miny, maxx, maxy = study.total_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = side_m / 2.0
    geom = box(cx - half, cy - half, cx + half, cy + half)
    gdf = gpd.GeoDataFrame({"name": [f"{side_m // 1000}km"]}, geometry=[geom], crs=study.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def write_synthetic_ceh(boundary_path: Path, output_path: Path, resolution_m: int = 10) -> Path:
    """Write a CEH-shaped categorical GeoTIFF covering the buffered square."""
    boundary = gpd.read_file(boundary_path)
    buffered = boundary.copy()
    buffered["geometry"] = buffered.geometry.buffer(7_000)
    minx, miny, maxx, maxy = buffered.total_bounds
    width = int(math.ceil((maxx - minx) / resolution_m))
    height = int(math.ceil((maxy - miny) / resolution_m))
    # Cap synthetic raster size for CI-friendly smoke; full profile uses real sizes.
    max_edge = 12_000  # 120 km at 10 m
    if width > max_edge or height > max_edge:
        raise ValueError(f"Synthetic CEH too large: {width}x{height}")

    rng = np.random.default_rng(42)
    # CEH LCM class codes 1–21 on band 1; confidence on band 2.
    classes = rng.integers(1, 22, size=(height, width), dtype=np.uint8)
    confidence = rng.integers(50, 100, size=(height, width), dtype=np.uint8)
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="uint8",
        crs="EPSG:27700",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(classes, 1)
        dst.write(confidence, 2)
        dst.set_band_description(1, "class")
        dst.set_band_description(2, "confidence")
    return output_path


def write_synthetic_coastline(boundary_path: Path, output_path: Path) -> Path:
    """Write a simple polyline coast stand-in west of the square (OSGB)."""
    boundary = gpd.read_file(boundary_path)
    minx, miny, maxx, maxy = boundary.total_bounds
    # Vertical line west of the AOI — enough for distance-to-coast maths.
    line = LineString([(minx - 5_000, miny - 2_000), (minx - 5_000, maxy + 2_000)])
    gdf = gpd.GeoDataFrame({"CODE": ["0071"]}, geometry=[line], crs=boundary.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path)
    return output_path


def profile_size(
    *,
    label: str,
    side_m: int,
    root: Path,
    study_boundary: Path,
    live_download: bool,
    skip_network: bool,
    buffer_distance_m: float = 7_000,
) -> dict:
    """Run rebuild steps for one square and return metrics payload."""
    work = root / label
    work.mkdir(parents=True, exist_ok=True)
    boundary_path = write_square_boundary(study_boundary, side_m, work / "boundary.geojson")
    caches = rebuild_cache_layout(work / "cache")
    ceh_path = write_synthetic_ceh(boundary_path, work / "synthetic" / "ceh_standin.tif")
    coast_path = write_synthetic_coastline(
        boundary_path, work / "synthetic" / "coast_standin.shp"
    )

    recorder = StepMetricsRecorder()
    resolution = int(load_config()["spatial"]["resolution"])
    notes: list[str] = []

    def _try(step: str, fn) -> None:
        try:
            recorder.run(step, fn)
        except Exception as exc:  # noqa: BLE001 — assessment continues on step failure
            LOGGER.exception("%s / %s failed", label, step)
            notes.append(f"{step}: {type(exc).__name__}: {exc}")

    if not skip_network:
        _try(
            "generate_terrain_data",
            lambda: generate_terrain_data(
                output_dir=work / "terrain",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                max_concurrent_downloads=4,
            ),
        )
        terrain_dem = work / "terrain" / "dtm_dsm_100m.tif"
        if terrain_dem.exists():
            _try(
                "generate_terrain_stats",
                lambda: generate_terrain_stats(
                    input_dem_path=terrain_dem,
                    output_path=work / "terrain_stats.tif",
                ),
            )
        _try(
            "generate_climate_data",
            lambda: generate_climate_data(
                output_dir=work / "climate",
                boundary_path=boundary_path,
                worldclim_cache_dir=caches["worldclim"],
                run_stats=True,
            ),
        )
        _try(
            "generate_vom_data",
            lambda: generate_vom_data(
                output_dir=work / "vom",
                boundary_path=boundary_path,
                buffer_distance_m=buffer_distance_m,
                max_concurrent_downloads=3,
            ),
        )
        _try(
            "process_os_data",
            lambda: process_os_data(
                output_dir=work,
                boundary_path=boundary_path,
                buffer_distance=buffer_distance_m,
                parquet_dir=caches["os_parquets"],
                raw_dir=caches["os_raw"],
                live_download=live_download,
            ),
        )
        _try(
            "generate_coastal_distance",
            lambda: generate_coastal_distance(
                boundary_path=boundary_path,
                output_dir=work,
                coastline_path=coast_path,
                live_download=False,
                cache_dir=caches["boundary_line"],
            ),
        )
    else:
        notes.append("skip_network: omitted terrain/climate/vom/os/coastal live steps")

    _try(
        "generate_ceh_lc_data",
        lambda: generate_ceh_lc_data(
            output_dir=work / "landcover",
            boundary_path=boundary_path,
            ceh_data_path=ceh_path,
        ),
    )

    # Merge / stack only when enough inputs exist; otherwise note and skip.
    merge_inputs = build_ev_dataset_inputs(work, resolution, skip_ceh=False, skip_coastal=False)
    missing = [item for item in merge_inputs if not Path(item.split("=", 1)[1]).exists()]
    if missing:
        notes.append(f"merge skipped; missing {len(missing)} inputs")
    else:
        merged_path = work / "all-evs.tif"
        _try(
            "merge_ev_layers",
            lambda: merge_ev_layers(
                dataset_inputs=merge_inputs,
                boundary_path=boundary_path,
                output_path=merged_path,
                boundary_buffer_m=buffer_distance_m,
            ),
        )
        if merged_path.exists():
            try:
                settings = build_settings_from_config(load_config().get("ev_stack"))
            except FileNotFoundError:
                settings = None

            def _stack() -> None:
                build_model_ev_stack(
                    merged_path,
                    work / "evs-to-model.tif",
                    build_settings=settings,
                )

            _try("build_model_ev_stack", _stack)

    metrics_paths = recorder.write(work, stem="scaling_metrics")
    steps = []
    for step in recorder.steps:
        steps.append(
            {
                "step": step.step,
                "duration_seconds": step.duration_seconds,
                "peak_rss_bytes": step.peak_rss_bytes,
                "peak_rss_gb": _bytes_to_gb(step.peak_rss_bytes),
                "status": step.status,
                "error": step.error,
            }
        )
    return {
        "label": label,
        "side_m": side_m,
        "area_km2": AREA_KM2[label],
        "boundary_path": str(boundary_path),
        "work_dir": str(work),
        "steps": steps,
        "notes": notes,
        "metrics_json": str(metrics_paths["json"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scaling-bench"),
        help="Working folder for squares, caches, and metrics (not the live EV set)",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=Path("data/processed/boundary.geojson"),
        help="Study boundary used to centre the squares",
    )
    parser.add_argument(
        "--sizes",
        default="10,50",
        help="Comma-separated square sizes in km (subset of 10,50)",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Only run CEH stand-in + any local-only steps (no WCS/OS/WorldClim)",
    )
    parser.add_argument(
        "--no-live-download",
        action="store_true",
        help="Pass live_download=False to OS processing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    wanted = []
    for token in args.sizes.split(","):
        token = token.strip().lower().replace("km", "")
        if not token:
            continue
        key = f"{int(token)}km"
        if key not in SQUARE_SPECS_M:
            raise SystemExit(f"Unsupported size {token!r}; use 10 and/or 50")
        wanted.append(key)

    summary_path = root / "scaling_profile_summary.json"
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}
    else:
        results = {}
    results["study_area_baseline_provided"] = STUDY_AREA_BASELINE
    results["area_km2"] = AREA_KM2
    existing = {
        run["label"]: run
        for run in results.get("runs", [])
        if isinstance(run, dict) and "label" in run
    }

    for label in wanted:
        LOGGER.info("Profiling %s square", label)
        existing[label] = profile_size(
            label=label,
            side_m=SQUARE_SPECS_M[label],
            root=root,
            study_boundary=args.boundary,
            live_download=not args.no_live_download,
            skip_network=args.skip_network,
        )

    # Stable order: 10km then 50km then any others.
    order = [name for name in ("10km", "50km") if name in existing] + sorted(
        name for name in existing if name not in {"10km", "50km"}
    )
    results["runs"] = [existing[name] for name in order]
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", summary_path)
    print(json.dumps({"summary": str(summary_path), "runs": len(results["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
