"""Dry-run plan for ``sdm data --plan`` (no downloads, no writes)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import geopandas as gpd

from sdm.data.coastline_download import LEGACY_BGS_GEOCOAST
from sdm.data.os_download import BASE_URL as OS_DOWNLOADS_BASE_URL
from sdm.data.os_download import osgb_100km_tiles_for_bbox

DEFAULT_CEH_PATH = Path(
    "data/raw/big-files/CEH/data/"
    "7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"
)

WORLDCLIM_URL_TEMPLATE = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/"
    "GBR_wc2.1_30s_{var}.tif"
)
WORLDCLIM_URL_PATTERN = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/"
    "GBR_wc2.1_30s_{var}.tif"
)
TERRAIN_WCS_ENDPOINTS = {
    "dtm": (
        "https://environment.data.gov.uk/spatialdata/"
        "lidar-composite-digital-terrain-model-dtm-1m/wcs"
    ),
    "dsm": (
        "https://environment.data.gov.uk/spatialdata/"
        "lidar-composite-digital-surface-model-last-return-dsm-1m/wcs"
    ),
}
VOM_WCS_ENDPOINT = (
    "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs"
)
BOUNDARY_LINE_PRODUCT = "BoundaryLine (OS Downloads API)"


@dataclass(frozen=True)
class PlanStep:
    """One rebuild step and its primary inputs."""

    name: str
    source: str
    local_path: Optional[Path]
    present: Optional[bool]
    required: bool
    notes: str = ""


def _path_present(path: Optional[Path]) -> Optional[bool]:
    if path is None:
        return None
    return path.exists()


def _boundary_bbox(
    boundary_path: Path,
    *,
    buffer_m: float,
) -> tuple[float, float, float, float]:
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        raise ValueError(f"Boundary has no CRS: {boundary_path}")
    if str(boundary.crs) != "EPSG:27700":
        boundary = boundary.to_crs("EPSG:27700")
    if buffer_m:
        boundary = boundary.copy()
        boundary["geometry"] = boundary.geometry.buffer(buffer_m)
    minx, miny, maxx, maxy = boundary.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def os_tiles_for_boundary(
    boundary_path: Path,
    *,
    buffer_m: float = 7000,
) -> list[str]:
    """OS Vector Map District 100 km tiles intersecting the buffered AOI."""
    return osgb_100km_tiles_for_bbox(*_boundary_bbox(boundary_path, buffer_m=buffer_m))


def build_data_plan(
    *,
    boundary_path: Path,
    buffer_m: float = 7000,
    ceh_data_path: Path = DEFAULT_CEH_PATH,
    cache_dir: Optional[Path] = None,
    skip_ceh: bool = False,
    skip_coastal: bool = False,
) -> tuple[list[PlanStep], list[str], list[str]]:
    """Return plan steps, OS tiles, and human notes (optional BGS, etc.)."""
    cache_root = Path(cache_dir) if cache_dir is not None else Path("<output-dir>/cache")
    worldclim_cache = cache_root / "worldclim"
    os_raw = cache_root / "os-vector-map"
    boundary_line_cache = cache_root / "boundary-line"

    steps: list[PlanStep] = [
        PlanStep(
            name="generate_terrain_data",
            source="; ".join(
                f"{name}={url}" for name, url in sorted(TERRAIN_WCS_ENDPOINTS.items())
            ),
            local_path=None,
            present=None,
            required=True,
            notes="EA LIDAR Composite DTM/DSM WCS (live download)",
        ),
        PlanStep(
            name="generate_terrain_stats",
            source="local: <output-dir>/terrain/dtm_dsm_100m.tif",
            local_path=None,
            present=None,
            required=True,
            notes="Derived from terrain step output",
        ),
        PlanStep(
            name="generate_climate_data",
            source="; ".join(
                WORLDCLIM_URL_TEMPLATE.format(var=var)
                for var in ("bio", "tavg", "prec", "wind")
            ),
            local_path=worldclim_cache if cache_dir is not None else None,
            present=_path_present(worldclim_cache) if cache_dir is not None else None,
            required=True,
            notes=f"WorldClim cache under {worldclim_cache}",
        ),
    ]

    if skip_ceh:
        steps.append(
            PlanStep(
                name="generate_ceh_lc_data",
                source=f"manual GeoTIFF (skipped): {ceh_data_path}",
                local_path=Path(ceh_data_path),
                present=_path_present(Path(ceh_data_path)),
                required=False,
                notes="Skipped via --skip-ceh",
            )
        )
    else:
        steps.append(
            PlanStep(
                name="generate_ceh_lc_data",
                source=(
                    "manual CEH Land Cover Map 2023 10 m GeoTIFF "
                    f"(local path): {ceh_data_path}"
                ),
                local_path=Path(ceh_data_path),
                present=_path_present(Path(ceh_data_path)),
                required=True,
                notes="Required manual input — not downloaded by sdm data",
            )
        )

    steps.extend(
        [
            PlanStep(
                name="generate_vom_data",
                source=VOM_WCS_ENDPOINT,
                local_path=None,
                present=None,
                required=True,
                notes="EA Vegetation Object Model WCS (live download)",
            ),
        ]
    )

    if skip_coastal:
        steps.append(
            PlanStep(
                name="generate_coastal_distance",
                source=f"{BOUNDARY_LINE_PRODUCT} via {OS_DOWNLOADS_BASE_URL} (skipped)",
                local_path=boundary_line_cache if cache_dir is not None else None,
                present=(
                    _path_present(boundary_line_cache) if cache_dir is not None else None
                ),
                required=False,
                notes="Skipped via --skip-coastal",
            )
        )
    else:
        steps.append(
            PlanStep(
                name="generate_coastal_distance",
                source=f"{BOUNDARY_LINE_PRODUCT} via {OS_DOWNLOADS_BASE_URL}",
                local_path=boundary_line_cache if cache_dir is not None else None,
                present=(
                    _path_present(boundary_line_cache) if cache_dir is not None else None
                ),
                required=True,
                notes=(
                    "Primary: OS Boundary-Line MHW. "
                    f"Optional like-for-like BGS GeoCoast shapefile: {LEGACY_BGS_GEOCOAST}"
                ),
            )
        )

    tiles: list[str] = []
    boundary_present = boundary_path.exists()
    if boundary_present:
        tiles = os_tiles_for_boundary(boundary_path, buffer_m=buffer_m)

    steps.append(
        PlanStep(
            name="process_os_data",
            source=f"OS Vector Map District via {OS_DOWNLOADS_BASE_URL}/products/VectorMapDistrict/downloads",
            local_path=os_raw if cache_dir is not None else None,
            present=_path_present(os_raw) if cache_dir is not None else None,
            required=True,
            notes=(
                f"Needs OS 100 km tiles: {', '.join(tiles) if tiles else '(boundary missing — tiles unknown)'}"
            ),
        )
    )

    steps.extend(
        [
            PlanStep(
                name="merge_ev_layers",
                source="local: per-layer GeoTIFFs under <output-dir>",
                local_path=None,
                present=None,
                required=True,
                notes="Writes <output-dir>/all-evs.tif",
            ),
            PlanStep(
                name="build_model_ev_stack",
                source="local: <output-dir>/all-evs.tif + data/ev_clusters_selected.csv",
                local_path=Path("data/ev_clusters_selected.csv"),
                present=_path_present(Path("data/ev_clusters_selected.csv")),
                required=True,
                notes=(
                    "Rename, layer clipping (ON), selected subset, multiscale → "
                    "<output-dir>/evs-to-model.tif"
                ),
            ),
        ]
    )

    notes = [
        (
            "Optional BGS GeoCoast shapefile for a like-for-like coastal layer "
            f"(manual): {LEGACY_BGS_GEOCOAST}"
        ),
        "CEH LCM 2023 10 m GeoTIFF is a required manual input when CEH is not skipped.",
        f"WorldClim URL pattern: {WORLDCLIM_URL_PATTERN}",
    ]
    return steps, tiles, notes


def format_plan(
    steps: Sequence[PlanStep],
    tiles: Sequence[str],
    notes: Sequence[str],
    *,
    boundary_path: Path,
) -> str:
    lines: list[str] = [
        "sdm data — rebuild plan (no downloads, no writes)",
        f"Boundary: {boundary_path} ({'present' if boundary_path.exists() else 'MISSING'})",
        f"OS 100 km tiles: {', '.join(tiles) if tiles else '(none — fix boundary first)'}",
        "",
        "Steps:",
    ]
    for index, step in enumerate(steps, start=1):
        if step.present is True:
            presence = "present"
        elif step.present is False:
            presence = "MISSING"
        else:
            presence = "n/a (download / derived)"
        req = "required" if step.required else "optional"
        lines.append(f"  {index}. {step.name} [{req}]")
        lines.append(f"     source: {step.source}")
        if step.local_path is not None:
            lines.append(f"     local:  {step.local_path} ({presence})")
        else:
            lines.append(f"     input:  {presence}")
        if step.notes:
            lines.append(f"     notes:  {step.notes}")
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def missing_required_inputs(steps: Iterable[PlanStep]) -> list[PlanStep]:
    return [
        step
        for step in steps
        if step.required and step.local_path is not None and step.present is False
    ]


def run_data_plan(
    *,
    boundary_path: Path,
    buffer_m: float = 7000,
    ceh_data_path: Path = DEFAULT_CEH_PATH,
    cache_dir: Optional[Path] = None,
    skip_ceh: bool = False,
    skip_coastal: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Print the plan; return 0 if OK, 1 if a required manual input is missing."""
    out = stream or sys.stdout
    steps, tiles, notes = build_data_plan(
        boundary_path=boundary_path,
        buffer_m=buffer_m,
        ceh_data_path=ceh_data_path,
        cache_dir=cache_dir,
        skip_ceh=skip_ceh,
        skip_coastal=skip_coastal,
    )
    out.write(
        format_plan(steps, tiles, notes, boundary_path=boundary_path)
    )
    missing = missing_required_inputs(steps)
    if not boundary_path.exists():
        out.write("\nPlan incomplete: boundary file is missing.\n")
        return 1
    if missing:
        out.write("\nRequired manual inputs missing:\n")
        for step in missing:
            out.write(f"  - {step.name}: {step.local_path}\n")
        return 1
    return 0


__all__ = [
    "DEFAULT_CEH_PATH",
    "PlanStep",
    "build_data_plan",
    "format_plan",
    "missing_required_inputs",
    "os_tiles_for_boundary",
    "run_data_plan",
]
