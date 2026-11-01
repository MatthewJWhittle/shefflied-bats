"""Stage-one VOM canopy component layers (1 m EA VOM only; scratch output).

Opt-in rebuild step ``vom_canopy``: lazy tilearray WCS fetch, local-morphology
masks under ``map_overlap``-sized chunks with halo, summarise to 100 m, then
neighbourhood means. Not wired into the live EV merge or model configs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from rasterio.transform import from_origin
from tilearray import compute_with_policy

from sdm.commands.data_preparation.environmental.generate_vom_data import (
    VEGETATION_HEIGHT_MAX_M,
    cap_vegetation_heights,
)
from sdm.commands.data_preparation.processing.build_model_ev_stack import (
    add_multiscale_layers,
)
from sdm.data.wcs_tilearray import (
    EA_VOM_COVERAGE_ID,
    EA_VOM_WCS_ENDPOINT,
    fetch_wcs_dataarray,
)
from sdm.raster.vom_canopy import (
    COMPONENT_LAYER_IDS,
    DEFAULT_SUMMARY_RESOLUTION_M,
    LINEAR_MAX_WIDTH_M,
    WOODLAND_EROSION_M,
    CanopyMasks,
    build_canopy_masks,
    compare_narrow_canopy_retention,
    mask_overlap_halo_px,
    summarise_masks_by_coordinates,
    summarise_masks_to_grid,
)
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging

# Neighbourhood windows in 100 m cells (500 / 1000 / 1500 m) — shared recipe.
DEFAULT_MULTISCALE_WINDOWS_CELLS: tuple[int, ...] = (10, 20, 30)
DEFAULT_MULTISCALE_GROUPS: tuple[str, ...] = ("vom_",)

# Back-compat aliases for notes / callers.
VOM_WCS_ENDPOINT = EA_VOM_WCS_ENDPOINT
VOM_COVERAGE_ID = EA_VOM_COVERAGE_ID


def _summary_grid_coords(
    grid_bounds: tuple[float, float, float, float],
    resolution_m: int,
) -> tuple[np.ndarray, np.ndarray, Any, float, float]:
    """Cell-centre x/y, north-up Affine, and snapped SW origin (minx, miny)."""
    minx, miny, maxx, maxy = grid_bounds
    res = float(resolution_m)
    minx_s = np.floor(minx / res) * res
    miny_s = np.floor(miny / res) * res
    maxx_s = np.ceil(maxx / res) * res
    maxy_s = np.ceil(maxy / res) * res
    width = int(round((maxx_s - minx_s) / res))
    height = int(round((maxy_s - miny_s) / res))
    x = minx_s + res / 2.0 + res * np.arange(width)
    y = maxy_s - res / 2.0 - res * np.arange(height)
    transform = from_origin(minx_s, maxy_s, res, res)
    return y, x, transform, float(minx_s), float(miny_s)


def _empty_summary_arrays(
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    return {
        name: np.full((height, width), np.nan, dtype=np.float64)
        for name in COMPONENT_LAYER_IDS
    }


def _accumulate_coordinate_summaries(
    mosaic: dict[str, np.ndarray],
    cell_values: dict[tuple[int, int], dict[str, float]],
    *,
    minx_s: float,
    maxy_s: float,
    summary_resolution_m: int,
) -> None:
    """Scatter coordinate-keyed cell stats into the north-up mosaic arrays."""
    res = float(summary_resolution_m)
    n_rows = next(iter(mosaic.values())).shape[0]
    n_cols = next(iter(mosaic.values())).shape[1]
    for (floor_y, floor_x), values in cell_values.items():
        col = int(round(floor_x - minx_s / res))
        row = int(round((maxy_s / res - 1) - floor_y))
        if row < 0 or col < 0 or row >= n_rows or col >= n_cols:
            continue
        for name, value in values.items():
            mosaic[name][row, col] = value


def process_height_tile(
    height: xr.DataArray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    tile_id: Optional[str] = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Build masks and coordinate-keyed 100 m summaries for one height tile."""
    capped = cap_vegetation_heights(height)
    values = np.asarray(capped.values, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    masks = build_canopy_masks(values, resolution_m=fine_resolution_m)
    return summarise_masks_by_coordinates(
        masks,
        values,
        x_coords=np.asarray(capped.x.values, dtype=np.float64),
        y_coords=np.asarray(capped.y.values, dtype=np.float64),
        fine_resolution_m=fine_resolution_m,
        summary_resolution_m=summary_resolution_m,
        tile_id=tile_id,
    )


def process_height_array_block_aligned(
    height: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
) -> dict[str, np.ndarray]:
    """Block-aligned summarisation for synthetic tests (origin-agnostic arrays)."""
    masks = build_canopy_masks(height, resolution_m=fine_resolution_m)
    return summarise_masks_to_grid(
        masks,
        height,
        fine_resolution_m=fine_resolution_m,
        summary_resolution_m=summary_resolution_m,
    )


def _chunk_slices(
    size: int, chunk_sizes: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Inclusive-exclusive ``(start, stop)`` for each chunk along one axis."""
    out: list[tuple[int, int]] = []
    start = 0
    for length in chunk_sizes:
        stop = start + int(length)
        out.append((start, min(stop, size)))
        start = stop
    return out


def _process_lazy_canopy_chunks(
    lazy_height: xr.DataArray,
    mosaic: dict[str, np.ndarray],
    *,
    fine_resolution_m: float,
    summary_resolution_m: int,
    minx_s: float,
    maxy_s: float,
    halo_px: int,
    write_resolution_compare: bool,
) -> tuple[int, int, Optional[dict[str, Any]]]:
    """Compute tilearray chunks with halo; accumulate 100 m summaries.

    Returns ``(n_chunks, n_failed, compare_stats)``.
    """
    if "band" in lazy_height.dims:
        lazy_height = lazy_height.isel(band=0)
    # Ensure y, x dim order.
    if lazy_height.dims != ("y", "x"):
        lazy_height = lazy_height.transpose("y", "x")

    data = lazy_height.data
    if not hasattr(data, "chunks") or data.chunks is None:
        # Already concrete — treat as one chunk.
        cell_values = process_height_tile(
            lazy_height,
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
            tile_id="full",
        )
        _accumulate_coordinate_summaries(
            mosaic,
            cell_values,
            minx_s=minx_s,
            maxy_s=maxy_s,
            summary_resolution_m=summary_resolution_m,
        )
        compare_stats = None
        if write_resolution_compare:
            values = np.asarray(
                cap_vegetation_heights(lazy_height).values, dtype=np.float64
            )
            compare_stats = dict(
                compare_narrow_canopy_retention(
                    values, resolution_1m=fine_resolution_m
                )
            )
        return 1, 0, compare_stats

    chy = tuple(int(c) for c in data.chunks[0])
    chx = tuple(int(c) for c in data.chunks[1])
    ny = int(lazy_height.sizes["y"])
    nx = int(lazy_height.sizes["x"])
    y_slices = _chunk_slices(ny, chy)
    x_slices = _chunk_slices(nx, chx)
    n_chunks = len(y_slices) * len(x_slices)
    n_failed = 0
    compare_stats: Optional[dict[str, Any]] = None
    done = 0

    for iy0, iy1 in y_slices:
        for ix0, ix1 in x_slices:
            ya = max(0, iy0 - halo_px)
            yb = min(ny, iy1 + halo_px)
            xa = max(0, ix0 - halo_px)
            xb = min(nx, ix1 + halo_px)
            tile_id = f"y{iy0}:{iy1}_x{ix0}:{ix1}"
            try:
                expanded = lazy_height.isel(y=slice(ya, yb), x=slice(xa, xb))
                computed = compute_with_policy(expanded)
                capped = cap_vegetation_heights(computed)
                values = np.asarray(capped.values, dtype=np.float64)
                if values.ndim == 3:
                    values = values[0]
                # Masks on expanded (halo) so opening/EDT see neighbours; crop
                # to chunk interior before 100 m accumulate (no double-count).
                masks = build_canopy_masks(
                    values, resolution_m=fine_resolution_m
                )
                y_off, x_off = iy0 - ya, ix0 - xa
                y_len, x_len = iy1 - iy0, ix1 - ix0
                masks_inner = CanopyMasks(
                    canopy=masks.canopy[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    band_2p5_5=masks.band_2p5_5[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    band_5_15=masks.band_5_15[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    band_gt15=masks.band_gt15[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    woodland=masks.woodland[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    trees_outside_woodland=masks.trees_outside_woodland[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    linear=masks.linear[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                    canopy_edge=masks.canopy_edge[
                        y_off : y_off + y_len, x_off : x_off + x_len
                    ],
                )
                height_inner = values[
                    y_off : y_off + y_len, x_off : x_off + x_len
                ]
                x_inner = np.asarray(capped.x.values, dtype=np.float64)[
                    x_off : x_off + x_len
                ]
                y_inner = np.asarray(capped.y.values, dtype=np.float64)[
                    y_off : y_off + y_len
                ]
                if write_resolution_compare and compare_stats is None:
                    compare_stats = dict(
                        compare_narrow_canopy_retention(
                            height_inner, resolution_1m=fine_resolution_m
                        )
                    )
                    compare_stats["sample_chunk"] = tile_id
                    logging.info(
                        "10 m vs 1 m narrow-canopy sample: %s", compare_stats
                    )
                cell_values = summarise_masks_by_coordinates(
                    masks_inner,
                    height_inner,
                    x_coords=x_inner,
                    y_coords=y_inner,
                    fine_resolution_m=fine_resolution_m,
                    summary_resolution_m=summary_resolution_m,
                    tile_id=tile_id,
                )
                _accumulate_coordinate_summaries(
                    mosaic,
                    cell_values,
                    minx_s=minx_s,
                    maxy_s=maxy_s,
                    summary_resolution_m=summary_resolution_m,
                )
            except Exception:
                n_failed += 1
                logging.exception("VOM canopy chunk failed: %s", tile_id)
            done += 1
            if done % 4 == 0 or done == n_chunks:
                logging.info("VOM canopy chunks %d / %d", done, n_chunks)
    return n_chunks, n_failed, compare_stats


async def fetch_and_process_vom_canopy(
    output_dir: Path,
    boundary_path: Path,
    *,
    buffer_distance: float = 0.0,
    wcs_tile_pixels: tuple[int, int] = (1024, 1024),
    wcs_download_resolution_m: float = 1.0,
    summary_target_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    max_concurrent_downloads: int = 4,
    multiscale_windows_cells: Sequence[int] = DEFAULT_MULTISCALE_WINDOWS_CELLS,
    sample_compare_bbox: Optional[tuple[float, float, float, float]] = None,
    write_resolution_compare: bool = True,
) -> Path:
    """Lazy tilearray VOM fetch + chunked canopy components → GeoTIFF."""
    del sample_compare_bbox  # compare uses first successful chunk; bbox kept for API

    boundary_gdf, _model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path, buffer_distance=buffer_distance)
    )
    model_crs = boundary_gdf.crs
    summary_res = int(summary_target_resolution_m)
    fine_res = float(wcs_download_resolution_m)
    halo_px = mask_overlap_halo_px(resolution_m=fine_res)

    mosaic_y, mosaic_x, mosaic_transform, minx_s, miny_s = _summary_grid_coords(
        grid_bounds, summary_res
    )
    maxy_s = float(mosaic_y[0] + summary_res / 2.0)
    mosaic = _empty_summary_arrays(len(mosaic_y), len(mosaic_x))
    del miny_s

    logging.info(
        "VOM canopy: tilearray lazy fetch at %.0f m chunks=%s halo=%d px over %s",
        fine_res,
        wcs_tile_pixels,
        halo_px,
        grid_bounds,
    )
    lazy = await asyncio.to_thread(
        fetch_wcs_dataarray,
        endpoint=EA_VOM_WCS_ENDPOINT,
        coverage_id=EA_VOM_COVERAGE_ID,
        bbox=grid_bounds,
        resolution_m=fine_res,
        chunk_size=wcs_tile_pixels,
        max_concurrent=max_concurrent_downloads,
        compute=False,
    )

    n_chunks, n_failed, compare_stats = await asyncio.to_thread(
        _process_lazy_canopy_chunks,
        lazy,
        mosaic,
        fine_resolution_m=fine_res,
        summary_resolution_m=summary_res,
        minx_s=minx_s,
        maxy_s=maxy_s,
        halo_px=halo_px,
        write_resolution_compare=write_resolution_compare,
    )

    data_vars = {}
    for name, arr in mosaic.items():
        da = xr.DataArray(
            arr, coords={"y": mosaic_y, "x": mosaic_x}, dims=("y", "x"), name=name
        )
        da.attrs["long_name"] = name
        data_vars[name] = da
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(model_crs)
    ds = ds.rio.write_transform(mosaic_transform)
    for name in ds.data_vars:
        ds[name].rio.write_nodata(np.nan, inplace=True)

    logging.info(
        "Adding neighbourhood means for windows %s cells",
        list(multiscale_windows_cells),
    )
    ds = add_multiscale_layers(
        ds,
        groups=list(DEFAULT_MULTISCALE_GROUPS),
        windows_cells=list(multiscale_windows_cells),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vom_canopy_features_100m.tif"
    logging.info("Writing VOM canopy components to %s", output_path)
    ds.rio.to_raster(output_path)

    n_ok = n_chunks - n_failed
    if n_failed:
        logging.warning(
            "VOM canopy finished with %d failed chunk(s) of %d "
            "(%d processed); coverage gaps possible — see notes sidecar",
            n_failed,
            n_chunks,
            n_ok,
        )
    else:
        logging.info(
            "VOM canopy finished with 0 failed chunks (%d processed)", n_ok
        )

    notes = {
        "source": "EA Lidar Vegetation Object Model (VOM) only",
        "fetch": "tilearray.create_array + compute_with_policy (chunked)",
        "wcs_endpoint": EA_VOM_WCS_ENDPOINT,
        "coverage_id": EA_VOM_COVERAGE_ID,
        "download_resolution_m": fine_res,
        "summary_resolution_m": summary_res,
        "canopy_min_height_m": 2.5,
        "height_cap_m": VEGETATION_HEIGHT_MAX_M,
        "woodland_method": "euclidean_morphological_opening",
        "woodland_erosion_m": WOODLAND_EROSION_M,
        "linear_method": "local_edt_width_plus_directional_density",
        "linear_max_width_m": LINEAR_MAX_WIDTH_M,
        "mask_overlap_halo_px": halo_px,
        "multiscale_windows_cells": list(multiscale_windows_cells),
        "n_chunks": n_chunks,
        "n_chunks_processed": n_ok,
        "n_chunks_failed": n_failed,
        "spatial_resolution_model_m": spatial_config.get("resolution"),
        "component_layers": list(COMPONENT_LAYER_IDS),
        "not_in_model_stack": True,
        "resolution_compare_10m_vs_1m": compare_stats,
        "canopy_edge_connectivity": "8-connected (3x3 structure; docstring matches code)",
    }
    notes_path = output_dir / "vom_canopy_features_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True), encoding="utf-8")
    logging.info(
        "Wrote processing notes to %s (n_chunks_failed=%d)",
        notes_path,
        n_failed,
    )
    return output_path


def generate_vom_canopy_features(
    output_dir: Path,
    boundary_path: Path,
    *,
    buffer_distance_m: float = 0.0,
    wcs_tile_width_px: int = 1024,
    wcs_tile_height_px: int = 1024,
    wcs_download_resolution_m: float = 1.0,
    summary_target_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    max_concurrent_downloads: int = 4,
    verbose: bool = False,
) -> Path:
    """Public entry: build stage-one VOM canopy component layers into ``output_dir``."""
    setup_logging(verbose=verbose)
    logging.info("Starting VOM canopy feature generation (1 m source)...")
    result = asyncio.run(
        fetch_and_process_vom_canopy(
            output_dir=output_dir,
            boundary_path=boundary_path,
            buffer_distance=buffer_distance_m,
            wcs_tile_pixels=(wcs_tile_width_px, wcs_tile_height_px),
            wcs_download_resolution_m=wcs_download_resolution_m,
            summary_target_resolution_m=summary_target_resolution_m,
            max_concurrent_downloads=max_concurrent_downloads,
        )
    )
    logging.info("VOM canopy feature generation finished: %s", result)
    return result


__all__ = [
    "DEFAULT_MULTISCALE_GROUPS",
    "DEFAULT_MULTISCALE_WINDOWS_CELLS",
    "VOM_COVERAGE_ID",
    "VOM_WCS_ENDPOINT",
    "fetch_and_process_vom_canopy",
    "generate_vom_canopy_features",
    "process_height_array_block_aligned",
    "process_height_tile",
]
