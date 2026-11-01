"""Stage-one VOM canopy component layers (1 m EA VOM only; scratch output).

Opt-in rebuild step ``vom_canopy``: one lazy tilearray WCS mosaic; per-chunk
masks under Dask overlap (halos from neighbouring chunks — each source tile
fetched once) that reduce to 100 m float32 summaries before gather;
neighbourhood means. Not wired into the live EV merge or model configs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

import dask
import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from dask.array.overlap import ensure_minimum_chunksize, overlap
from rasterio.transform import from_origin

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
    WOODLAND_CLOSE_WINDOW_M,
    WOODLAND_MIN_FRAC,
    WOODLAND_WINDOW_M,
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

# Abort before writing outputs when this fraction of chunks fail (e.g. 403s).
MAX_FAILED_CHUNK_FRACTION = 0.05

# Back-compat aliases for notes / callers.
VOM_WCS_ENDPOINT = EA_VOM_WCS_ENDPOINT
VOM_COVERAGE_ID = EA_VOM_COVERAGE_ID


class VomCanopyChunkFailureError(RuntimeError):
    """Raised when too many canopy chunks fail (no output written)."""


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
        name: np.full((height, width), np.nan, dtype=np.float32)
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


def _cap_height_block(block: np.ndarray) -> np.ndarray:
    """Apply the shared vegetation height cap on a numpy block (float32)."""
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    out = arr.copy()
    out[finite & (out > VEGETATION_HEIGHT_MAX_M)] = np.float32(
        VEGETATION_HEIGHT_MAX_M
    )
    return out


def _trim_masks_to_interior(
    masks: CanopyMasks, halo_px: int
) -> CanopyMasks:
    """Drop the overlap halo so summaries cover only the source chunk interior."""
    sl = (slice(halo_px, -halo_px), slice(halo_px, -halo_px))
    return CanopyMasks(
        canopy=np.asarray(masks.canopy[sl], dtype=bool),
        band_2p5_5=np.asarray(masks.band_2p5_5[sl], dtype=bool),
        band_5_15=np.asarray(masks.band_5_15[sl], dtype=bool),
        band_gt15=np.asarray(masks.band_gt15[sl], dtype=bool),
        woodland=np.asarray(masks.woodland[sl], dtype=bool),
        trees_outside_woodland=np.asarray(
            masks.trees_outside_woodland[sl], dtype=bool
        ),
        linear=np.asarray(masks.linear[sl], dtype=bool),
        canopy_edge=np.asarray(masks.canopy_edge[sl], dtype=bool),
    )


def _process_halo_chunk(
    block: np.ndarray,
    y_interior: np.ndarray,
    x_interior: np.ndarray,
    *,
    halo_px: int,
    fine_resolution_m: float,
    summary_resolution_m: int,
    want_compare: bool,
) -> tuple[dict[tuple[int, int], dict[str, float]], int, Optional[dict[str, Any]]]:
    """Masks + 100 m summary for one overlapped chunk; catch failures per chunk.

    Returns ``(cell_values, failed_flag, compare_stats)``. On failure the cell
    dict is empty (caller leaves mosaic NaN) and ``failed_flag`` is 1.
    """
    try:
        capped = _cap_height_block(block)
        masks = build_canopy_masks(capped, resolution_m=fine_resolution_m)
        interior_masks = _trim_masks_to_interior(masks, halo_px)
        interior_h = capped[halo_px:-halo_px, halo_px:-halo_px]
        if interior_h.shape != (len(y_interior), len(x_interior)):
            raise ValueError(
                f"halo interior shape {interior_h.shape} != coords "
                f"({len(y_interior)}, {len(x_interior)})"
            )
        cells = summarise_masks_by_coordinates(
            interior_masks,
            interior_h,
            x_coords=np.asarray(x_interior, dtype=np.float64),
            y_coords=np.asarray(y_interior, dtype=np.float64),
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
        )
        # float32 mosaic: downcast summary scalars
        for key, vals in cells.items():
            for name, value in list(vals.items()):
                vals[name] = float(np.float32(value))
        compare_stats = None
        if want_compare:
            compare_stats = dict(
                compare_narrow_canopy_retention(
                    interior_h, resolution_1m=fine_resolution_m
                )
            )
            compare_stats["sample_chunk"] = (
                f"y{len(y_interior)}_x{len(x_interior)}"
            )
        return cells, 0, compare_stats
    except Exception:
        logging.exception(
            "VOM canopy chunk failed (interior y=%d x=%d); writing NaNs",
            len(y_interior),
            len(x_interior),
        )
        return {}, 1, None


def _num_workers_for_lazy(lazy_height: xr.DataArray) -> int:
    """Thread workers aligned with tilearray's recommended fetch pool."""
    attrs = getattr(lazy_height, "attrs", {}) or {}
    recommended = attrs.get("tilearray_recommended_num_workers")
    if recommended is not None:
        return max(1, int(recommended))
    policy = attrs.get("tilearray_fetch_policy") or {}
    if isinstance(policy, dict) and policy.get("max_concurrent") is not None:
        return max(1, int(policy["max_concurrent"]))
    return 8


def _rechunk_for_overlap_halo(data: Any, halo_px: int) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    """Match ``overlap``'s ``ensure_minimum_chunksize`` so block grids agree.

    Dask merges trailing slivers thinner than ``halo_px`` before building
    overlapped blocks; iterating the pre-overlap chunk grid then IndexErrors.
    """
    halo = int(halo_px)
    chunks0 = tuple(int(c) for c in data.chunks[0])
    chunks1 = tuple(int(c) for c in data.chunks[1])
    ensured0 = ensure_minimum_chunksize(halo, chunks0)
    ensured1 = ensure_minimum_chunksize(halo, chunks1)
    if (ensured0, ensured1) != (chunks0, chunks1):
        logging.info(
            "VOM canopy rechunk for halo=%d: y %s → %s; x %s → %s",
            halo,
            chunks0,
            ensured0,
            chunks1,
            ensured1,
        )
        data = data.rechunk((ensured0, ensured1))
    return data, ensured0, ensured1


def _compute_chunk_task(task: Any) -> tuple[
    dict[tuple[int, int], dict[str, float]], int, Optional[dict[str, Any]]
]:
    """Compute one chunk task; catch fetch/process failures as a failed chunk."""
    try:
        return dask.compute(task, scheduler="threads", num_workers=1)[0]
    except Exception:
        logging.exception(
            "VOM canopy chunk fetch/compute failed; writing NaNs for chunk"
        )
        return {}, 1, None


def _process_lazy_canopy_map_overlap(
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
    """Per-chunk overlap → masks → 100 m reduce; gather only summary cells.

    Each source tile is still fetched once (shared overlap graph). Fine-resolution
    height/masks never materialise for the whole AOI — only float32 100 m
    summaries are accumulated. Per-chunk fetch/process exceptions become NaN
    cells + a failure count (abort threshold applied by the caller).

    Returns ``(n_chunks, n_failed, compare_stats)``.
    """
    if "band" in lazy_height.dims:
        lazy_height = lazy_height.isel(band=0)
    if lazy_height.dims != ("y", "x"):
        lazy_height = lazy_height.transpose("y", "x")

    data = lazy_height.data
    if not hasattr(data, "chunks") or data.chunks is None:
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
                cap_vegetation_heights(lazy_height).values, dtype=np.float32
            )
            compare_stats = dict(
                compare_narrow_canopy_retention(
                    values, resolution_1m=fine_resolution_m
                )
            )
        return 1, 0, compare_stats

    y_coords = np.asarray(lazy_height.y.values)
    x_coords = np.asarray(lazy_height.x.values)

    # float32 source; rechunk so overlap's merged sliver grid matches our loop.
    if getattr(data, "dtype", None) != np.float32:
        data = data.astype(np.float32)
    data, chy, chx = _rechunk_for_overlap_halo(data, int(halo_px))
    n_chunks = len(chy) * len(chx)

    overlapped = overlap(
        data, depth={0: int(halo_px), 1: int(halo_px)}, boundary=0
    )
    delayed_blocks = overlapped.to_delayed()
    if delayed_blocks.shape != (len(chy), len(chx)):
        raise RuntimeError(
            "VOM canopy overlap block grid mismatch: delayed "
            f"{delayed_blocks.shape} vs chunks ({len(chy)}, {len(chx)})"
        )

    tasks = []
    y0 = 0
    for iy, chy_i in enumerate(chy):
        x0 = 0
        for ix, chx_i in enumerate(chx):
            want_compare = bool(
                write_resolution_compare and iy == 0 and ix == 0
            )
            tasks.append(
                dask.delayed(_process_halo_chunk)(
                    delayed_blocks[iy, ix],
                    y_coords[y0 : y0 + chy_i],
                    x_coords[x0 : x0 + chx_i],
                    halo_px=int(halo_px),
                    fine_resolution_m=float(fine_resolution_m),
                    summary_resolution_m=int(summary_resolution_m),
                    want_compare=want_compare,
                )
            )
            x0 += chx_i
        y0 += chy_i

    num_workers = _num_workers_for_lazy(lazy_height)
    # Per-task compute so one failed fetch/process does not abort siblings.
    results: list[
        tuple[dict[tuple[int, int], dict[str, float]], int, Optional[dict[str, Any]]]
    ] = [({}, 1, None)] * len(tasks)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_map = {
            pool.submit(_compute_chunk_task, task): i
            for i, task in enumerate(tasks)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                logging.exception(
                    "VOM canopy chunk worker crashed; writing NaNs for chunk"
                )
                results[idx] = ({}, 1, None)

    n_failed = 0
    compare_stats: Optional[dict[str, Any]] = None
    for cells, failed_flag, chunk_compare in results:
        n_failed += int(failed_flag)
        if chunk_compare is not None and compare_stats is None:
            compare_stats = chunk_compare
            logging.info(
                "10 m vs 1 m narrow-canopy sample: %s", compare_stats
            )
        if not cells:
            continue
        _accumulate_coordinate_summaries(
            mosaic,
            cells,
            minx_s=minx_s,
            maxy_s=maxy_s,
            summary_resolution_m=summary_resolution_m,
        )

    return n_chunks, n_failed, compare_stats


async def fetch_and_process_vom_canopy(
    output_dir: Path,
    boundary_path: Path,
    *,
    buffer_distance: float = 0.0,
    wcs_tile_pixels: tuple[int, int] = (1024, 1024),
    wcs_download_resolution_m: float = 1.0,
    summary_target_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    max_concurrent_downloads: int = 8,
    multiscale_windows_cells: Sequence[int] = DEFAULT_MULTISCALE_WINDOWS_CELLS,
    sample_compare_bbox: Optional[tuple[float, float, float, float]] = None,
    write_resolution_compare: bool = True,
    max_failed_chunk_fraction: float = MAX_FAILED_CHUNK_FRACTION,
) -> Path:
    """Lazy tilearray VOM fetch + map_overlap canopy components → GeoTIFF."""
    del sample_compare_bbox  # compare uses first chunk; bbox kept for API

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

    # Disk tile cache: per-chunk compute isolates fetch failures without
    # re-downloading shared halo tiles (tilearray aborts sibling fetches in-session).
    wcs_cache_dir = output_dir / ".wcs_tile_cache"
    wcs_cache_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "VOM canopy: tilearray lazy fetch at %.0f m chunks=%s halo=%d px "
        "concurrency=%d over %s (cache=%s)",
        fine_res,
        wcs_tile_pixels,
        halo_px,
        max_concurrent_downloads,
        grid_bounds,
        wcs_cache_dir,
    )
    lazy = await asyncio.to_thread(
        fetch_wcs_dataarray,
        endpoint=EA_VOM_WCS_ENDPOINT,
        coverage_id=EA_VOM_COVERAGE_ID,
        bbox=grid_bounds,
        resolution_m=fine_res,
        chunk_size=wcs_tile_pixels,
        max_concurrent=max_concurrent_downloads,
        cache_dir=wcs_cache_dir,
        compute=False,
    )

    n_chunks, n_failed, compare_stats = await asyncio.to_thread(
        _process_lazy_canopy_map_overlap,
        lazy,
        mosaic,
        fine_resolution_m=fine_res,
        summary_resolution_m=summary_res,
        minx_s=minx_s,
        maxy_s=maxy_s,
        halo_px=halo_px,
        write_resolution_compare=write_resolution_compare,
    )

    n_ok = n_chunks - n_failed
    fail_frac = (n_failed / n_chunks) if n_chunks else 1.0
    if fail_frac > max_failed_chunk_fraction:
        raise VomCanopyChunkFailureError(
            f"VOM canopy abort: {n_failed}/{n_chunks} chunks failed "
            f"({fail_frac:.0%} > allowed {max_failed_chunk_fraction:.0%}); "
            "no output written"
        )

    data_vars = {}
    for name, arr in mosaic.items():
        da_out = xr.DataArray(
            arr, coords={"y": mosaic_y, "x": mosaic_x}, dims=("y", "x"), name=name
        )
        da_out.attrs["long_name"] = name
        data_vars[name] = da_out
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
        "fetch": (
            "tilearray.create_array + dask overlap per-chunk "
            "mask+100m-reduce (once-fetch shared tiles)"
        ),
        "wcs_endpoint": EA_VOM_WCS_ENDPOINT,
        "coverage_id": EA_VOM_COVERAGE_ID,
        "download_resolution_m": fine_res,
        "summary_resolution_m": summary_res,
        "summary_dtype": "float32",
        "canopy_min_height_m": 2.5,
        "height_cap_m": VEGETATION_HEIGHT_MAX_M,
        "woodland_method": "box_mean_core_plus_binary_propagation",
        "woodland_close_window_m": WOODLAND_CLOSE_WINDOW_M,
        "woodland_window_m": WOODLAND_WINDOW_M,
        "woodland_min_frac": WOODLAND_MIN_FRAC,
        "linear_method": "box_mean_narrow_plus_directional_density",
        "mask_overlap_halo_px": halo_px,
        "mask_overlap_halo_note": (
            f"{halo_px} px = woodland_core_r + linear_r "
            "(box/uniform path; earlier opening/EDT path used ~45 px)"
        ),
        "max_concurrent_downloads": max_concurrent_downloads,
        "multiscale_windows_cells": list(multiscale_windows_cells),
        "n_chunks": n_chunks,
        "n_chunks_processed": n_ok,
        "n_chunks_failed": n_failed,
        "max_failed_chunk_fraction": max_failed_chunk_fraction,
        "spatial_resolution_model_m": spatial_config.get("resolution"),
        "component_layers": list(COMPONENT_LAYER_IDS),
        "not_in_model_stack": True,
        "resolution_compare_10m_vs_1m": compare_stats,
        "canopy_edge_connectivity": "8-connected (3x3 box-mean < 1)",
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
    max_concurrent_downloads: int = 8,
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
    "MAX_FAILED_CHUNK_FRACTION",
    "VOM_COVERAGE_ID",
    "VOM_WCS_ENDPOINT",
    "VomCanopyChunkFailureError",
    "fetch_and_process_vom_canopy",
    "generate_vom_canopy_features",
    "process_height_array_block_aligned",
    "process_height_tile",
]
