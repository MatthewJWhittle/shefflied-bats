"""Simple VOM height bands + woody cover at 100 m (#121).

One lazy tilearray mosaic + ``map_blocks`` summarise under a plan-time
soft-fail session (failed tiles → NaN; siblings keep fetching under the
shared AIMD ceiling). Neighbourhood woody cover uses a box mean on the
100 m grid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from rasterio.transform import from_origin
from tilearray import compute_with_policy
from tilearray import array as tilearray_array
from tilearray.fetch import FetchProgress

from sdm.commands.data_preparation.environmental.generate_vom_data import (
    VEGETATION_HEIGHT_MAX_M,
    cap_vegetation_heights,
)
from sdm.data.wcs_tilearray import (
    EA_VOM_COVERAGE_ID,
    EA_VOM_WCS_ENDPOINT,
    fetch_wcs_dataarray,
)
from sdm.raster.processing import calculate_multiscale_variables
from sdm.raster.vom_height_bands import (
    CELL_LAYER_STEMS,
    DEFAULT_SUMMARY_RESOLUTION_M,
    LAYER_IDS,
    pairwise_pearson,
    summarise_height_block,
)
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging

# Default abort threshold (configurable). 0.15 lets a 10% injected-failure
# proof complete while still refusing a mostly-empty mosaic.
DEFAULT_MAX_FAILED_CHUNK_FRACTION = 0.15

# EA DSP polite ceiling used by tilearray presets (8–12). Cap here too so a
# caller cannot open more parallel WCS requests than the shared fetcher allows.
EA_FETCH_CONCURRENCY_MIN_PARALLEL = 8
EA_FETCH_CONCURRENCY_MAX = 12

# 10 cells at 100 m → ``_500m`` suffix via calculate_multiscale_variables.
WOODY_NEIGHBOURHOOD_WINDOW_CELLS = 10

# Back-compat alias.
MAX_FAILED_CHUNK_FRACTION = DEFAULT_MAX_FAILED_CHUNK_FRACTION

BBox = tuple[float, float, float, float]


class VomHeightBandsChunkFailureError(RuntimeError):
    """Raised when too many chunks fail (no GeoTIFF written)."""


def resolve_vom_fetch_concurrency(max_concurrent_downloads: int) -> int:
    """Clamp WCS concurrency to the EA-safe band without collapsing to 1.

    Explicit ``1`` is allowed (serial debug). Any parallel request
    (``>= 2``) is raised to at least ``EA_FETCH_CONCURRENCY_MIN_PARALLEL``
    and capped at ``EA_FETCH_CONCURRENCY_MAX`` so ThreadPool workers are not
    serialized by a shared TileFetcher with ``max_concurrent=1``.
    """
    n = int(max_concurrent_downloads)
    if n < 1:
        raise ValueError("max_concurrent_downloads must be >= 1")
    if n == 1:
        return 1
    return max(EA_FETCH_CONCURRENCY_MIN_PARALLEL, min(n, EA_FETCH_CONCURRENCY_MAX))


@contextmanager
def tilearray_soft_fail_tiles() -> Iterator[list[str]]:
    """Install soft-fail tile loader before tilearray plans the Dask graph.

    ``create_array`` binds ``_load_tile_array`` into each delayed tile at plan
    time. Installing this wrapper only around ``compute_with_policy`` leaves
    the hard-fail loader in the graph (first error aborts siblings). Wrap
    **plan + compute** together.

    Yields a list that collects failure messages (one entry per soft-failed
    tile). ``len(failed)`` is the authoritative failed-chunk count for the
    15% abort gate.

    Interim until tilearray ships first-class soft-fail
    (https://github.com/MatthewJWhittle/tilearray/issues/65).
    """
    original_abort = FetchProgress.abort
    original_load = tilearray_array._load_tile_array
    failed: list[str] = []

    def soft_abort(self: FetchProgress, reason: str) -> None:
        with self._lock:
            self.errors.append(reason)
            # Intentionally leave ``_aborted`` false so siblings continue.

    def soft_load(
        request,
        cache_dir,
        decoder,
        dtype,
        fetch_policy=None,
        progress=None,
    ):
        try:
            return original_load(
                request, cache_dir, decoder, dtype, fetch_policy, progress
            )
        except Exception as exc:
            height = int(request.height or 0)
            width = int(request.width or 0)
            if height <= 0 or width <= 0:
                raise
            failed.append(f"{type(exc).__name__}: {exc}")
            logging.warning(
                "VOM tile soft-fail → NaN block %sx%s: %s", height, width, exc
            )
            return np.full((height, width), np.nan, dtype=dtype)

    FetchProgress.abort = soft_abort  # type: ignore[method-assign]
    tilearray_array._load_tile_array = soft_load  # type: ignore[assignment]
    try:
        yield failed
    finally:
        FetchProgress.abort = original_abort  # type: ignore[method-assign]
        tilearray_array._load_tile_array = original_load  # type: ignore[assignment]


def _summary_grid_coords(
    grid_bounds: BBox,
    resolution_m: int,
) -> tuple[np.ndarray, np.ndarray, Any, int, int]:
    """Cell-centre x/y, north-up Affine, and output height/width."""
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
    return y, x, transform, height, width


def _map_blocks_summaries_fused(
    lazy_height: xr.DataArray,
    *,
    fine_resolution_m: float,
    summary_resolution_m: int,
) -> tuple[xr.DataArray, int, dict[str, float]]:
    """One mosaic graph: map_blocks summarise + compute_with_policy.

    Caller must enter :func:`tilearray_soft_fail_tiles` so fetch failures
    become NaN tiles without aborting siblings. Returns
    ``(stacked, n_source_chunks, stage_timings)`` — failed-tile count comes
    from the soft-fail loader, not from scanning rechunked summaries.
    """
    stages: dict[str, float] = {}
    t_prep = time.perf_counter()
    if "band" in lazy_height.dims:
        lazy_height = lazy_height.isel(band=0)
    if lazy_height.dims != ("y", "x"):
        lazy_height = lazy_height.transpose("y", "x")

    data = lazy_height.data
    factor = int(round(summary_resolution_m / fine_resolution_m))
    n_layers = len(CELL_LAYER_STEMS)

    if not hasattr(data, "chunks") or data.chunks is None:
        stacked = summarise_height_block(
            np.asarray(lazy_height.values),
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
            max_height_m=VEGETATION_HEIGHT_MAX_M,
        )
        y0 = float(lazy_height.y.values[0])
        x0 = float(lazy_height.x.values[0])
        dy = (
            float(lazy_height.y.values[1] - lazy_height.y.values[0])
            if lazy_height.sizes["y"] > 1
            else -fine_resolution_m
        )
        dx = (
            float(lazy_height.x.values[1] - lazy_height.x.values[0])
            if lazy_height.sizes["x"] > 1
            else fine_resolution_m
        )
        y_c = y0 + dy * (factor / 2.0 - 0.5) + dy * factor * np.arange(
            stacked.shape[1]
        )
        x_c = x0 + dx * (factor / 2.0 - 0.5) + dx * factor * np.arange(
            stacked.shape[2]
        )
        da_out = xr.DataArray(
            stacked,
            dims=("band", "y", "x"),
            coords={"band": list(CELL_LAYER_STEMS), "y": y_c, "x": x_c},
            name="vom_height_bands",
        )
        stages["prep_s"] = time.perf_counter() - t_prep
        stages["compute_s"] = 0.0
        return da_out, 1, stages

    ny = (int(data.shape[0]) // factor) * factor
    nx = (int(data.shape[1]) // factor) * factor
    if ny == 0 or nx == 0:
        raise ValueError("VOM extent smaller than one summary cell after trim")
    trimmed = data[:ny, :nx]
    # Prefer ~1k blocks that still divide the factor (1000 = 10×100).
    block_px = factor * max(1, 1024 // factor)
    rechunked = trimmed.rechunk({0: block_px, 1: block_px})

    def _kernel(block: np.ndarray) -> np.ndarray:
        return summarise_height_block(
            block,
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
            max_height_m=VEGETATION_HEIGHT_MAX_M,
        )

    in_chy = tuple(int(c) for c in rechunked.chunks[0])
    in_chx = tuple(int(c) for c in rechunked.chunks[1])
    out_chy = tuple(c // factor for c in in_chy)
    out_chx = tuple(c // factor for c in in_chx)
    n_chunks = int(np.prod([len(c) for c in trimmed.chunks]))
    stages["prep_s"] = time.perf_counter() - t_prep

    t_compute = time.perf_counter()
    delayed = da.map_blocks(
        _kernel,
        rechunked,
        dtype=np.float32,
        chunks=((n_layers,), out_chy, out_chx),
        new_axis=0,
    )
    computed = compute_with_policy(delayed)
    stages["compute_s"] = time.perf_counter() - t_compute

    y_fine = np.asarray(lazy_height.y.values[:ny], dtype=np.float64)
    x_fine = np.asarray(lazy_height.x.values[:nx], dtype=np.float64)
    y_c = y_fine.reshape(-1, factor).mean(axis=1)
    x_c = x_fine.reshape(-1, factor).mean(axis=1)
    da_out = xr.DataArray(
        np.asarray(computed, dtype=np.float32),
        dims=("band", "y", "x"),
        coords={"band": list(CELL_LAYER_STEMS), "y": y_c, "x": x_c},
        name="vom_height_bands",
    )
    return da_out, n_chunks, stages


def _add_woody_neighbourhood(cell_ds: xr.Dataset) -> xr.Dataset:
    """Box-mean woody cover over ``WOODY_NEIGHBOURHOOD_WINDOW_CELLS`` (→ ``_500m``)."""
    woody = cell_ds[["woody_cover_share"]]
    neighbourhood = calculate_multiscale_variables(
        woody, WOODY_NEIGHBOURHOOD_WINDOW_CELLS
    )
    merged = xr.merge([cell_ds, neighbourhood])
    for stem in merged.data_vars:
        merged[stem].attrs["long_name"] = str(stem)
        merged[stem].rio.write_nodata(np.nan, inplace=True)
    return merged


def _dataset_from_stacked(
    stacked: xr.DataArray,
    *,
    crs,
    transform,
) -> xr.Dataset:
    data_vars = {
        str(stem): stacked.sel(band=stem).drop_vars("band")
        for stem in CELL_LAYER_STEMS
    }
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    for stem in ds.data_vars:
        ds[stem].attrs["long_name"] = stem
        ds[stem].rio.write_nodata(np.nan, inplace=True)
    return ds


async def fetch_and_process_vom_height_bands(
    output_dir: Path,
    boundary_path: Path,
    *,
    buffer_distance: float,
    wcs_tile_pixels: tuple[int, int],
    wcs_download_resolution_m: float,
    max_concurrent_downloads: int,
    summary_target_resolution_m: int,
    max_failed_chunk_fraction: float = DEFAULT_MAX_FAILED_CHUNK_FRACTION,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Fetch 1 m VOM once-per-tile and write height-band / woody-cover GeoTIFF.

    Persistent WCS disk cache is off by default (full-area would otherwise
    write tens of GB under the output dir). Pass ``cache_dir`` explicitly to
    enable a bounded external cache.
    """
    boundary_gdf, _model_transform, grid_bounds, _spatial = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance
    )
    model_crs = boundary_gdf.crs
    cache: Optional[Path] = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    concurrency = resolve_vom_fetch_concurrency(max_concurrent_downloads)

    logging.info(
        "Downloading VOM height bands via tilearray at %sm (concurrency=%s, "
        "cache=%s)...",
        wcs_download_resolution_m,
        concurrency,
        cache if cache is not None else "off",
    )
    t0 = time.perf_counter()
    y_coords, x_coords, transform, _out_h, _out_w = _summary_grid_coords(
        grid_bounds, summary_target_resolution_m
    )

    def _plan_and_summarise() -> tuple[
        xr.DataArray, int, int, dict[str, float], float
    ]:
        t_plan = time.perf_counter()
        # Soft-fail loader must wrap plan (delayed bind) and compute.
        with tilearray_soft_fail_tiles() as soft_failed:
            lazy_height = fetch_wcs_dataarray(
                endpoint=EA_VOM_WCS_ENDPOINT,
                coverage_id=EA_VOM_COVERAGE_ID,
                bbox=grid_bounds,
                resolution_m=float(wcs_download_resolution_m),
                chunk_size=wcs_tile_pixels,
                max_concurrent=concurrency,
                cache_dir=cache,
                compute=False,
            )
            lazy_height = cap_vegetation_heights(lazy_height)
            plan_s = time.perf_counter() - t_plan
            stacked, n_chunks, stages = _map_blocks_summaries_fused(
                lazy_height,
                fine_resolution_m=float(wcs_download_resolution_m),
                summary_resolution_m=summary_target_resolution_m,
            )
            return stacked, n_chunks, len(soft_failed), stages, plan_s

    stacked, n_chunks, n_failed, stages, plan_s = await asyncio.to_thread(
        _plan_and_summarise
    )
    wall_s = time.perf_counter() - t0

    fail_frac = (n_failed / n_chunks) if n_chunks else 1.0
    if n_chunks > 0 and fail_frac > max_failed_chunk_fraction:
        raise VomHeightBandsChunkFailureError(
            f"Failed chunk fraction {n_failed}/{n_chunks} ({fail_frac:.0%}) exceeds "
            f"allowed {max_failed_chunk_fraction:.0%}; refusing to write GeoTIFF"
        )

    t_write = time.perf_counter()
    cell_ds = _dataset_from_stacked(stacked, crs=model_crs, transform=transform)
    if cell_ds.sizes.get("y") == len(y_coords) and cell_ds.sizes.get("x") == len(
        x_coords
    ):
        cell_ds = cell_ds.assign_coords(y=y_coords, x=x_coords)
        cell_ds = cell_ds.rio.write_transform(transform)
    else:
        res = float(summary_target_resolution_m)
        x_vals = np.asarray(cell_ds.x.values, dtype=np.float64)
        y_vals = np.asarray(cell_ds.y.values, dtype=np.float64)
        coarse_transform = from_origin(
            float(x_vals[0] - res / 2.0),
            float(y_vals[0] + res / 2.0),
            res,
            res,
        )
        cell_ds = cell_ds.rio.write_transform(coarse_transform)

    full_ds = _add_woody_neighbourhood(cell_ds)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vom_height_bands_{summary_target_resolution_m}m.tif"
    logging.info("Saving VOM height bands to %s", output_path)
    full_ds.rio.to_raster(output_path)
    write_s = time.perf_counter() - t_write

    corr_layers = {
        name: np.asarray(full_ds[name].values) for name in full_ds.data_vars
    }
    corr = {
        f"{a}__{b}": (None if np.isnan(r) else round(float(r), 4))
        for (a, b), r in pairwise_pearson(corr_layers).items()
    }
    tiles_per_s = (n_chunks / wall_s) if wall_s > 0 else None
    notes: dict[str, Any] = {
        "issue": 121,
        "product_review": (
            "https://github.com/MatthewJWhittle/shefflied-bats/issues/121"
            "#product-review"
        ),
        "layers": list(full_ds.data_vars),
        "layer_ids_cell": list(LAYER_IDS),
        "n_chunks": n_chunks,
        "n_failed": n_failed,
        "wall_seconds": round(wall_s, 3),
        "tiles_per_second_e2e": (
            None if tiles_per_s is None else round(tiles_per_s, 3)
        ),
        "wcs_download_resolution_m": wcs_download_resolution_m,
        "summary_resolution_m": summary_target_resolution_m,
        "pairwise_pearson": corr,
        "max_failed_chunk_fraction": max_failed_chunk_fraction,
        "max_concurrent_downloads": concurrency,
        "wcs_cache_dir": None if cache is None else str(cache),
        "wcs_cache": "off" if cache is None else "on",
        "fetch": "fused mosaic + tilearray_soft_fail_tiles (plan+compute) + map_blocks",
        "summary": "xarray.coarsen mean/std",
        "stage_seconds": {
            "plan_lazy_mosaic": round(plan_s, 3),
            "prep": round(float(stages.get("prep_s", 0.0)), 3),
            "compute_fetch_summarise": round(float(stages.get("compute_s", 0.0)), 3),
            "neighbourhood_write": round(write_s, 3),
            "wall": round(wall_s, 3),
        },
    }
    notes_path = output_dir / "vom_height_bands_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "VOM height bands done: chunks=%s failed=%s wall=%.1fs (%.2f tiles/s) "
        "concurrency=%s stages=%s",
        n_chunks,
        n_failed,
        wall_s,
        tiles_per_s or 0.0,
        concurrency,
        notes["stage_seconds"],
    )
    return output_path


def generate_vom_height_bands(
    output_dir: Path,
    boundary_path: Path,
    *,
    buffer_distance_m: float = 0,
    wcs_tile_width_px: int = 1024,
    wcs_tile_height_px: int = 1024,
    wcs_download_resolution_m: float = 1.0,
    max_concurrent_downloads: int = 8,
    summary_target_resolution_m: int = 100,
    max_failed_chunk_fraction: float = DEFAULT_MAX_FAILED_CHUNK_FRACTION,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """Download 1 m VOM and write simple height-band / woody-cover layers."""
    setup_logging(verbose=verbose)
    logging.info("Starting VOM height-band generation (#121)...")
    result = asyncio.run(
        fetch_and_process_vom_height_bands(
            output_dir=output_dir,
            boundary_path=boundary_path,
            buffer_distance=buffer_distance_m,
            wcs_tile_pixels=(wcs_tile_width_px, wcs_tile_height_px),
            wcs_download_resolution_m=wcs_download_resolution_m,
            max_concurrent_downloads=max_concurrent_downloads,
            summary_target_resolution_m=summary_target_resolution_m,
            max_failed_chunk_fraction=max_failed_chunk_fraction,
            cache_dir=cache_dir,
        )
    )
    logging.info("VOM height-band generation finished.")
    return result


__all__ = [
    "DEFAULT_MAX_FAILED_CHUNK_FRACTION",
    "EA_FETCH_CONCURRENCY_MAX",
    "EA_FETCH_CONCURRENCY_MIN_PARALLEL",
    "MAX_FAILED_CHUNK_FRACTION",
    "VomHeightBandsChunkFailureError",
    "_map_blocks_summaries_fused",
    "fetch_and_process_vom_height_bands",
    "generate_vom_height_bands",
    "resolve_vom_fetch_concurrency",
    "tilearray_soft_fail_tiles",
]
