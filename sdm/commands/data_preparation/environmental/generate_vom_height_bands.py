"""Simple VOM height bands + woody cover at 100 m (#121).

Each WCS tile is fetched inside its own per-chunk task (try/except → NaN
summary + failed count) so one bad tile does not abort siblings via tilearray's
shared fetch abort. Summaries use xarray ``coarsen``. Neighbourhood woody cover
uses a box mean on the 100 m grid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from rasterio.transform import from_origin

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
    summarise_height_to_grid,
)
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging

# Default abort threshold (configurable). 0.15 lets a 10% injected-failure
# proof complete while still refusing a mostly-empty mosaic.
DEFAULT_MAX_FAILED_CHUNK_FRACTION = 0.15

# 10 cells at 100 m → ``_500m`` suffix via calculate_multiscale_variables.
WOODY_NEIGHBOURHOOD_WINDOW_CELLS = 10

# Back-compat alias.
MAX_FAILED_CHUNK_FRACTION = DEFAULT_MAX_FAILED_CHUNK_FRACTION

BBox = tuple[float, float, float, float]
FetchChunkFn = Callable[[BBox], np.ndarray]


class VomHeightBandsChunkFailureError(RuntimeError):
    """Raised when too many chunks fail (no GeoTIFF written)."""


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


def _chunk_slices(chunks: tuple[int, ...]) -> list[tuple[int, int]]:
    start = 0
    out: list[tuple[int, int]] = []
    for size in chunks:
        out.append((start, start + int(size)))
        start += int(size)
    return out


def _bbox_from_centres(
    y_centres: np.ndarray,
    x_centres: np.ndarray,
    *,
    resolution_m: float,
) -> BBox:
    """Pixel-edge bbox covering the given cell centres (OSGB-style)."""
    res = float(resolution_m)
    y = np.asarray(y_centres, dtype=np.float64)
    x = np.asarray(x_centres, dtype=np.float64)
    return (
        float(x.min() - res / 2.0),
        float(y.min() - res / 2.0),
        float(x.max() + res / 2.0),
        float(y.max() + res / 2.0),
    )


def process_height_array_block_aligned(
    height: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
) -> dict[str, np.ndarray]:
    """Block-aligned summarisation for synthetic tests."""
    return summarise_height_to_grid(
        height,
        fine_resolution_m=fine_resolution_m,
        summary_resolution_m=summary_resolution_m,
    )


def _nan_summary(
    fine_ny: int,
    fine_nx: int,
    *,
    fine_resolution_m: float,
    summary_resolution_m: int,
) -> np.ndarray:
    factor = int(round(summary_resolution_m / fine_resolution_m))
    return np.full(
        (len(CELL_LAYER_STEMS), fine_ny // factor, fine_nx // factor),
        np.nan,
        dtype=np.float32,
    )


def _default_fetch_chunk(
    bbox: BBox,
    *,
    resolution_m: float,
    chunk_size: tuple[int, int],
    max_concurrent: int,
    cache_dir: Optional[Path],
) -> np.ndarray:
    """Fetch one chunk as a numpy height array (raises on hard failure)."""
    da = fetch_wcs_dataarray(
        endpoint=EA_VOM_WCS_ENDPOINT,
        coverage_id=EA_VOM_COVERAGE_ID,
        bbox=bbox,
        resolution_m=float(resolution_m),
        chunk_size=chunk_size,
        max_concurrent=max_concurrent,
        cache_dir=cache_dir,
        compute=True,
    )
    if "band" in da.dims:
        da = da.isel(band=0)
    return np.asarray(cap_vegetation_heights(da).values, dtype=np.float64)


def summarise_chunks_resilient(
    *,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    chunks: tuple[tuple[int, ...], tuple[int, ...]],
    fine_resolution_m: float,
    summary_resolution_m: int,
    fetch_chunk: FetchChunkFn,
    max_workers: int = 8,
) -> tuple[np.ndarray, int, int]:
    """Fetch+summarise each chunk in its own task; failures → NaN + count.

    ``fetch_chunk(bbox)`` must return a 2-D height array or raise. Raises are
    caught per chunk so siblings continue. Returns
    ``(stacked [band,y,x], n_chunks, n_failed)``.
    """
    factor = int(round(summary_resolution_m / fine_resolution_m))
    y_slices = _chunk_slices(chunks[0])
    x_slices = _chunk_slices(chunks[1])
    n_chunks = len(y_slices) * len(x_slices)
    out_ny = (len(y_coords) // factor) * factor // factor
    out_nx = (len(x_coords) // factor) * factor // factor
    # Trim coords to whole summary cells for output placement.
    y_trim = np.asarray(y_coords[: out_ny * factor], dtype=np.float64)
    x_trim = np.asarray(x_coords[: out_nx * factor], dtype=np.float64)
    mosaic = np.full(
        (len(CELL_LAYER_STEMS), out_ny, out_nx), np.nan, dtype=np.float32
    )
    n_failed = 0

    def _one(
        iy: int, ix: int, ys: int, ye: int, xs: int, xe: int
    ) -> tuple[int, int, np.ndarray, bool]:
        # Clip to trimmed extent.
        ye_t = min(ye, len(y_trim))
        xe_t = min(xe, len(x_trim))
        if ys >= ye_t or xs >= xe_t:
            return iy, ix, _nan_summary(
                max(0, ye_t - ys),
                max(0, xe_t - xs),
                fine_resolution_m=fine_resolution_m,
                summary_resolution_m=summary_resolution_m,
            ), True
        bbox = _bbox_from_centres(
            y_trim[ys:ye_t],
            x_trim[xs:xe_t],
            resolution_m=fine_resolution_m,
        )
        try:
            height = fetch_chunk(bbox)
            stacked = summarise_height_block(
                height,
                fine_resolution_m=fine_resolution_m,
                summary_resolution_m=summary_resolution_m,
                max_height_m=VEGETATION_HEIGHT_MAX_M,
            )
            failed = bool(stacked.size and np.all(~np.isfinite(stacked)))
            return iy, ix, stacked, failed
        except Exception:
            logging.exception(
                "VOM height-band chunk fetch/summarise failed (iy=%s ix=%s)",
                iy,
                ix,
            )
            return (
                iy,
                ix,
                _nan_summary(
                    ye_t - ys,
                    xe_t - xs,
                    fine_resolution_m=fine_resolution_m,
                    summary_resolution_m=summary_resolution_m,
                ),
                True,
            )

    jobs = [
        (iy, ix, ys, ye, xs, xe)
        for iy, (ys, ye) in enumerate(y_slices)
        for ix, (xs, xe) in enumerate(x_slices)
    ]
    workers = max(1, min(int(max_workers), n_chunks or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one, iy, ix, ys, ye, xs, xe)
            for iy, ix, ys, ye, xs, xe in jobs
        ]
        for fut in as_completed(futures):
            iy, ix, stacked, failed = fut.result()
            if failed:
                n_failed += 1
            ys, ye = y_slices[iy]
            xs, xe = x_slices[ix]
            row0, col0 = ys // factor, xs // factor
            _, rh, rw = stacked.shape
            mosaic[:, row0 : row0 + rh, col0 : col0 + rw] = stacked

    return mosaic, n_chunks, n_failed


def _map_blocks_summaries(
    lazy_height: xr.DataArray,
    *,
    fine_resolution_m: float,
    summary_resolution_m: int,
    fetch_chunk: Optional[FetchChunkFn] = None,
    max_workers: int = 8,
) -> tuple[xr.DataArray, int, int]:
    """Summarise the AOI; network path uses resilient per-chunk fetch.

    When ``fetch_chunk`` is set (live rebuild), each chunk is fetched inside its
    own task. In-memory / test arrays (no fetch_chunk) summarise locally.
    """
    if "band" in lazy_height.dims:
        lazy_height = lazy_height.isel(band=0)
    if lazy_height.dims != ("y", "x"):
        lazy_height = lazy_height.transpose("y", "x")

    data = lazy_height.data
    factor = int(round(summary_resolution_m / fine_resolution_m))
    y_coords = np.asarray(lazy_height.y.values, dtype=np.float64)
    x_coords = np.asarray(lazy_height.x.values, dtype=np.float64)

    # Eager numpy path (unit tests / small fixtures).
    if not hasattr(data, "chunks") or data.chunks is None or fetch_chunk is None:
        stacked = summarise_height_block(
            np.asarray(lazy_height.values),
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
            max_height_m=VEGETATION_HEIGHT_MAX_M,
        )
        failed = int(stacked.size > 0 and np.all(~np.isfinite(stacked)))
        y0 = float(y_coords[0])
        x0 = float(x_coords[0])
        dy = (
            float(y_coords[1] - y_coords[0])
            if len(y_coords) > 1
            else -fine_resolution_m
        )
        dx = (
            float(x_coords[1] - x_coords[0])
            if len(x_coords) > 1
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
        return da_out, 1, failed

    chunks = (
        tuple(int(c) for c in data.chunks[0]),
        tuple(int(c) for c in data.chunks[1]),
    )
    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y_coords,
        x_coords=x_coords,
        chunks=chunks,
        fine_resolution_m=fine_resolution_m,
        summary_resolution_m=summary_resolution_m,
        fetch_chunk=fetch_chunk,
        max_workers=max_workers,
    )
    ny = mosaic.shape[1] * factor
    nx = mosaic.shape[2] * factor
    y_c = y_coords[:ny].reshape(-1, factor).mean(axis=1)
    x_c = x_coords[:nx].reshape(-1, factor).mean(axis=1)
    da_out = xr.DataArray(
        mosaic,
        dims=("band", "y", "x"),
        coords={"band": list(CELL_LAYER_STEMS), "y": y_c, "x": x_c},
        name="vom_height_bands",
    )
    return da_out, n_chunks, n_failed


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
    fetch_chunk: Optional[FetchChunkFn] = None,
) -> Path:
    """Fetch 1 m VOM once-per-tile and write height-band / woody-cover GeoTIFF."""
    boundary_gdf, _model_transform, grid_bounds, _spatial = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance
    )
    model_crs = boundary_gdf.crs
    cache = Path(cache_dir) if cache_dir is not None else output_dir / "wcs-cache"
    cache.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Planning VOM height-band tiles via tilearray at %sm...",
        wcs_download_resolution_m,
    )
    t0 = time.perf_counter()
    # Lazy mosaic for coords + chunk layout only (no full compute).
    lazy_height = await asyncio.to_thread(
        fetch_wcs_dataarray,
        endpoint=EA_VOM_WCS_ENDPOINT,
        coverage_id=EA_VOM_COVERAGE_ID,
        bbox=grid_bounds,
        resolution_m=float(wcs_download_resolution_m),
        chunk_size=wcs_tile_pixels,
        max_concurrent=max_concurrent_downloads,
        cache_dir=cache,
        compute=False,
    )
    lazy_height = cap_vegetation_heights(lazy_height)

    y_coords, x_coords, transform, _out_h, _out_w = _summary_grid_coords(
        grid_bounds, summary_target_resolution_m
    )

    if fetch_chunk is None:

        def fetch_chunk(bbox: BBox) -> np.ndarray:
            return _default_fetch_chunk(
                bbox,
                resolution_m=float(wcs_download_resolution_m),
                chunk_size=wcs_tile_pixels,
                max_concurrent=1,
                cache_dir=cache,
            )

    stacked, n_chunks, n_failed = await asyncio.to_thread(
        _map_blocks_summaries,
        lazy_height,
        fine_resolution_m=float(wcs_download_resolution_m),
        summary_resolution_m=summary_target_resolution_m,
        fetch_chunk=fetch_chunk,
        max_workers=max_concurrent_downloads,
    )
    wall_s = time.perf_counter() - t0

    fail_frac = (n_failed / n_chunks) if n_chunks else 1.0
    if n_chunks > 0 and fail_frac > max_failed_chunk_fraction:
        raise VomHeightBandsChunkFailureError(
            f"Failed chunk fraction {n_failed}/{n_chunks} ({fail_frac:.0%}) exceeds "
            f"allowed {max_failed_chunk_fraction:.0%}; refusing to write GeoTIFF"
        )

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
        "fetch": "per-chunk fetch_wcs_dataarray (soft-fail → NaN)",
        "summary": "xarray.coarsen mean/std",
    }
    notes_path = output_dir / "vom_height_bands_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "VOM height bands done: chunks=%s failed=%s wall=%.1fs (%.2f tiles/s)",
        n_chunks,
        n_failed,
        wall_s,
        tiles_per_s or 0.0,
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
    fetch_chunk: Optional[FetchChunkFn] = None,
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
            fetch_chunk=fetch_chunk,
        )
    )
    logging.info("VOM height-band generation finished.")
    return result


__all__ = [
    "DEFAULT_MAX_FAILED_CHUNK_FRACTION",
    "MAX_FAILED_CHUNK_FRACTION",
    "VomHeightBandsChunkFailureError",
    "fetch_and_process_vom_height_bands",
    "generate_vom_height_bands",
    "process_height_array_block_aligned",
    "summarise_chunks_resilient",
]
