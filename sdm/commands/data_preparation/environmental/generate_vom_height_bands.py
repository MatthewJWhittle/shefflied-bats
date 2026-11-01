"""Simple VOM height bands + woody cover at 100 m (#121).

One lazy tilearray mosaic; ``map_blocks`` per chunk (no halo — coarsen to
100 m is local); xarray coarsen-style block reduce; per-chunk failures become
NaN. Neighbourhood woody cover uses a box mean on the 100 m grid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from rasterio.transform import from_origin
from tilearray import compute_with_policy

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

# Abort when this fraction of chunks fail (soft-fail NaN tiles).
MAX_FAILED_CHUNK_FRACTION = 0.05

# 10 cells at 100 m → ``_500m`` suffix via calculate_multiscale_variables.
WOODY_NEIGHBOURHOOD_WINDOW_CELLS = 10


class VomHeightBandsChunkFailureError(RuntimeError):
    """Raised when too many chunks fail (no GeoTIFF written)."""


def _summary_grid_coords(
    grid_bounds: tuple[float, float, float, float],
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


def _map_blocks_summaries(
    lazy_height: xr.DataArray,
    *,
    fine_resolution_m: float,
    summary_resolution_m: int,
) -> tuple[xr.DataArray, int, int]:
    """Summarise each fine chunk once via ``map_blocks`` (no halo).

    Rechunks so every spatial block is an exact multiple of the coarsen
    factor (WCS tiles are often 1024 px; 100 m cells need multiples of 100).
    Returns ``(stacked summaries, n_chunks, n_failed)``.
    """
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
        failed = int(stacked.size > 0 and np.all(~np.isfinite(stacked)))
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
        return da_out, 1, failed

    # Trim to whole summary cells, then rechunk to multiples of ``factor`` so
    # per-block reduce does not drop cross-chunk remainders.
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

    delayed = da.map_blocks(
        _kernel,
        rechunked,
        dtype=np.float32,
        chunks=((n_layers,), out_chy, out_chx),
        new_axis=0,
    )
    # Count source WCS tiles (pre-rechunk) for tiles/s reporting.
    n_chunks = int(np.prod([len(c) for c in trimmed.chunks]))
    computed = compute_with_policy(delayed)

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

    n_failed = 0
    band0 = np.asarray(da_out.isel(band=0).values)
    row_starts = np.cumsum((0,) + out_chy[:-1])
    col_starts = np.cumsum((0,) + out_chx[:-1])
    for rs, rh in zip(row_starts, out_chy):
        for cs, cw in zip(col_starts, out_chx):
            tile = band0[rs : rs + rh, cs : cs + cw]
            if tile.size and not np.any(np.isfinite(tile)):
                n_failed += 1

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
) -> Path:
    """Fetch 1 m VOM once and write height-band / woody-cover GeoTIFF + notes."""
    boundary_gdf, _model_transform, grid_bounds, _spatial = load_boundary_and_transform(
        boundary_path, buffer_distance=buffer_distance
    )
    model_crs = boundary_gdf.crs

    logging.info(
        "Downloading VOM via tilearray at %sm (lazy; height bands)...",
        wcs_download_resolution_m,
    )
    t0 = time.perf_counter()
    lazy_height = await asyncio.to_thread(
        fetch_wcs_dataarray,
        endpoint=EA_VOM_WCS_ENDPOINT,
        coverage_id=EA_VOM_COVERAGE_ID,
        bbox=grid_bounds,
        resolution_m=float(wcs_download_resolution_m),
        chunk_size=wcs_tile_pixels,
        max_concurrent=max_concurrent_downloads,
        compute=False,
    )
    lazy_height = cap_vegetation_heights(lazy_height)

    y_coords, x_coords, transform, out_h, out_w = _summary_grid_coords(
        grid_bounds, summary_target_resolution_m
    )
    del out_h, out_w  # sizes come from the summarised array

    stacked, n_chunks, n_failed = await asyncio.to_thread(
        _map_blocks_summaries,
        lazy_height,
        fine_resolution_m=float(wcs_download_resolution_m),
        summary_resolution_m=summary_target_resolution_m,
    )
    wall_s = time.perf_counter() - t0

    if n_chunks > 0 and (n_failed / n_chunks) > MAX_FAILED_CHUNK_FRACTION:
        raise VomHeightBandsChunkFailureError(
            f"Failed chunk fraction {n_failed}/{n_chunks} exceeds "
            f"{MAX_FAILED_CHUNK_FRACTION:.0%}; refusing to write GeoTIFF"
        )

    cell_ds = _dataset_from_stacked(stacked, crs=model_crs, transform=transform)
    # Prefer the snapped model grid when shapes match; otherwise keep
    # coarsened centres from the fine mosaic.
    if cell_ds.sizes.get("y") == len(y_coords) and cell_ds.sizes.get("x") == len(
        x_coords
    ):
        cell_ds = cell_ds.assign_coords(y=y_coords, x=x_coords)
        cell_ds = cell_ds.rio.write_transform(transform)
    else:
        # Derive affine from coarsened cell centres.
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
        name: np.asarray(full_ds[name].values)
        for name in full_ds.data_vars
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
        "max_failed_chunk_fraction": MAX_FAILED_CHUNK_FRACTION,
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
        )
    )
    logging.info("VOM height-band generation finished.")
    return result


__all__ = [
    "MAX_FAILED_CHUNK_FRACTION",
    "VomHeightBandsChunkFailureError",
    "fetch_and_process_vom_height_bands",
    "generate_vom_height_bands",
    "process_height_array_block_aligned",
]
