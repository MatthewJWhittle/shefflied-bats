"""Stage-one VOM canopy component layers (1 m EA VOM only; scratch output).

Opt-in rebuild step ``vom_canopy``: download VOM at 1 m tile-by-tile, build
canopy / woodland / linear / height-band masks, summarise to 100 m, then add
neighbourhood means via the shared multiscale helper. Not wired into the live
EV merge or model configs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import aiohttp
import numpy as np
import rioxarray  # noqa: F401  # registers .rio on xarray
import xarray as xr
from rasterio.transform import from_origin

from sdm.commands.data_preparation.environmental.generate_vom_data import (
    VEGETATION_HEIGHT_MAX_M,
    cap_vegetation_heights,
)
from sdm.commands.data_preparation.processing.build_model_ev_stack import (
    add_multiscale_layers,
)
from sdm.data import WCSDownloader
from sdm.raster.vom_canopy import (
    COMPONENT_LAYER_IDS,
    DEFAULT_SUMMARY_RESOLUTION_M,
    build_canopy_masks,
    compare_narrow_canopy_retention,
    summarise_masks_by_coordinates,
    summarise_masks_to_grid,
)
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging

VOM_WCS_ENDPOINT = (
    "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs"
)
VOM_COVERAGE_ID = (
    "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
)

# Neighbourhood windows in 100 m cells (500 / 1000 / 1500 m) — shared recipe.
DEFAULT_MULTISCALE_WINDOWS_CELLS: tuple[int, ...] = (10, 20, 30)
DEFAULT_MULTISCALE_GROUPS: tuple[str, ...] = ("vom_",)


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
        # floor_* are floor(coord/res); mosaic row 0 is northernmost cell.
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


async def _download_tile_array(
    downloader: WCSDownloader,
    session: aiohttp.ClientSession,
    tile_bbox: tuple[float, float, float, float],
    resolution: float,
) -> Optional[xr.DataArray]:
    """Download one WCS tile to a DataArray."""
    result = await downloader._download_tile(session, tile_bbox, resolution)
    if result is None:
        return None
    if isinstance(result, str):
        da = xr.open_dataarray(result, engine="rasterio")
        if "band" in da.dims:
            da = da.isel(band=0)
        return da.load()
    da = downloader._bytes_to_xarray(result)
    if "band" in da.dims:
        da = da.isel(band=0)
    return da


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
    """Download 1 m VOM tile-by-tile and write canopy component GeoTIFF."""
    boundary_gdf, _model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path, buffer_distance=buffer_distance)
    )
    model_crs = boundary_gdf.crs
    summary_res = int(summary_target_resolution_m)
    fine_res = float(wcs_download_resolution_m)

    mosaic_y, mosaic_x, mosaic_transform, minx_s, miny_s = _summary_grid_coords(
        grid_bounds, summary_res
    )
    maxy_s = float(mosaic_y[0] + summary_res / 2.0)
    mosaic = _empty_summary_arrays(len(mosaic_y), len(mosaic_x))
    del miny_s  # origin helper returns it; row mapping uses maxy_s

    downloader = WCSDownloader(
        endpoint=VOM_WCS_ENDPOINT,
        coverage_id=VOM_COVERAGE_ID,
        request_tile_pixels=wcs_tile_pixels,
        use_temp_storage=True,
    )
    tiles = downloader.tile_bboxes_for_coverage(grid_bounds, fine_res)
    logging.info(
        "VOM canopy: %d tiles at %.0f m over bounds %s → %d m summaries",
        len(tiles),
        fine_res,
        grid_bounds,
        summary_res,
    )

    compare_stats: Optional[dict[str, Any]] = None
    compare_bbox = sample_compare_bbox
    if compare_bbox is None and tiles:
        compare_bbox = tiles[0]

    downloader.temp_dir = Path(tempfile.mkdtemp())
    try:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(tiles), max_concurrent_downloads):
                batch = tiles[i : i + max_concurrent_downloads]
                arrays = await asyncio.gather(
                    *[
                        _download_tile_array(downloader, session, tile, fine_res)
                        for tile in batch
                    ]
                )
                for tile_bbox, da in zip(batch, arrays):
                    if da is None:
                        logging.warning("Skipping failed VOM tile %s", tile_bbox)
                        continue
                    try:
                        if (
                            write_resolution_compare
                            and compare_stats is None
                            and compare_bbox is not None
                            and tile_bbox == compare_bbox
                        ):
                            values = np.asarray(
                                cap_vegetation_heights(da).values, dtype=np.float64
                            )
                            if values.ndim == 3:
                                values = values[0]
                            compare_stats = dict(
                                compare_narrow_canopy_retention(
                                    values, resolution_1m=fine_res
                                )
                            )
                            compare_stats["sample_tile_bbox"] = list(tile_bbox)
                            logging.info(
                                "10 m vs 1 m narrow-canopy sample: %s", compare_stats
                            )

                        cell_values = process_height_tile(
                            da,
                            fine_resolution_m=fine_res,
                            summary_resolution_m=summary_res,
                        )
                        _accumulate_coordinate_summaries(
                            mosaic,
                            cell_values,
                            minx_s=minx_s,
                            maxy_s=maxy_s,
                            summary_resolution_m=summary_res,
                        )
                    finally:
                        da.close()
                logging.info(
                    "VOM canopy tiles %d–%d / %d",
                    i + 1,
                    min(i + max_concurrent_downloads, len(tiles)),
                    len(tiles),
                )
    finally:
        if downloader.temp_dir and downloader.temp_dir.exists():
            for path in downloader.temp_dir.iterdir():
                path.unlink(missing_ok=True)
            downloader.temp_dir.rmdir()

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

    notes = {
        "source": "EA Lidar Vegetation Object Model (VOM) only",
        "wcs_endpoint": VOM_WCS_ENDPOINT,
        "coverage_id": VOM_COVERAGE_ID,
        "download_resolution_m": fine_res,
        "summary_resolution_m": summary_res,
        "canopy_min_height_m": 2.5,
        "height_cap_m": VEGETATION_HEIGHT_MAX_M,
        "woodland_erosion_m": 12.0,
        "woodland_min_area_m2": 5000.0,
        "multiscale_windows_cells": list(multiscale_windows_cells),
        "n_tiles": len(tiles),
        "spatial_resolution_model_m": spatial_config.get("resolution"),
        "component_layers": list(COMPONENT_LAYER_IDS),
        "not_in_model_stack": True,
        "resolution_compare_10m_vs_1m": compare_stats,
    }
    notes_path = output_dir / "vom_canopy_features_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Wrote processing notes to %s", notes_path)
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
