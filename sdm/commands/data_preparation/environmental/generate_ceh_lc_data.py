"""
CEH land-cover EV generation.

Chunking rules
--------------
* Every source chunk is aligned to the output grid (default 100 m): chunk
  offsets and sizes are multiples of ``coarsen_factor`` (source_res → output_res),
  so no output cell is ever split across two chunks.
* This step emits **in-cell** class area layers only. The 500 / 1000 / 1500 m
  neighbourhood averages are produced later by the multiscale stack
  (``calculate_multiscale_variables``). No halo overlap is required here; if
  neighbourhood averages were ever moved into this path, chunks would need a
  halo of the rolling-window radius to avoid edge seams.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

from sdm.data.landcover import define_broad_habitat_categories, get_ceh_land_cover_codes_v2023
from sdm.raster.processing import (
    aggregate_categorical_rasters,
    create_binary_raster_from_category,
)
from sdm.raster.utils import construct_transform_shift_bounds
from sdm.utils.io import load_boundary, load_spatial_config
from sdm.utils.logging_utils import setup_logging

# Default window edge in source pixels. Snapped down to a multiple of the
# coarsen factor (typically 10 for 10 m → 100 m). Peak memory scales with this
# window, not AOI size.
DEFAULT_WINDOW_PIXELS = 2000


def _aligned_window_starts(length: int, window: int, coarsen_factor: int) -> Iterator[Tuple[int, int]]:
    """
    Yield (start, size) slices along one axis, aligned to the output grid.

    ``start`` and ``size`` are always multiples of ``coarsen_factor``, so chunk
    edges never fall mid-way through a coarse (e.g. 100 m) output cell. A
    requested ``window`` that is not a multiple is snapped down (e.g. 105 → 100
    when coarsen_factor=10). Remainder source pixels beyond the trimmed length
    match xarray ``coarsen(..., boundary="trim")``.
    """
    if coarsen_factor < 1:
        raise ValueError(f"coarsen_factor must be >= 1, got {coarsen_factor}")
    trimmed = length - (length % coarsen_factor)
    if trimmed <= 0:
        return
    step = max(coarsen_factor, (window // coarsen_factor) * coarsen_factor)
    start = 0
    while start < trimmed:
        size = min(step, trimmed - start)
        size = size - (size % coarsen_factor)
        if size <= 0:
            break
        if start % coarsen_factor != 0 or size % coarsen_factor != 0:
            raise RuntimeError(
                f"Chunk not aligned to output grid: start={start}, size={size}, "
                f"coarsen_factor={coarsen_factor}"
            )
        yield start, size
        start += size


def _window_slices(
    height: int, width: int, window_pixels: int, coarsen_factor: int
) -> Iterator[Tuple[slice, slice]]:
    """Yield y/x slices for chunks aligned to the coarse output grid."""
    for y0, yh in _aligned_window_starts(height, window_pixels, coarsen_factor):
        for x0, xw in _aligned_window_starts(width, window_pixels, coarsen_factor):
            yield slice(y0, y0 + yh), slice(x0, x0 + xw)


def _class_codes_and_labels() -> List[Tuple[int, str]]:
    return [(int(key), label) for key, label in get_ceh_land_cover_codes_v2023().items()]


def _coarsen_class_areas_full(
    class_raster: xr.DataArray,
    area_per_pixel: float,
    coarsen_factor: int,
) -> xr.Dataset:
    """
    Legacy path: float conversion + one full-size layer per class, then coarsen.

    Kept for exactness tests and before/after benchmarks. Do not use on large AOIs.
    """
    working = class_raster
    nodata = working.rio.nodata
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        working = working.where(working != nodata, np.nan)
        working.rio.write_nodata(np.nan, inplace=True)

    categories = [
        create_binary_raster_from_category(working, code, label, nodata_val=np.nan)
        for code, label in _class_codes_and_labels()
    ]
    lc_stack = xr.merge(categories)
    lc_stack = lc_stack * area_per_pixel
    lc_coarse = lc_stack.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").sum(
        skipna=False
    )
    return lc_coarse.astype(np.float32)


def _coarsen_window_class_areas(
    values: np.ndarray,
    nodata: Optional[float],
    codes_labels: List[Tuple[int, str]],
    area_per_pixel: float,
    coarsen_factor: int,
) -> Dict[str, np.ndarray]:
    """
    Coarsen one window using whole-number class masks (no full float stack).

    Matches the legacy path's effective behaviour: former nodata cells count as 0
    for every class (see create_binary_raster_from_category with nan nodata).
    ``nodata`` is reserved for call-site clarity; class codes never equal the
    sentinel, so integer equality is enough (legacy path also treats nodata as 0).
    """
    _ = nodata
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D window, got shape {values.shape}")

    height, width = values.shape
    out_h = height // coarsen_factor
    out_w = width // coarsen_factor
    if out_h == 0 or out_w == 0:
        return {label: np.zeros((0, 0), dtype=np.float32) for _, label in codes_labels}

    trimmed = values[: out_h * coarsen_factor, : out_w * coarsen_factor]
    blocks = trimmed.reshape(out_h, coarsen_factor, out_w, coarsen_factor)
    result: Dict[str, np.ndarray] = {}
    for code, label in codes_labels:
        # Whole-number mask → count → area. One class at a time keeps peak low.
        counts = (blocks == code).sum(axis=(1, 3), dtype=np.float64)
        result[label] = (counts * area_per_pixel).astype(np.float32)
    return result


def coarsen_land_cover_categories(
    class_raster: xr.DataArray,
    output_resolution_m: int,
    window_pixels: int = DEFAULT_WINDOW_PIXELS,
    method: str = "chunked",
) -> xr.Dataset:
    """
    Convert a categorical land-cover raster into per-class area layers at a coarser grid.

    Args:
        class_raster: 2-D (or band-squeezable) categorical DataArray in cell units.
        output_resolution_m: Target cell size in metres (must be a multiple of source).
        window_pixels: Source-pixel edge length for chunked processing.
        method: ``"chunked"`` (default) or ``"full"`` (legacy, high memory).
    """
    if "band" in class_raster.dims:
        class_raster = class_raster.squeeze("band", drop=True)
    if class_raster.ndim != 2:
        # rip open_rasterio often yields (band, y, x); take first band.
        class_raster = class_raster[0]

    original_resolution = abs(float(class_raster.rio.resolution()[0]))
    if output_resolution_m % int(original_resolution) != 0:
        raise ValueError(
            f"output_resolution_m ({output_resolution_m}) must be a multiple of "
            f"source resolution ({original_resolution})"
        )
    coarsen_factor = int(output_resolution_m / original_resolution)
    area_per_pixel = original_resolution * original_resolution

    if method == "full":
        return _coarsen_class_areas_full(class_raster, area_per_pixel, coarsen_factor)
    if method != "chunked":
        raise ValueError(f"Unknown method {method!r}; expected 'chunked' or 'full'")

    codes_labels = _class_codes_and_labels()
    nodata = class_raster.rio.nodata
    height = int(class_raster.sizes["y"])
    width = int(class_raster.sizes["x"])
    out_h = height // coarsen_factor
    out_w = width // coarsen_factor

    # Pre-allocate coarse outputs (100 m grid is ~100× smaller than source).
    coarse_arrays = {
        label: np.zeros((out_h, out_w), dtype=np.float32) for _, label in codes_labels
    }

    y_name = class_raster.rio.y_dim or "y"
    x_name = class_raster.rio.x_dim or "x"

    for y_slice, x_slice in _window_slices(height, width, window_pixels, coarsen_factor):
        window_da = class_raster.isel({y_name: y_slice, x_name: x_slice})
        # Load only this window into memory (supports dask-backed or file-backed arrays).
        values = np.asanyarray(window_da.values)
        if values.ndim == 3:
            values = values[0]
        window_areas = _coarsen_window_class_areas(
            values, nodata, codes_labels, area_per_pixel, coarsen_factor
        )
        y0 = y_slice.start // coarsen_factor
        x0 = x_slice.start // coarsen_factor
        for label, arr in window_areas.items():
            yh, xw = arr.shape
            coarse_arrays[label][y0 : y0 + yh, x0 : x0 + xw] = arr

    # Build Dataset with coarsened coordinates matching xarray.coarsen(..., boundary="trim").
    y_coords = class_raster[y_name].values[: out_h * coarsen_factor]
    x_coords = class_raster[x_name].values[: out_w * coarsen_factor]
    y_coarse = y_coords.reshape(out_h, coarsen_factor).mean(axis=1)
    x_coarse = x_coords.reshape(out_w, coarsen_factor).mean(axis=1)

    data_vars = {
        label: xr.DataArray(
            arr,
            dims=(y_name, x_name),
            coords={y_name: y_coarse, x_name: x_coarse},
            name=label,
        )
        for label, arr in coarse_arrays.items()
    }
    ds = xr.Dataset(data_vars)
    if class_raster.rio.crs is not None:
        ds = ds.rio.write_crs(class_raster.rio.crs)
    # Affine from coarsened grid extent.
    transform = class_raster.rio.transform()
    coarse_transform = transform * transform.scale(coarsen_factor, coarsen_factor)
    ds = ds.rio.write_transform(coarse_transform)
    return ds


def generate_ceh_lc_data(
    output_dir: Path,
    boundary_path: Path,
    ceh_data_path: Path,
    buffer_distance_m: float = 7000,  # Match original default
    output_resolution_m: int = 100,
    verbose: bool = False,
    window_pixels: int = DEFAULT_WINDOW_PIXELS,
    coarsen_method: str = "chunked",
) -> Path:
    """
    Process CEH land cover data based on a given boundary.

    Args:
        output_dir: Directory where the output data will be saved
        boundary_path: Path to the boundary GeoJSON file
        ceh_data_path: Path to the CEH land cover data file
        buffer_distance_m: Buffer distance in meters to add around the boundary
        output_resolution_m: Target output resolution in meters for the processed land cover EV
        verbose: Enable verbose logging
        window_pixels: Source-pixel window edge for chunked class aggregation
        coarsen_method: ``"chunked"`` (default) or ``"full"`` (legacy)

    Returns:
        Path to the output file

    This function performs the following steps:
    1. Loads the boundary from the specified path
    2. Opens the CEH land cover data (windowed / chunked aggregation)
    3. Clips the data to the boundary with buffer
    4. Converts land cover classes to area layers via whole-number masks per window
    5. Aggregates habitat categories
    6. Reprojects to the model CRS and writes the output
    """
    setup_logging(verbose=verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open then clip; class aggregation runs in output-grid-aligned windows so
    # peak memory stays bounded by window size rather than full AOI × n_classes.
    logging.info("Loading CEH land cover data from %s", ceh_data_path)
    land_cover = rxr.open_rasterio(ceh_data_path)

    logging.info("Loading boundary from %s", boundary_path)
    boundary = load_boundary(
        boundary_path, buffer_distance=0, target_crs=land_cover.rio.crs
    )
    boundary["geometry"] = boundary.geometry.buffer(buffer_distance_m)

    logging.info("Clipping land cover to boundary")
    land_cover = land_cover.rio.clip_box(*boundary.total_bounds, crs=land_cover.rio.crs)

    logging.info(
        "Converting land cover to category area layers (%s, window=%d)",
        coarsen_method,
        window_pixels,
    )
    lc_coarse = coarsen_land_cover_categories(
        land_cover,
        output_resolution_m=output_resolution_m,
        window_pixels=window_pixels,
        method=coarsen_method,
    )

    logging.info("Performing feature engineering")
    broad_habitat_categories = define_broad_habitat_categories()
    categories_to_drop = ["Inland rock", "Marine, Littoral", "Freshwater"]
    lc_processed = aggregate_categorical_rasters(
        lc_coarse,
        aggregation_map=broad_habitat_categories,
        categories_to_drop=categories_to_drop,
    )

    spatial_config = load_spatial_config()
    model_transform, _bounds = construct_transform_shift_bounds(
        tuple(boundary.to_crs(spatial_config["crs"]).total_bounds), output_resolution_m
    )

    lc_projected = lc_processed.rio.reproject(
        spatial_config["crs"],
        transform=model_transform,
        resampling=Resampling.nearest,
    )

    output_path = output_dir / f"ceh-land-cover-{output_resolution_m}m.tif"
    logging.info("Writing data to %s", output_path)
    lc_projected.rio.to_raster(output_path)

    return output_path
