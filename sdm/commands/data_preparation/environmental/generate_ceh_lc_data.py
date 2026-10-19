"""
CEH land-cover EV generation.

Memory-efficient class aggregation uses dask chunks with xarray
``coarsen`` (integer per-class counts). Dask chunk boundaries need not
align with the 100 m output cells — ``coarsen(..., boundary="trim")`` is
correct regardless; identity tests cover chunks that split coarse cells
after an arbitrary clip. The model 100 m grid is applied later at
reproject time.

This step emits **in-cell** class area layers only. The 500 / 1000 /
1500 m neighbourhood averages are produced later by the multiscale stack
(``calculate_multiscale_variables``). No halo is required here.
"""

import logging
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

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

# Default dask chunk edge in source pixels. Peak memory scales with this
# (and n_classes during compute), not AOI size.
DEFAULT_CHUNK_PIXELS = 2000
DEFAULT_CHUNKS: Mapping[str, int] = {
    "band": 1,
    "y": DEFAULT_CHUNK_PIXELS,
    "x": DEFAULT_CHUNK_PIXELS,
}


def _class_codes_and_labels() -> List[Tuple[int, str]]:
    return [(int(key), label) for key, label in get_ceh_land_cover_codes_v2023().items()]


def _coarsen_class_areas_full(
    class_raster: xr.DataArray,
    area_per_pixel: float,
    coarsen_factor: int,
) -> xr.Dataset:
    """
    Legacy single-pass path: float conversion + one full-size layer per class.

    Kept for exactness tests and before/after benchmarks. Do not use on large AOIs.
    """
    working = class_raster
    # Materialise if dask-backed so the legacy path stays the old eager behaviour.
    if hasattr(working.data, "chunks") and working.data.chunks is not None:
        working = working.compute()
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


def _coarsen_class_areas_dask(
    class_raster: xr.DataArray,
    area_per_pixel: float,
    coarsen_factor: int,
    chunks: Mapping[str, int],
) -> xr.Dataset:
    """
    Native dask/xarray path: integer class equality + coarsen sum per class.

    Nodata never equals a class code, so it contributes 0 (same as the legacy
    path's effective behaviour with nan nodata). Chunk alignment is irrelevant
    to correctness.
    """
    da = class_raster
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    # Ensure 2-D spatial array with dask chunks on y/x.
    chunk_y = int(chunks.get("y", DEFAULT_CHUNK_PIXELS))
    chunk_x = int(chunks.get("x", DEFAULT_CHUNK_PIXELS))
    spatial_chunks = {dim: size for dim, size in (("y", chunk_y), ("x", chunk_x)) if dim in da.dims}
    if not hasattr(da.data, "chunks") or da.data.chunks is None:
        da = da.chunk(spatial_chunks)
    else:
        # Rechunk if caller passed an explicit size (e.g. tests with odd chunks).
        da = da.chunk(spatial_chunks)

    data_vars = {
        label: (
            (da == code).coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").sum()
            * area_per_pixel
        ).astype(np.float32)
        for code, label in _class_codes_and_labels()
    }
    ds = xr.Dataset(data_vars)
    # Compute once so callers get an in-memory Dataset (reproject / to_raster).
    ds = ds.compute()
    if class_raster.rio.crs is not None:
        ds = ds.rio.write_crs(class_raster.rio.crs)
    transform = class_raster.rio.transform()
    coarse_transform = transform * transform.scale(coarsen_factor, coarsen_factor)
    ds = ds.rio.write_transform(coarse_transform)
    return ds


def coarsen_land_cover_categories(
    class_raster: xr.DataArray,
    output_resolution_m: int,
    chunks: Optional[Mapping[str, int]] = None,
    method: str = "dask",
    # Deprecated alias kept so older call sites/tests keep working during transition.
    window_pixels: Optional[int] = None,
) -> xr.Dataset:
    """
    Convert a categorical land-cover raster into per-class area layers at a coarser grid.

    Args:
        class_raster: 2-D (or band-squeezable) categorical DataArray in cell units.
        output_resolution_m: Target cell size in metres (must be a multiple of source).
        chunks: Dask chunk sizes for ``y`` / ``x`` (and optional ``band``).
        method: ``"dask"`` (default, memory-efficient) or ``"full"`` (legacy).
        window_pixels: Deprecated; if set, used as square ``y``/``x`` chunk size.
    """
    if "band" in class_raster.dims and class_raster.sizes.get("band", 1) == 1:
        class_raster = class_raster.squeeze("band", drop=True)
    if class_raster.ndim != 2 and "band" in class_raster.dims:
        class_raster = class_raster.isel(band=0, drop=True)
    if class_raster.ndim != 2:
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
    if method not in ("dask", "chunked"):
        raise ValueError(f"Unknown method {method!r}; expected 'dask' or 'full'")

    effective_chunks: Mapping[str, int]
    if chunks is not None:
        effective_chunks = chunks
    elif window_pixels is not None:
        effective_chunks = {"y": int(window_pixels), "x": int(window_pixels)}
    else:
        effective_chunks = DEFAULT_CHUNKS

    return _coarsen_class_areas_dask(
        class_raster, area_per_pixel, coarsen_factor, effective_chunks
    )


def generate_ceh_lc_data(
    output_dir: Path,
    boundary_path: Path,
    ceh_data_path: Path,
    buffer_distance_m: float = 7000,  # Match original default
    output_resolution_m: int = 100,
    verbose: bool = False,
    chunks: Optional[Mapping[str, int]] = None,
    coarsen_method: str = "dask",
    window_pixels: Optional[int] = None,
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
        chunks: Dask chunk sizes when ``coarsen_method="dask"``
        coarsen_method: ``"dask"`` (default) or ``"full"`` (legacy)
        window_pixels: Deprecated alias for square chunk size

    Returns:
        Path to the output file
    """
    setup_logging(verbose=verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    open_chunks: Union[Mapping[str, int], bool]
    if coarsen_method == "full":
        open_chunks = False
    elif chunks is not None:
        open_chunks = dict(chunks)
        open_chunks.setdefault("band", 1)
    elif window_pixels is not None:
        open_chunks = {"band": 1, "y": int(window_pixels), "x": int(window_pixels)}
    else:
        open_chunks = dict(DEFAULT_CHUNKS)

    logging.info("Loading CEH land cover data from %s", ceh_data_path)
    if open_chunks is False:
        land_cover = rxr.open_rasterio(ceh_data_path)
    else:
        land_cover = rxr.open_rasterio(ceh_data_path, chunks=open_chunks)

    logging.info("Loading boundary from %s", boundary_path)
    boundary = load_boundary(
        boundary_path, buffer_distance=0, target_crs=land_cover.rio.crs
    )
    boundary["geometry"] = boundary.geometry.buffer(buffer_distance_m)

    logging.info("Clipping land cover to boundary")
    land_cover = land_cover.rio.clip_box(*boundary.total_bounds, crs=land_cover.rio.crs)
    # Real LCM 10 m has band 1 = class, band 2 = confidence. Select class
    # explicitly; squeeze fails when the confidence band is present.
    if "band" in land_cover.dims:
        if land_cover.sizes["band"] == 1:
            land_cover = land_cover.squeeze("band", drop=True)
        else:
            band_coords = land_cover.coords.get("band")
            if band_coords is not None and 1 in np.asarray(band_coords):
                land_cover = land_cover.sel(band=1, drop=True)
            else:
                land_cover = land_cover.isel(band=0, drop=True)

    logging.info(
        "Converting land cover to category area layers (%s)",
        coarsen_method,
    )
    lc_coarse = coarsen_land_cover_categories(
        land_cover,
        output_resolution_m=output_resolution_m,
        chunks=None if open_chunks is False else open_chunks,
        method=coarsen_method,
        window_pixels=window_pixels,
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
