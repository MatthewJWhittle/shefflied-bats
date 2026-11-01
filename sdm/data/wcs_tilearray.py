"""Defra/EA WCS mosaics via tilearray (Dask + adaptive fetch).

Replaces bespoke ``WCSDownloader.get_coverage`` mosaics for whole-AOI
downloads. Prefer :func:`compute_with_policy` so AIMD concurrency and the
403 circuit breaker stay aligned with Dask workers.

Per-tile walkers that still need ``tile_bboxes_for_coverage`` /
``_download_tile`` keep using ``WCSDownloader`` until those paths move to
``map_overlap``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import xarray as xr
from tilearray import WCSConfig, compute_with_policy, create_array
from tilearray.fetch_presets import ea_dsp_fetch_defaults
from tilearray.types import CRS, Format

logger = logging.getLogger(__name__)

# EA Data Services Platform WCS endpoints / coverage IDs used by the EV pipeline.
EA_DTM_WCS_ENDPOINT = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-terrain-model-dtm-1m/wcs"
)
EA_DTM_COVERAGE_ID = (
    "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
)
EA_DSM_WCS_ENDPOINT = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-surface-model-last-return-dsm-1m/wcs"
)
EA_DSM_COVERAGE_ID = (
    "9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m"
)
EA_VOM_WCS_ENDPOINT = (
    "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs"
)
EA_VOM_COVERAGE_ID = (
    "ecae3bef-1e1d-4051-887b-9dc613c928ec:Vegetation_Object_Model_Elevation_2022"
)

BBox = Tuple[float, float, float, float]


def _clamp_concurrency(max_concurrent: int) -> dict[str, int]:
    """Keep AIMD ``min ≤ initial ≤ max`` when the caller caps concurrency.

    EA DSP presets start at initial=8 / min=4 / max=10. Passing
    ``max_concurrent_requests=4`` alone trips tilearray's adaptive gate
    (``initial cannot exceed maximum``).
    """
    maximum = max(1, int(max_concurrent))
    defaults = ea_dsp_fetch_defaults()
    initial = min(int(defaults["initial_concurrent_requests"]), maximum)
    minimum = min(int(defaults["min_concurrent_requests"]), initial)
    return {
        "max_concurrent_requests": maximum,
        "initial_concurrent_requests": initial,
        "min_concurrent_requests": minimum,
    }


def ea_wcs_config(
    endpoint: str,
    coverage_id: str,
    *,
    chunk_size: Tuple[int, int] = (1024, 1024),
    resolution_m: Optional[float] = None,
    max_concurrent: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> WCSConfig:
    """``WCSConfig.for_ea_dsp`` with chunk/resolution overrides for this repo.

    ``resolution`` is always a ``(x, y)`` metres-per-pixel tuple (tilearray
    pydantic schema). Concurrent caps are clamped so AIMD stays valid.
    """
    kwargs: dict = {
        "crs": CRS.EPSG_27700,
        "output_format": Format.GEOTIFF,
        "chunk_size": chunk_size,
    }
    if resolution_m is not None:
        res = float(resolution_m)
        kwargs["resolution"] = (res, res)
    if max_concurrent is not None:
        kwargs.update(_clamp_concurrency(max_concurrent))
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return WCSConfig.for_ea_dsp(endpoint, coverage_id=coverage_id, **kwargs)


def fetch_wcs_dataarray(
    *,
    endpoint: str,
    coverage_id: str,
    bbox: BBox,
    resolution_m: float,
    chunk_size: Tuple[int, int] = (1024, 1024),
    max_concurrent: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    compute: bool = True,
) -> xr.DataArray:
    """Lazy (or computed) WCS mosaic as a single DataArray.

    When ``compute=False``, callers should use :func:`compute_with_policy`
    (or pass the array through dask ``map_overlap``) so 1 m rasters are never
    held whole unless requested.
    """
    config = ea_wcs_config(
        endpoint,
        coverage_id,
        chunk_size=chunk_size,
        resolution_m=resolution_m,
        max_concurrent=max_concurrent,
        cache_dir=cache_dir,
    )
    logger.info(
        "tilearray WCS create_array coverage=%s res=%.3fm chunks=%s compute=%s",
        coverage_id,
        resolution_m,
        chunk_size,
        compute,
    )
    da = create_array(
        config,
        bbox=bbox,
        crs=CRS.EPSG_27700,
        compute=False,
        dtype="float32",
    )
    if da.name is None:
        da = da.rename(coverage_id)
    if compute:
        da = compute_with_policy(da)
    return da


def fetch_wcs_dataset(
    *,
    endpoint: str,
    coverage_id: str,
    bbox: BBox,
    resolution_m: float,
    chunk_size: Tuple[int, int] = (1024, 1024),
    max_concurrent: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    compute: bool = True,
) -> xr.Dataset:
    """WCS mosaic as a one-variable Dataset named ``coverage_id``.

    Matches the shape previously returned by ``WCSDownloader.get_coverage``.
    """
    da = fetch_wcs_dataarray(
        endpoint=endpoint,
        coverage_id=coverage_id,
        bbox=bbox,
        resolution_m=resolution_m,
        chunk_size=chunk_size,
        max_concurrent=max_concurrent,
        cache_dir=cache_dir,
        compute=compute,
    )
    if da.name != coverage_id:
        da = da.rename(coverage_id)
    return da.to_dataset(name=coverage_id)


def fetch_ea_terrain_layers(
    bbox: BBox,
    *,
    resolution_m: float,
    chunk_size: Tuple[int, int] = (1024, 1024),
    max_concurrent: Optional[int] = None,
    layer_names: Sequence[str] = ("dtm", "dsm"),
    compute: bool = True,
) -> dict[str, xr.Dataset]:
    """Download EA DTM/DSM mosaics keyed by short layer name."""
    specs = {
        "dtm": (EA_DTM_WCS_ENDPOINT, EA_DTM_COVERAGE_ID),
        "dsm": (EA_DSM_WCS_ENDPOINT, EA_DSM_COVERAGE_ID),
    }
    out: dict[str, xr.Dataset] = {}
    for name in layer_names:
        if name not in specs:
            raise ValueError(f"Unknown terrain layer {name!r}; expected dtm/dsm")
        endpoint, coverage_id = specs[name]
        out[name] = fetch_wcs_dataset(
            endpoint=endpoint,
            coverage_id=coverage_id,
            bbox=bbox,
            resolution_m=resolution_m,
            chunk_size=chunk_size,
            max_concurrent=max_concurrent,
            compute=compute,
        )
    return out


__all__ = [
    "EA_DSM_COVERAGE_ID",
    "EA_DSM_WCS_ENDPOINT",
    "EA_DTM_COVERAGE_ID",
    "EA_DTM_WCS_ENDPOINT",
    "EA_VOM_COVERAGE_ID",
    "EA_VOM_WCS_ENDPOINT",
    "ea_wcs_config",
    "fetch_ea_terrain_layers",
    "fetch_wcs_dataarray",
    "fetch_wcs_dataset",
]
