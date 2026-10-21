"""Artificial light at night (ALAN) layer helpers.

Expects a local yearly VIIRS night-lights GeoTIFF already downloaded by the
operator (NASA Black Marble or Earth Observation Group annual composites).
Nothing is fetched from the network here, and no radiance files belong in git.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling

from sdm.raster.utils import reproject_data, squeeze_dataset

# Default local cache path for a yearly VIIRS composite (not committed).
DEFAULT_ALAN_SOURCE_PATH = Path("data/raw/big-files/alan/viirs_annual.tif")
ALAN_BAND_NAME = "viirs_radiance"
ALAN_OUTPUT_NAME = "alan_viirs_radiance.tif"


def load_alan_source(source_path: Union[str, Path]) -> xr.DataArray:
    """Open a local ALAN GeoTIFF as a single-band DataArray."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ALAN source not found: {path}. Place a yearly VIIRS night-lights "
            "GeoTIFF there (or pass alan_source_path). Do not commit the file."
        )
    data = rxr.open_rasterio(path, masked=True)
    if isinstance(data, xr.Dataset):
        # Single-band rasters sometimes open as Dataset depending on tags.
        first = next(iter(data.data_vars))
        data = data[first]
    if "band" in data.dims and data.sizes.get("band", 1) == 1:
        data = data.squeeze("band", drop=True)
    elif "band" in data.dims and data.sizes["band"] > 1:
        logging.warning(
            "ALAN source has %d bands; using band 1 only", data.sizes["band"]
        )
        data = data.isel(band=0, drop=True)
    data.attrs["long_name"] = ALAN_BAND_NAME
    return data


def reproject_alan_to_model_grid(
    source: xr.DataArray,
    *,
    target_crs: Union[str, int],
    target_transform: Affine,
    target_resolution: float,
    target_bounds: tuple[float, float, float, float],
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Warp coarse VIIRS radiance onto the model template grid.

    VIIRS annual composites are about 500 m; the model grid is typically 100 m.
    Upsampling uses bilinear resampling by default so values stay continuous.
    Document the chosen product year and resampling in the rebuild metrics.
    """
    reprojected = reproject_data(
        source,
        crs=target_crs,
        transform=target_transform,
        resolution=target_resolution,
        resampling=resampling,
        bounds=target_bounds,
    )
    if isinstance(reprojected, xr.Dataset):
        first = next(iter(reprojected.data_vars))
        reprojected = reprojected[first]
    reprojected.attrs["long_name"] = ALAN_BAND_NAME
    reprojected.attrs["units"] = source.attrs.get(
        "units", "nW/cm2/sr (VIIRS radiance; confirm per product)"
    )
    return reprojected


def write_alan_raster(data: xr.DataArray, output_path: Union[str, Path]) -> Path:
    """Write the ALAN layer as a single-band GeoTIFF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    squeezed = squeeze_dataset(data.to_dataset(name=ALAN_BAND_NAME))
    squeezed.rio.to_raster(path)
    logging.info("Wrote ALAN layer to %s", path)
    return path


__all__ = [
    "ALAN_BAND_NAME",
    "ALAN_OUTPUT_NAME",
    "DEFAULT_ALAN_SOURCE_PATH",
    "load_alan_source",
    "reproject_alan_to_model_grid",
    "write_alan_raster",
]
