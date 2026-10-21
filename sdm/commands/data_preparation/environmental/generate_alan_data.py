"""Build an artificial light at night (ALAN) layer from a local VIIRS GeoTIFF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from rasterio.enums import Resampling

from sdm.data.alan import (
    ALAN_OUTPUT_NAME,
    DEFAULT_ALAN_SOURCE_PATH,
    load_alan_source,
    reproject_alan_to_model_grid,
    write_alan_raster,
)
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging


def generate_alan_data(
    output_dir: Path,
    boundary_path: Path,
    *,
    alan_source_path: Optional[Union[str, Path]] = None,
    buffer_distance_m: float = 7000,
    resampling: Resampling = Resampling.bilinear,
    verbose: bool = False,
) -> Path:
    """Reproject a local yearly VIIRS night-lights raster onto the model grid.

    Parameters
    ----------
    output_dir:
        Directory that receives ``alan_viirs_radiance.tif``.
    boundary_path:
        Study-area boundary used for the model transform.
    alan_source_path:
        Path to a local VIIRS annual composite. Defaults to
        ``data/raw/big-files/alan/viirs_annual.tif``. The file must already
        exist; this step does not download.
    buffer_distance_m:
        Boundary buffer matching other environmental producers.
    resampling:
        Resampling used when upsampling ~500 m VIIRS cells to the model grid
        (default bilinear).
    verbose:
        Verbose logging.

    Returns
    -------
    Path
        Written GeoTIFF path.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(
        alan_source_path if alan_source_path is not None else DEFAULT_ALAN_SOURCE_PATH
    )

    logging.info("Starting ALAN layer generation from %s", source_path)
    boundary_gdf, model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path, buffer_distance=buffer_distance_m)
    )
    source = load_alan_source(source_path)
    reprojected = reproject_alan_to_model_grid(
        source,
        target_crs=boundary_gdf.crs,
        target_transform=model_transform,
        target_resolution=spatial_config["resolution"],
        target_bounds=grid_bounds,
        resampling=resampling,
    )
    output_path = output_dir / ALAN_OUTPUT_NAME
    write_alan_raster(reprojected, output_path)
    logging.info(
        "ALAN layer complete (source=%s, resampling=%s, resolution=%sm)",
        source_path,
        resampling.name,
        spatial_config["resolution"],
    )
    return output_path


__all__ = ["generate_alan_data"]
