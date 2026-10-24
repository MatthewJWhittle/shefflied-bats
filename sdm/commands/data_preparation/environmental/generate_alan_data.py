"""Build an artificial light at night (ALAN) layer from a local VIIRS GeoTIFF."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from rasterio.enums import Resampling

from sdm.data.alan import (
    ALAN_OUTPUT_NAME,
    DEFAULT_ALAN_CF_CVG_PATH,
    DEFAULT_ALAN_SOURCE_PATH,
    DEFAULT_INTERP_MARGIN_PIXELS,
    build_alan_provenance,
    load_alan_source_windowed,
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
    alan_cf_cvg_path: Optional[Union[str, Path]] = None,
    buffer_distance_m: float = 0,
    interp_margin_pixels: int = DEFAULT_INTERP_MARGIN_PIXELS,
    resampling: Resampling = Resampling.bilinear,
    compute_checksum: bool = True,
    verbose: bool = False,
) -> Path:
    """Window-read and reproject a local yearly VIIRS night-lights raster.

    Reads only the study-area window (+ interpolation margin) from the global
    source — never materialises the full ~11.6 GB array for reproject.

    Parameters
    ----------
    output_dir:
        Directory that receives ``alan_viirs_radiance.tif``.
    boundary_path:
        Study-area boundary used for the model transform (already buffered by
        ``sdm setup`` when using the standard land-clip boundary).
    alan_source_path:
        Path to a local single-band VIIRS annual composite. Defaults to
        ``data/raw/big-files/alan/viirs_annual.tif``. The file must already
        exist; this step does not download.
    alan_cf_cvg_path:
        Optional matching EOG ``cf_cvg`` coverage GeoTIFF. When omitted, the
        default ``data/raw/big-files/alan/viirs_cf_cvg.tif`` is used if present.
    buffer_distance_m:
        Extra boundary buffer. Prefer 0 when the boundary already includes the
        study-area buffer from setup.
    interp_margin_pixels:
        Extra source pixels retained beyond the model footprint for bilinear.
    resampling:
        Resampling used when upsampling ~500 m VIIRS cells to the model grid
        (default bilinear).
    compute_checksum:
        When True, SHA-256 the source file(s) into provenance (slow on 11 GB).
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
    if alan_cf_cvg_path is not None:
        cf_cvg_path: Optional[Path] = Path(alan_cf_cvg_path)
    elif DEFAULT_ALAN_CF_CVG_PATH.exists():
        cf_cvg_path = DEFAULT_ALAN_CF_CVG_PATH
    else:
        cf_cvg_path = None

    logging.info("Starting ALAN layer generation from %s", source_path)
    boundary_gdf, model_transform, grid_bounds, spatial_config = (
        load_boundary_and_transform(boundary_path, buffer_distance=buffer_distance_m)
    )
    source = load_alan_source_windowed(
        source_path,
        target_bounds=grid_bounds,
        target_crs=boundary_gdf.crs,
        margin_pixels=interp_margin_pixels,
        cf_cvg_path=cf_cvg_path,
    )
    reprojected = reproject_alan_to_model_grid(
        source,
        target_crs=boundary_gdf.crs,
        target_transform=model_transform,
        target_resolution=spatial_config["resolution"],
        target_bounds=grid_bounds,
        resampling=resampling,
    )
    provenance = build_alan_provenance(
        source_path,
        cf_cvg_path=cf_cvg_path,
        resampling=resampling,
        checksum=compute_checksum,
    )
    provenance["interp_margin_pixels"] = interp_margin_pixels
    provenance["target_resolution_m"] = spatial_config["resolution"]
    provenance["note_100m_cells"] = (
        "Model cells are 100 m but still hold coarse (~500 m) satellite radiance"
    )

    output_path = output_dir / ALAN_OUTPUT_NAME
    write_alan_raster(reprojected, output_path, provenance=provenance)

    provenance_path = output_dir / "alan_provenance.json"
    _write_json(provenance_path, provenance)
    logging.info(
        "ALAN layer complete (source=%s, year=%s, resampling=%s, resolution=%sm, "
        "cf_cvg=%s)",
        source_path.name,
        provenance.get("year"),
        resampling.name,
        spatial_config["resolution"],
        bool(cf_cvg_path and Path(cf_cvg_path).exists()),
    )
    return output_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["generate_alan_data"]
