"""Build an artificial light at night (ALAN) layer from VIIRS GeoTIFF."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from rasterio.enums import Resampling

from sdm.data.alan import (
    ALAN_CANONICAL_PRODUCT_VERSION,
    ALAN_CANONICAL_RADIANCE_BAND,
    ALAN_CANONICAL_YEAR,
    ALAN_OUTPUT_NAME,
    DEFAULT_ALAN_CF_CVG_URI,
    DEFAULT_ALAN_SOURCE_URI,
    DEFAULT_INTERP_MARGIN_PIXELS,
    build_alan_provenance,
    load_alan_source_windowed,
    load_model_grid_from_template,
    reproject_alan_to_model_grid,
    resolve_alan_cf_cvg_uri,
    resolve_alan_source_uri,
    write_alan_raster,
)
from sdm.utils.fs import uri_exists
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.logging_utils import setup_logging


def generate_alan_data(
    output_dir: Path,
    boundary_path: Path,
    *,
    alan_source_path: Optional[Union[str, Path]] = None,
    alan_cf_cvg_path: Optional[Union[str, Path]] = None,
    template_raster: Optional[Union[str, Path]] = None,
    buffer_distance_m: float = 0,
    interp_margin_pixels: int = DEFAULT_INTERP_MARGIN_PIXELS,
    resampling: Resampling = Resampling.bilinear,
    compute_checksum: bool = True,
    verbose: bool = False,
) -> Path:
    """Window-read and reproject yearly VIIRS night-lights onto the model grid.

    Defaults resolve local cache first, then bucket objects under
    ``gs://habitat-suitability-modelling-data/raw/big-files/alan/``.
    Canonical product: 2025 EOG VNL v2.2 ``average_masked`` + matching ``cf_cvg``.

    When ``template_raster`` exists (typically ``paths.ev_tiff``), the output
    locks to that GeoTIFF's CRS / transform / extent so the layer matches the
    training stack. Otherwise the grid comes from ``boundary_path`` (same as
    other EV builders).
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_uri = resolve_alan_source_uri(alan_source_path)
    cf_cvg_uri = resolve_alan_cf_cvg_uri(alan_cf_cvg_path, required=False)

    logging.info("Starting ALAN layer generation from %s", source_uri)
    template_ref = str(template_raster) if template_raster is not None else None
    if template_ref and uri_exists(template_ref):
        target_crs, model_transform, grid_bounds, resolution = (
            load_model_grid_from_template(template_ref)
        )
        logging.info(
            "ALAN target grid from template %s (extent=%s, resolution=%sm)",
            template_ref,
            grid_bounds,
            resolution,
        )
        spatial_config = {"crs": str(target_crs), "resolution": resolution}
    else:
        if template_ref:
            logging.info(
                "ALAN template %s not present; using boundary-derived grid",
                template_ref,
            )
        boundary_gdf, model_transform, grid_bounds, spatial_config = (
            load_boundary_and_transform(
                boundary_path, buffer_distance=buffer_distance_m
            )
        )
        target_crs = boundary_gdf.crs
        resolution = spatial_config["resolution"]
    source = load_alan_source_windowed(
        source_uri,
        target_bounds=grid_bounds,
        target_crs=target_crs,
        margin_pixels=interp_margin_pixels,
        cf_cvg_path=cf_cvg_uri,
    )
    reprojected = reproject_alan_to_model_grid(
        source,
        target_crs=target_crs,
        target_transform=model_transform,
        target_resolution=resolution,
        target_bounds=grid_bounds,
        resampling=resampling,
    )
    checksum = compute_checksum and not str(source_uri).startswith("gs://")
    provenance = build_alan_provenance(
        source_uri,
        cf_cvg_path=cf_cvg_uri,
        resampling=resampling,
        checksum=checksum,
    )
    provenance["source_uri"] = source_uri
    provenance["cf_cvg_uri"] = cf_cvg_uri
    provenance["canonical_year"] = ALAN_CANONICAL_YEAR
    provenance["canonical_product_version"] = ALAN_CANONICAL_PRODUCT_VERSION
    provenance["canonical_radiance_band"] = ALAN_CANONICAL_RADIANCE_BAND
    provenance["bucket_radiance"] = DEFAULT_ALAN_SOURCE_URI
    provenance["bucket_cf_cvg"] = DEFAULT_ALAN_CF_CVG_URI
    provenance["interp_margin_pixels"] = interp_margin_pixels
    provenance["target_resolution_m"] = resolution
    provenance["target_bounds"] = list(grid_bounds)
    if template_ref and uri_exists(template_ref):
        provenance["template_raster"] = template_ref
    provenance["note_100m_cells"] = (
        "Model cells are 100 m but still hold coarse (~500 m) satellite radiance"
    )

    output_path = output_dir / ALAN_OUTPUT_NAME
    write_alan_raster(reprojected, output_path, provenance=provenance)
    _write_json(output_dir / "alan_provenance.json", provenance)
    logging.info(
        "ALAN layer complete (source=%s, year=%s, product=%s, resampling=%s, "
        "resolution=%sm, cf_cvg=%s)",
        source_uri.rsplit("/", 1)[-1],
        provenance.get("year", ALAN_CANONICAL_YEAR),
        ALAN_CANONICAL_PRODUCT_VERSION,
        resampling.name,
        resolution,
        bool(cf_cvg_uri and uri_exists(cf_cvg_uri)),
    )
    return output_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["generate_alan_data"]
