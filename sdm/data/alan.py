"""Artificial light at night (ALAN) layer helpers.

Expects a local yearly VIIRS night-lights GeoTIFF already downloaded by the
operator (EOG annual VNL / NASA Black Marble). Nothing is fetched from the
network here, and no radiance files belong in git.

ML handling (issue #37):
- Windowed study-area read only — never load the full global ~11.6 GB array.
- Preserve valid radiance zeros (average_masked background).
- Optional matching ``cf_cvg`` marks missing coverage as NaN (separate from zeros).
- Bilinear warp onto the exact model template grid.
- Band name ``alan_viirs_radiance`` end-to-end.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from sdm.raster.utils import reproject_data, squeeze_dataset
from sdm.utils.fs import uri_exists, uri_for_rasterio

# Default local cache paths (not committed). Prefer these when present.
DEFAULT_ALAN_SOURCE_PATH = Path("data/raw/big-files/alan/viirs_annual.tif")
DEFAULT_ALAN_CF_CVG_PATH = Path("data/raw/big-files/alan/viirs_cf_cvg.tif")

# Bucket raw-input objects (2025 EOG VNL v2.2). Used when the local cache is absent.
ALAN_BUCKET_PREFIX = "gs://habitat-suitability-modelling-data/raw/big-files/alan"
DEFAULT_ALAN_SOURCE_URI = f"{ALAN_BUCKET_PREFIX}/viirs_annual.tif"
DEFAULT_ALAN_CF_CVG_URI = f"{ALAN_BUCKET_PREFIX}/viirs_cf_cvg.tif"
DEFAULT_ALAN_CF_CVG_PROVENANCE_URI = (
    f"{ALAN_BUCKET_PREFIX}/viirs_cf_cvg.provenance.json"
)

# Canonical product (Matt: year 2025; average_masked radiance).
ALAN_CANONICAL_YEAR = 2025
ALAN_CANONICAL_PRODUCT_VERSION = "VNL v2.2"
ALAN_CANONICAL_RADIANCE_BAND = "average_masked"

# Stack / merge / species / selection name (single band end-to-end).
ALAN_BAND_NAME = "alan_viirs_radiance"
ALAN_OUTPUT_NAME = "alan_viirs_radiance.tif"
ALAN_UNITS = "nW/cm^2/sr"

# Bilinear needs a few source pixels beyond the model footprint.
DEFAULT_INTERP_MARGIN_PIXELS = 2

# EOG annual VNL filename tokens (product/version/year for provenance).
_VNL_NAME_RE = re.compile(
    r"VNL_npp_(?P<year>\d{4})_global_(?P<config>[^_]+)_v(?P<version>[\d.]+)",
    re.IGNORECASE,
)


def resolve_alan_raster_uri(
    override: Optional[Union[str, Path]] = None,
    *,
    local_default: Union[str, Path],
    bucket_default: str,
    label: str,
) -> str:
    """Resolve an ALAN raster URI: explicit override, else local cache, else bucket.

    Local paths win when the file exists so offline rebuilds keep working.
    Bucket defaults are the canonical 2025 VNL v2.2 objects under
    ``gs://habitat-suitability-modelling-data/raw/big-files/alan/``.
    """
    if override is not None:
        ref = str(override)
        if uri_exists(ref):
            return ref
        raise FileNotFoundError(
            f"ALAN {label} not found at {ref}. Place the GeoTIFF locally or "
            f"pass a reachable path / gs:// URI."
        )
    local_ref = str(local_default)
    if uri_exists(local_ref):
        return local_ref
    if uri_exists(bucket_default):
        logging.info("ALAN %s: using bucket object %s", label, bucket_default)
        return bucket_default
    raise FileNotFoundError(
        f"ALAN {label} not found at {local_ref} or {bucket_default}. "
        "Sync from the bucket into data/raw/big-files/alan/ or pass an explicit path."
    )


def resolve_alan_source_uri(override: Optional[Union[str, Path]] = None) -> str:
    """Resolve the canonical 2025 average_masked radiance GeoTIFF."""
    return resolve_alan_raster_uri(
        override,
        local_default=DEFAULT_ALAN_SOURCE_PATH,
        bucket_default=DEFAULT_ALAN_SOURCE_URI,
        label="radiance source",
    )


def resolve_alan_cf_cvg_uri(
    override: Optional[Union[str, Path]] = None,
    *,
    required: bool = False,
) -> Optional[str]:
    """Resolve matching ``cf_cvg`` coverage; ``None`` when optional and absent."""
    if override is not None:
        return resolve_alan_raster_uri(
            override,
            local_default=DEFAULT_ALAN_CF_CVG_PATH,
            bucket_default=DEFAULT_ALAN_CF_CVG_URI,
            label="cf_cvg",
        )
    local_ref = str(DEFAULT_ALAN_CF_CVG_PATH)
    if uri_exists(local_ref):
        return local_ref
    if uri_exists(DEFAULT_ALAN_CF_CVG_URI):
        logging.info("ALAN cf_cvg: using bucket object %s", DEFAULT_ALAN_CF_CVG_URI)
        return DEFAULT_ALAN_CF_CVG_URI
    if required:
        raise FileNotFoundError(
            f"ALAN cf_cvg not found at {local_ref} or {DEFAULT_ALAN_CF_CVG_URI}."
        )
    return None


def file_sha256(path: Union[str, Path], *, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a local file (not for ``gs://`` URIs)."""
    ref = str(path)
    if ref.startswith("gs://"):
        raise ValueError(f"file_sha256 requires a local path, got {ref}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_vnl_source_metadata(source_path: Union[str, Path]) -> dict[str, Any]:
    """Extract year/product/version hints from path and GeoTIFF tags."""
    ref = str(source_path)
    filename = ref.rsplit("/", 1)[-1]
    meta: dict[str, Any] = {
        "filename": filename,
        "units": ALAN_UNITS,
        "product": "EOG VIIRS Nighttime Lights (VNL)",
        "canonical_year": ALAN_CANONICAL_YEAR,
        "canonical_product_version": ALAN_CANONICAL_PRODUCT_VERSION,
        "canonical_radiance_band": ALAN_CANONICAL_RADIANCE_BAND,
    }
    tag_name = ""
    if not ref.startswith("gs://") and Path(ref).exists():
        path = Path(ref)
        with rio.open(path) as src:
            tags = src.tags()
            tag_name = str(tags.get("TIFFTAG_DOCUMENTNAME", "") or "")
            meta["source_tags"] = {
                key: tags[key]
                for key in (
                    "TIFFTAG_DOCUMENTNAME",
                    "TIFFTAG_DATETIME",
                    "TIFFTAG_SOFTWARE",
                )
                if key in tags
            }
            meta["source_crs"] = str(src.crs) if src.crs else None
            meta["source_dtype"] = str(src.dtypes[0]) if src.count else None
            meta["source_nodata"] = src.nodata
            meta["source_band_count"] = int(src.count)
    search_blob = f"{filename} {tag_name}"
    match = _VNL_NAME_RE.search(search_blob.replace("\\", "/"))
    if match:
        meta["year"] = int(match.group("year"))
        meta["config"] = match.group("config")
        meta["version"] = match.group("version")
        # Filenames use ``_v2_``; EOG publishes the series as Annual VNL v2.2.
        meta["product_version"] = (
            "VNL v2.2" if match.group("version").startswith("2") else f"VNL v{match.group('version')}"
        )
    if "average_masked" in search_blob.lower():
        meta["radiance_band"] = "average_masked"
    return meta


def _validate_single_radiance_band(src: rio.DatasetReader, path: Path) -> None:
    """Require exactly one radiance band; refuse silent band-1 picks."""
    if src.count < 1:
        raise ValueError(f"ALAN source has no bands: {path}")
    if src.count > 1:
        descriptions = list(src.descriptions)
        raise ValueError(
            f"ALAN source has {src.count} bands {descriptions!r} at {path}. "
            "Provide a single-band radiance GeoTIFF (e.g. EOG average_masked) "
            "or a path that resolves to exactly one radiance band."
        )


def _expand_window(
    window: Window,
    *,
    margin_pixels: int,
    width: int,
    height: int,
) -> Window:
    col_off = max(0, int(window.col_off) - margin_pixels)
    row_off = max(0, int(window.row_off) - margin_pixels)
    col_end = min(width, int(window.col_off + window.width) + margin_pixels)
    row_end = min(height, int(window.row_off + window.height) + margin_pixels)
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(0, col_end - col_off),
        height=max(0, row_end - row_off),
    )


def _window_for_bounds(
    src: rio.DatasetReader,
    bounds: tuple[float, float, float, float],
    bounds_crs: Union[str, int],
    *,
    margin_pixels: int,
) -> Window:
    """Window covering ``bounds`` in ``bounds_crs``, plus interpolation margin."""
    left, bottom, right, top = transform_bounds(
        bounds_crs, src.crs, *bounds, densify_pts=21
    )
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    return _expand_window(
        window,
        margin_pixels=margin_pixels,
        width=src.width,
        height=src.height,
    )


def _data_array_from_window(
    src: rio.DatasetReader,
    window: Window,
    *,
    name: str,
) -> xr.DataArray:
    """Read one band via Rasterio window into a georeferenced DataArray.

    Zeros are kept as valid values. Declared source nodata (if any and not 0)
    becomes NaN; nodata=0 is refused because it would drop valid dark cells.
    """
    if src.nodata is not None and float(src.nodata) == 0.0:
        raise ValueError(
            "ALAN source declares nodata=0, which would treat valid dark "
            "background as missing. Use a product without nodata=0 "
            "(EOG average_masked has undeclared nodata)."
        )
    data = src.read(1, window=window)
    transform = src.window_transform(window)
    values = np.asarray(data, dtype=np.float32)
    if src.nodata is not None and not (
        isinstance(src.nodata, float) and np.isnan(src.nodata)
    ):
        values = np.where(values == src.nodata, np.nan, values).astype(np.float32)
    height, width = values.shape
    # Cell-centre coordinates for rioxarray.
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name=name,
    )
    da = da.rio.write_crs(src.crs)
    da = da.rio.write_transform(transform)
    da.rio.write_nodata(np.nan, inplace=True)
    da.attrs["long_name"] = name
    return da


def load_alan_source_windowed(
    source_path: Union[str, Path],
    *,
    target_bounds: tuple[float, float, float, float],
    target_crs: Union[str, int],
    margin_pixels: int = DEFAULT_INTERP_MARGIN_PIXELS,
    cf_cvg_path: Optional[Union[str, Path]] = None,
) -> xr.DataArray:
    """Window-read radiance (and optional cf_cvg) for the study footprint.

    Parameters
    ----------
    source_path:
        Local path or ``gs://`` URI to a single-band VIIRS radiance GeoTIFF.
    target_bounds:
        Model-grid bounds (xmin, ymin, xmax, ymax) in ``target_crs``.
    target_crs:
        CRS of ``target_bounds`` (typically the model CRS / EPSG:27700).
    margin_pixels:
        Extra source pixels for bilinear interpolation.
    cf_cvg_path:
        Optional matching EOG ``cf_cvg`` GeoTIFF (local or ``gs://``). Cells with
        coverage <= 0 become NaN (missing), distinct from valid radiance zeros.
        When omitted, tries the local default then the bucket object.
    """
    source_ref = str(source_path)
    if not uri_exists(source_ref):
        raise FileNotFoundError(
            f"ALAN source not found: {source_ref}. Place a yearly VIIRS "
            "night-lights GeoTIFF locally, use the bucket object, or pass "
            "alan_source_path. Do not commit the file."
        )
    source_open = uri_for_rasterio(source_ref)

    with rio.open(source_open) as src:
        _validate_single_radiance_band(src, Path(source_ref.split("/")[-1]))
        window = _window_for_bounds(
            src, target_bounds, target_crs, margin_pixels=margin_pixels
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"ALAN window is empty for bounds {target_bounds} in {target_crs} "
                f"against {source_ref}"
            )
        logging.info(
            "ALAN windowed read %s window=%s (margin_pixels=%s)",
            source_ref.rsplit("/", 1)[-1],
            (
                int(window.col_off),
                int(window.row_off),
                int(window.width),
                int(window.height),
            ),
            margin_pixels,
        )
        radiance = _data_array_from_window(src, window, name=ALAN_BAND_NAME)

    radiance.attrs["units"] = ALAN_UNITS
    radiance.attrs["long_name"] = ALAN_BAND_NAME

    # Caller (generate_alan_data) resolves local-or-bucket; here only use an
    # explicit path or the local default so unit tests stay offline.
    if cf_cvg_path is not None:
        coverage_ref: Optional[str] = str(cf_cvg_path)
    elif uri_exists(str(DEFAULT_ALAN_CF_CVG_PATH)):
        coverage_ref = str(DEFAULT_ALAN_CF_CVG_PATH)
    else:
        coverage_ref = None
    if coverage_ref is not None:
        if not uri_exists(coverage_ref):
            raise FileNotFoundError(
                f"ALAN cf_cvg not found: {coverage_ref}. Place the matching "
                "EOG cf_cvg GeoTIFF locally or use the bucket object."
            )
        with rio.open(uri_for_rasterio(coverage_ref)) as cov_src:
            if cov_src.count != 1:
                raise ValueError(
                    f"ALAN cf_cvg must be single-band, got {cov_src.count} "
                    f"at {coverage_ref}"
                )
            cov_window = _window_for_bounds(
                cov_src, target_bounds, target_crs, margin_pixels=margin_pixels
            )
            coverage = _data_array_from_window(
                cov_src, cov_window, name="cf_cvg"
            )
        # Align coverage to radiance grid (same product grid expected).
        if coverage.shape != radiance.shape or coverage.rio.transform() != radiance.rio.transform():
            coverage = coverage.rio.reproject_match(
                radiance, resampling=Resampling.nearest
            )
        missing = coverage.values <= 0
        values = np.array(radiance.values, dtype=np.float32, copy=True)
        values[missing] = np.nan
        radiance = radiance.copy(data=values)
        radiance.attrs["validity"] = (
            "cf_cvg<=0 set to NaN (missing coverage); radiance zeros preserved"
        )
        radiance.attrs["cf_cvg_uri"] = coverage_ref
        logging.info(
            "Applied cf_cvg validity mask (%d missing cells of %d) from %s",
            int(np.isnan(values).sum()),
            values.size,
            coverage_ref,
        )
    else:
        radiance.attrs["validity"] = (
            "no cf_cvg supplied; radiance zeros preserved; coverage gaps not masked"
        )
        logging.warning(
            "ALAN cf_cvg not found at %s or %s — valid zeros preserved, but "
            "missing-coverage cells are not masked.",
            DEFAULT_ALAN_CF_CVG_PATH,
            DEFAULT_ALAN_CF_CVG_URI,
        )

    return radiance


def load_alan_source(source_path: Union[str, Path]) -> xr.DataArray:
    """Open a local ALAN GeoTIFF as a single-band DataArray (full extent).

    Prefer :func:`load_alan_source_windowed` for production rebuilds so the
    global composite is never materialised. This helper remains for small
    fixtures and validation of band rules.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ALAN source not found: {path}. Place a yearly VIIRS night-lights "
            "GeoTIFF there (or pass alan_source_path). Do not commit the file."
        )
    with rio.open(path) as src:
        _validate_single_radiance_band(src, path)
        if src.nodata is not None and float(src.nodata) == 0.0:
            raise ValueError(
                "ALAN source declares nodata=0, which would treat valid dark "
                "background as missing."
            )
    # masked=False so undeclared zeros stay valid float values.
    data = rxr.open_rasterio(path, masked=False)
    if isinstance(data, xr.Dataset):
        first = next(iter(data.data_vars))
        data = data[first]
    if "band" in data.dims and data.sizes.get("band", 1) == 1:
        data = data.squeeze("band", drop=True)
    data = data.astype(np.float32)
    data.rio.write_nodata(np.nan, inplace=True)
    data.attrs["long_name"] = ALAN_BAND_NAME
    data.attrs["units"] = ALAN_UNITS
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
    Missing (NaN) is preserved through the warp; valid zeros are preserved.
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
    reprojected.attrs["units"] = source.attrs.get("units", ALAN_UNITS)
    if "validity" in source.attrs:
        reprojected.attrs["validity"] = source.attrs["validity"]
    reprojected.attrs["resampling"] = resampling.name
    return reprojected


def write_alan_raster(
    data: xr.DataArray,
    output_path: Union[str, Path],
    *,
    provenance: Optional[dict[str, Any]] = None,
) -> Path:
    """Write the ALAN layer as a single-band GeoTIFF with band long_name set."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = data.copy()
    out.attrs["long_name"] = ALAN_BAND_NAME
    out.attrs.setdefault("units", ALAN_UNITS)
    if provenance:
        # GDAL metadata values must be strings.
        for key, value in provenance.items():
            if value is None:
                continue
            out.attrs[f"alan_{key}"] = str(value)
    squeezed = squeeze_dataset(out.to_dataset(name=ALAN_BAND_NAME))
    squeezed[ALAN_BAND_NAME].attrs["long_name"] = ALAN_BAND_NAME
    squeezed.rio.to_raster(path)
    logging.info("Wrote ALAN layer to %s", path)
    return path


def build_alan_provenance(
    source_path: Union[str, Path],
    *,
    cf_cvg_path: Optional[Union[str, Path]] = None,
    resampling: Resampling = Resampling.bilinear,
    checksum: bool = True,
) -> dict[str, Any]:
    """Provenance block for rebuild metrics / GeoTIFF tags."""
    source_ref = str(source_path)
    local_source = Path(source_ref) if not source_ref.startswith("gs://") else None
    meta = parse_vnl_source_metadata(local_source if local_source is not None else source_ref)
    meta["resampling"] = resampling.name
    meta["canonical_year"] = ALAN_CANONICAL_YEAR
    meta["canonical_product_version"] = ALAN_CANONICAL_PRODUCT_VERSION
    meta["canonical_radiance_band"] = ALAN_CANONICAL_RADIANCE_BAND
    cf_ref = str(cf_cvg_path) if cf_cvg_path is not None else None
    cf_present = bool(cf_ref and uri_exists(cf_ref))
    meta["validity"] = (
        "cf_cvg<=0 → NaN; radiance zeros preserved as valid background"
        if cf_present
        else "radiance zeros preserved; cf_cvg not applied"
    )
    if cf_ref is not None:
        meta["cf_cvg_path"] = cf_ref
        meta["cf_cvg_present"] = cf_present
        if (
            checksum
            and cf_present
            and not cf_ref.startswith("gs://")
            and Path(cf_ref).exists()
        ):
            meta["cf_cvg_sha256"] = file_sha256(cf_ref)
    if (
        checksum
        and local_source is not None
        and local_source.exists()
    ):
        meta["sha256"] = file_sha256(local_source)
    meta["path"] = source_ref
    return meta


__all__ = [
    "ALAN_BAND_NAME",
    "ALAN_BUCKET_PREFIX",
    "ALAN_CANONICAL_PRODUCT_VERSION",
    "ALAN_CANONICAL_RADIANCE_BAND",
    "ALAN_CANONICAL_YEAR",
    "ALAN_OUTPUT_NAME",
    "ALAN_UNITS",
    "DEFAULT_ALAN_CF_CVG_PATH",
    "DEFAULT_ALAN_CF_CVG_PROVENANCE_URI",
    "DEFAULT_ALAN_CF_CVG_URI",
    "DEFAULT_ALAN_SOURCE_PATH",
    "DEFAULT_ALAN_SOURCE_URI",
    "DEFAULT_INTERP_MARGIN_PIXELS",
    "build_alan_provenance",
    "file_sha256",
    "load_alan_source",
    "load_alan_source_windowed",
    "parse_vnl_source_metadata",
    "reproject_alan_to_model_grid",
    "resolve_alan_cf_cvg_uri",
    "resolve_alan_raster_uri",
    "resolve_alan_source_uri",
    "write_alan_raster",
]
