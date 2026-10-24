"""Git commit and source version/date metadata for rebuild_metrics.json."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from sdm.data.alan import parse_vnl_source_metadata
from sdm.data.coastline_download import PRODUCT_ID as BOUNDARY_LINE_PRODUCT_ID
from sdm.data.os_download import BASE_URL as OS_DOWNLOADS_BASE_URL
from sdm.data.os_download import PRODUCT_ID as OS_VMD_PRODUCT_ID

DEFAULT_CEH_PATH = Path(
    "data/raw/big-files/CEH/data/"
    "7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"
)
DEFAULT_ALAN_PATH = Path("data/raw/big-files/alan/viirs_annual.tif")
DEFAULT_ALAN_CF_CVG_PATH = Path("data/raw/big-files/alan/viirs_cf_cvg.tif")
ALAN_PRODUCT = "EOG VIIRS Nighttime Lights (VNL) annual average_masked"

WORLDCLIM_PRODUCT = "WorldClim 2.1"
WORLDCLIM_URL_PATTERN = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/"
    "GBR_wc2.1_30s_{var}.tif"
)
EA_DTM_WCS = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-terrain-model-dtm-1m/wcs"
)
EA_DSM_WCS = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-surface-model-last-return-dsm-1m/wcs"
)
EA_VOM_WCS = (
    "https://environment.data.gov.uk/spatialdata/vegetation-object-model/wcs"
)
EA_DTM_DATASET_NAME = "LiDAR Composite Digital Terrain Model (DTM) 1 m"
EA_DSM_DATASET_NAME = (
    "LiDAR Composite Digital Surface Model (DSM) last return 1 m"
)
EA_VOM_DATASET_NAME = "Vegetation Object Model (VOM)"
CEH_PRODUCT = "CEH Land Cover Map 2023 10 m"

_OS_PRODUCT_TIMEOUT_SECONDS = 15

logger = logging.getLogger(__name__)


def git_commit_sha(*, cwd: Optional[Path] = None) -> Optional[str]:
    """Return the current HEAD commit SHA, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = (result.stdout or "").strip()
    return sha or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _path_source_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "path": str(path),
        "present": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        meta["size_bytes"] = int(stat.st_size)
        meta["mtime_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).replace(microsecond=0).isoformat()
    return meta


def fetch_os_product_release(
    product_id: str,
    *,
    base_url: str = OS_DOWNLOADS_BASE_URL,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """Return OS Downloads product id, name, and release version/date.

    The OS Downloads API exposes ``version`` on ``GET /products/{id}``
    (e.g. ``2026-05`` for Vector Map District). Network failures leave
    ``version`` / ``name`` as ``None`` so rebuild metrics still write.
    """
    meta: dict[str, Any] = {
        "product_id": product_id,
        "name": None,
        "version": None,
        "product_url": f"{base_url.rstrip('/')}/products/{product_id}",
    }
    client = session or requests.Session()
    try:
        response = client.get(
            meta["product_url"],
            timeout=_OS_PRODUCT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (OSError, requests.RequestException, ValueError) as exc:
        logger.info("OS product metadata unavailable for %s: %s", product_id, exc)
        meta["lookup_error"] = type(exc).__name__
        return meta

    if isinstance(payload, dict):
        name = payload.get("name")
        version = payload.get("version")
        if name is not None:
            meta["name"] = str(name)
        if version is not None:
            # OS publishes release labels like "2026-05" / "2026-10".
            meta["version"] = str(version)
            meta["release"] = str(version)
    return meta


def _alan_source_meta(
    alan_path: Path,
    *,
    cf_cvg_path: Path,
) -> dict[str, Any]:
    """Local ALAN radiance + optional cf_cvg provenance (no secrets)."""
    meta: dict[str, Any] = {
        "product": ALAN_PRODUCT,
        "band_name": "alan_viirs_radiance",
        "units": "nW/cm^2/sr",
        "resampling": "bilinear",
        "validity": (
            "preserve radiance zeros; cf_cvg<=0 → NaN when coverage present"
        ),
        **_path_source_meta(alan_path),
        "cf_cvg": _path_source_meta(cf_cvg_path),
    }
    if alan_path.exists():
        try:
            parsed = parse_vnl_source_metadata(alan_path)
        except (OSError, ValueError) as exc:
            logger.info("ALAN tag parse skipped: %s", exc)
        else:
            for key in ("year", "product_version", "config", "radiance_band"):
                if key in parsed:
                    meta[key] = parsed[key]
            if "source_tags" in parsed:
                doc = parsed["source_tags"].get("TIFFTAG_DOCUMENTNAME")
                if doc:
                    meta["source_document_name"] = doc
    return meta


def collect_rebuild_provenance(
    *,
    ceh_data_path: Path | str | None = None,
    alan_source_path: Path | str | None = None,
    alan_cf_cvg_path: Path | str | None = None,
    cwd: Optional[Path] = None,
    session: Optional[requests.Session] = None,
    fetched_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    """Build provenance block for ``rebuild_metrics.json``.

    Records the git commit and each wired source's product/version or local
    file date. OS products include the Downloads API release version/date.
    Environment Agency WCS sources record the dataset name and fetch time
    (the service does not expose a stable product version). Does not include
    study-area names, grid references, or secrets.
    """
    ceh_path = Path(ceh_data_path) if ceh_data_path is not None else DEFAULT_CEH_PATH
    alan_path = (
        Path(alan_source_path) if alan_source_path is not None else DEFAULT_ALAN_PATH
    )
    cf_cvg_path = (
        Path(alan_cf_cvg_path)
        if alan_cf_cvg_path is not None
        else DEFAULT_ALAN_CF_CVG_PATH
    )
    fetched = fetched_at_utc or _utc_now_iso()
    return {
        "recorded_at_utc": fetched,
        "git_commit": git_commit_sha(cwd=cwd),
        "sources": {
            "worldclim": {
                "product": WORLDCLIM_PRODUCT,
                "version": "2.1",
                "url_pattern": WORLDCLIM_URL_PATTERN,
            },
            "os_vector_map_district": {
                **fetch_os_product_release(
                    OS_VMD_PRODUCT_ID, session=session
                ),
                "api_base_url": OS_DOWNLOADS_BASE_URL,
            },
            "os_boundary_line": {
                **fetch_os_product_release(
                    BOUNDARY_LINE_PRODUCT_ID, session=session
                ),
                "api_base_url": OS_DOWNLOADS_BASE_URL,
            },
            "ea_lidar_dtm": {
                "dataset_name": EA_DTM_DATASET_NAME,
                "endpoint": EA_DTM_WCS,
                "fetched_at_utc": fetched,
            },
            "ea_lidar_dsm": {
                "dataset_name": EA_DSM_DATASET_NAME,
                "endpoint": EA_DSM_WCS,
                "fetched_at_utc": fetched,
            },
            "ea_vom": {
                "dataset_name": EA_VOM_DATASET_NAME,
                "endpoint": EA_VOM_WCS,
                "fetched_at_utc": fetched,
            },
            "ceh_land_cover": {
                "product": CEH_PRODUCT,
                **_path_source_meta(ceh_path),
            },
            "alan_viirs": _alan_source_meta(alan_path, cf_cvg_path=cf_cvg_path),
        },
    }
