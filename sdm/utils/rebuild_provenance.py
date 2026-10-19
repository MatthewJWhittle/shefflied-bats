"""Git commit and source version/date metadata for rebuild_metrics.json."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sdm.data.coastline_download import PRODUCT_ID as BOUNDARY_LINE_PRODUCT_ID
from sdm.data.os_download import BASE_URL as OS_DOWNLOADS_BASE_URL
from sdm.data.os_download import PRODUCT_ID as OS_VMD_PRODUCT_ID

DEFAULT_CEH_PATH = Path(
    "data/raw/big-files/CEH/data/"
    "7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif"
)

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
CEH_PRODUCT = "CEH Land Cover Map 2023 10 m"


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


def collect_rebuild_provenance(
    *,
    ceh_data_path: Path | str | None = None,
    cwd: Optional[Path] = None,
) -> dict[str, Any]:
    """Build provenance block for ``rebuild_metrics.json``.

    Records the git commit and each wired source's product/version or local
    file date. Does not include study-area names, grid references, or secrets.
    """
    ceh_path = Path(ceh_data_path) if ceh_data_path is not None else DEFAULT_CEH_PATH
    return {
        "recorded_at_utc": _utc_now_iso(),
        "git_commit": git_commit_sha(cwd=cwd),
        "sources": {
            "worldclim": {
                "product": WORLDCLIM_PRODUCT,
                "version": "2.1",
                "url_pattern": WORLDCLIM_URL_PATTERN,
            },
            "os_vector_map_district": {
                "product_id": OS_VMD_PRODUCT_ID,
                "api_base_url": OS_DOWNLOADS_BASE_URL,
            },
            "os_boundary_line": {
                "product_id": BOUNDARY_LINE_PRODUCT_ID,
                "api_base_url": OS_DOWNLOADS_BASE_URL,
            },
            "ea_lidar_wcs": {
                "dtm": EA_DTM_WCS,
                "dsm": EA_DSM_WCS,
            },
            "ea_vom_wcs": {
                "endpoint": EA_VOM_WCS,
            },
            "ceh_land_cover": {
                "product": CEH_PRODUCT,
                **_path_source_meta(ceh_path),
            },
        },
    }
