"""OS Data Hub Downloads API client for Vector Map District."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

PRODUCT_ID = "VectorMapDistrict"
SHAPEFILE_FORMAT = "ESRI® Shapefile"
BASE_URL = "https://api.os.uk/downloads/v1"

# 100 km National Grid squares (rows N0..N12, columns E0..E6).
_GRID_100KM: tuple[tuple[str, ...], ...] = (
    ("SV", "SW", "SX", "SY", "SZ", "TV", ""),
    ("SQ", "SR", "SS", "ST", "SU", "TQ", "TR"),
    ("SL", "SM", "SN", "SO", "SP", "TL", "TM"),
    ("SF", "SG", "SH", "SJ", "SK", "TF", "TG"),
    ("SA", "SB", "SC", "SD", "SE", "TA", "TB"),
    ("NV", "NW", "NX", "NY", "NZ", "OV", ""),
    ("NQ", "NR", "NS", "NT", "NU", "OQ", "OR"),
    ("NL", "NM", "NN", "NO", "NP", "OL", "OM"),
    ("NF", "NG", "NH", "NJ", "NK", "OF", "OG"),
    ("NA", "NB", "NC", "ND", "NE", "OA", "OB"),
    ("HV", "HW", "HX", "HY", "HZ", "JV", ""),
    ("HQ", "HR", "HS", "HT", "HU", "JQ", "JR"),
    ("HL", "HM", "HN", "HO", "HP", "HL", "HM"),
)

REQUIRED_LAYERS = ("Building", "Water", "Woodland", "Road")

# Skipton ~3 km AOI used for smoke tests and documentation.
SKIPTON_AOI_BBOX: Tuple[float, float, float, float] = (
    397_100,
    450_300,
    400_100,
    453_300,
)


def get_os_data_hub_key() -> Optional[str]:
    """Return optional OS Data Hub API key from the environment."""
    return os.environ.get("OS_DATA_HUB_KEY") or os.environ.get("OS_DOWNLOADS_API_KEY")


def osgb_100km_tile(easting: float, northing: float) -> str:
    """Return the two-letter 100 km National Grid tile for an OSGB coordinate."""
    e100 = int(easting // 100_000)
    n100 = int(northing // 100_000)
    if not (0 <= e100 < len(_GRID_100KM[0]) and 0 <= n100 < len(_GRID_100KM)):
        raise ValueError(
            f"Coordinates outside British National Grid 100 km index: "
            f"E={easting}, N={northing}"
        )
    tile = _GRID_100KM[n100][e100]
    if not tile:
        raise ValueError(
            f"No 100 km tile defined for OSGB E={easting}, N={northing}"
        )
    return tile


def osgb_100km_tiles_for_bbox(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> list[str]:
    """Return sorted unique 100 km tile codes intersecting an OSGB bbox."""
    tiles: set[str] = set()
    for easting in (minx, maxx):
        for northing in (miny, maxy):
            tiles.add(osgb_100km_tile(easting, northing))
    return sorted(tiles)


class OSDownloadsClient:
    """Thin client for the OS Downloads API (OpenData Vector Map District)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else get_os_data_hub_key()
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _request_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"key": self.api_key}
        return {}

    def list_downloads(
        self,
        area: str,
        product_id: str = PRODUCT_ID,
        fmt: str = SHAPEFILE_FORMAT,
    ) -> list[dict]:
        """List available downloads for a product area and format."""
        url = f"{self.base_url}/products/{product_id}/downloads"
        response = self.session.get(
            url,
            params={"area": area, "format": fmt},
            headers=self._request_headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Expected list of downloads, got {type(payload)}")
        return payload

    def download_vector_map_district_tile(
        self,
        area: str,
        output_dir: Path,
        *,
        product_id: str = PRODUCT_ID,
        overwrite: bool = False,
    ) -> Path:
        """Download one Vector Map District shapefile zip for a 100 km tile."""
        downloads = self.list_downloads(area, product_id=product_id)
        if not downloads:
            raise FileNotFoundError(
                f"No Vector Map District downloads returned for tile {area}"
            )

        entry = downloads[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / entry["fileName"]
        if zip_path.exists() and not overwrite:
            logger.info("Using cached OS download %s", zip_path)
            return zip_path

        download_url = entry["url"]
        if "redirect" not in download_url:
            separator = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{separator}redirect"

        logger.info("Downloading OS Vector Map District tile %s to %s", area, zip_path)
        response = self.session.get(
            download_url,
            headers=self._request_headers(),
            timeout=300,
            stream=True,
        )
        response.raise_for_status()

        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        return zip_path

    @staticmethod
    def extract_shapefile_zip(zip_path: Path, extract_dir: Path) -> Path:
        """Extract a Vector Map District zip archive."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        return extract_dir


def _raw_dir_has_layers(raw_dir: Path, datasets: Sequence[str]) -> bool:
    """Return True when all requested layer shapefiles are present under raw_dir."""
    for dataset in datasets:
        if not list(raw_dir.glob(f"**/*{dataset}*.shp")):
            return False
    return True


def _tile_dir_has_data(tile_dir: Path) -> bool:
    return tile_dir.exists() and any(tile_dir.glob("**/*.shp"))


def ensure_os_vector_map_data(
    bbox: Tuple[float, float, float, float],
    raw_dir: Path | str,
    datasets: Sequence[str] = REQUIRED_LAYERS,
    *,
    api_key: Optional[str] = None,
    overwrite: bool = False,
    client: Optional[OSDownloadsClient] = None,
) -> list[str]:
    """Ensure Vector Map District shapefiles exist for the bbox, downloading if needed.

    Manual drops under ``raw_dir`` are kept when all requested layers are already
    present. Otherwise the client downloads the intersecting 100 km tiles from the
    OS Downloads API and extracts them under ``raw_dir/<TILE>/``.
    """
    raw_path = Path(raw_dir)
    if not overwrite and _raw_dir_has_layers(raw_path, datasets):
        logger.info("Using existing OS Vector Map shapefiles in %s", raw_path)
        return osgb_100km_tiles_for_bbox(*bbox)

    tiles = osgb_100km_tiles_for_bbox(*bbox)
    downloader = client or OSDownloadsClient(api_key=api_key)
    download_cache = raw_path / ".downloads"

    for tile in tiles:
        tile_dir = raw_path / tile
        if not overwrite and _tile_dir_has_data(tile_dir):
            logger.info("OS tile %s already extracted at %s", tile, tile_dir)
            continue

        zip_path = downloader.download_vector_map_district_tile(
            tile,
            download_cache,
            overwrite=overwrite,
        )
        OSDownloadsClient.extract_shapefile_zip(zip_path, tile_dir)

    if not _raw_dir_has_layers(raw_path, datasets):
        missing = [
            dataset
            for dataset in datasets
            if not list(raw_path.glob(f"**/*{dataset}*.shp"))
        ]
        raise FileNotFoundError(
            "OS Vector Map District layers still missing after live download: "
            f"{missing}. Place shapefiles under {raw_path} or check the AOI tiles "
            f"{tiles}."
        )

    return tiles
