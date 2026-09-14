"""ONS Open Geography Portal client for counties / unitary authorities boundaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

# December 2024 BDY_CTYUA (Full resolution - clipped to coastline / BFC).
PRODUCT_ID = "Counties_and_Unitary_Authorities_December_2024_Boundaries_UK_BFC"
FEATURE_SERVER_BASE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    f"{PRODUCT_ID}/FeatureServer/0"
)
QUERY_URL = f"{FEATURE_SERVER_BASE}/query"

DEFAULT_CACHE_DIR = Path("data/raw/ons-boundaries")
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "counties_unitary_authorities_dec2024_bfc.geojson"

# Legacy manual drop (May 2023 vintage) kept as a valid fallback.
LEGACY_MANUAL_FILE = Path(
    "data/raw/big-files/"
    "Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"
)

# Skipton sits within North Yorkshire CTYUA — used for smoke tests / docs.
SKIPTON_COUNTY_NAME = "North Yorkshire"
YORKSHIRE_SMOKE_COUNTIES = ("Sheffield", SKIPTON_COUNTY_NAME)

COUNTY_NAME_COLUMNS = ("CTYUA24NM", "CTYUA23NM", "CTYUA22NM")


def county_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Return the CTYUA name column present in an ONS counties GeoDataFrame."""
    for column in COUNTY_NAME_COLUMNS:
        if column in gdf.columns:
            return column
    raise ValueError(
        "Counties GeoJSON is missing a CTYUA name column "
        f"(expected one of {COUNTY_NAME_COLUMNS}). "
        f"Found columns: {list(gdf.columns)}"
    )


class ONSBoundariesClient:
    """Thin client for ONS Open Geography Portal FeatureServer queries."""

    def __init__(
        self,
        query_url: str = QUERY_URL,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.query_url = query_url
        self.session = session or requests.Session()

    def query_geojson(
        self,
        where: str = "1=1",
        out_fields: str = "*",
        out_sr: int = 4326,
        page_size: int = 2000,
    ) -> dict:
        """Fetch GeoJSON features, paginating when the service transfer limit applies."""
        features: list[dict] = []
        crs: Optional[dict] = None
        offset = 0

        while True:
            params = {
                "where": where,
                "outFields": out_fields,
                "outSR": out_sr,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            response = self.session.get(self.query_url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()

            if "error" in payload:
                raise ValueError(f"ONS FeatureServer error: {payload['error']}")

            page_features = payload.get("features", [])
            features.extend(page_features)
            if crs is None and "crs" in payload:
                crs = payload["crs"]

            exceeded = payload.get("properties", {}).get("exceededTransferLimit", False)
            if not exceeded or not page_features:
                break
            offset += len(page_features)

        result: dict = {"type": "FeatureCollection", "features": features}
        if crs is not None:
            result["crs"] = crs
        return result

    def download_counties_geojson(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Download the full UK counties / unitary authorities layer to GeoJSON."""
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info("Using cached ONS counties GeoJSON at %s", output_path)
            return output_path

        logger.info(
            "Downloading ONS %s from %s to %s",
            PRODUCT_ID,
            self.query_url,
            output_path,
        )
        payload = self.query_geojson()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        logger.info(
            "Saved %d ONS county features to %s",
            len(payload.get("features", [])),
            output_path,
        )
        return output_path


def resolve_counties_file(
    counties_file: Optional[Path] = None,
    *,
    cache_file: Path = DEFAULT_CACHE_FILE,
    live_download: bool = True,
    overwrite: bool = False,
    client: Optional[ONSBoundariesClient] = None,
) -> Path:
    """Resolve a counties GeoJSON path, preferring explicit/manual paths then live cache.

    Resolution order:
    1. Explicit ``counties_file`` when it exists on disk.
    2. Legacy manual drop under ``data/raw/big-files/`` (May 2023 vintage).
    3. Cached live download under ``data/raw/ons-boundaries/``.
    4. Live download from the ONS Open Geography Portal when ``live_download`` is True.
    """
    if counties_file is not None:
        path = Path(counties_file)
        if path.exists():
            logger.info("Using counties GeoJSON at %s", path)
            return path
        if not live_download:
            raise FileNotFoundError(f"Counties file not found: {path}")

    if LEGACY_MANUAL_FILE.exists():
        logger.info("Using legacy manual counties file at %s", LEGACY_MANUAL_FILE)
        return LEGACY_MANUAL_FILE

    cache_path = Path(cache_file)
    if cache_path.exists() and not overwrite:
        logger.info("Using cached ONS counties GeoJSON at %s", cache_path)
        return cache_path

    if not live_download:
        raise FileNotFoundError(
            "No counties GeoJSON found. Place a manual file at "
            f"{LEGACY_MANUAL_FILE} or {cache_path}, or enable live_download."
        )

    downloader = client or ONSBoundariesClient()
    return downloader.download_counties_geojson(cache_path, overwrite=overwrite)
