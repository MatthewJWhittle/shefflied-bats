"""Live coastline sources for coastal distance (OS Boundary-Line primary)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Literal, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import LineString

from sdm.data.os_download import BASE_URL, OSDownloadsClient

logger = logging.getLogger(__name__)

PRODUCT_ID = "BoundaryLine"
SHAPEFILE_FORMAT = "ESRI® Shapefile"
GB_AREA = "GB"
HIGH_WATER_LAYER = "high_water_polyline"
MHW_CODE = "0071"

DEFAULT_CACHE_DIR = Path("data/raw/boundary-line")
DEFAULT_MHW_SHAPEFILE = DEFAULT_CACHE_DIR / "high_water_polyline.shp"
DEFAULT_DOWNLOADS_DIR = DEFAULT_CACHE_DIR / ".downloads"
OSM_FALLBACK_GEOJSON = DEFAULT_CACHE_DIR / "osm_coastline_fallback.geojson"

LEGACY_BGS_GEOCOAST = Path(
    "data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp"
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Whitby / Scarborough smoke bbox (OSGB).
WHITBY_SMOKE_BBOX: Tuple[float, float, float, float] = (
    480_000,
    510_000,
    490_000,
    520_000,
)

# Skipton ~3 km AOI — inland; coastal distances should be large.
SKIPTON_AOI_BBOX: Tuple[float, float, float, float] = (
    397_100,
    450_300,
    400_100,
    453_300,
)

CoastlineSourceKind = Literal["boundary_line", "bgs_geocoast", "osm_fallback"]

# Expand OSM fallback queries so inland AOIs still reach the nearest coast.
OSM_FALLBACK_BUFFER_M = 250_000


class BoundaryLineClient:
    """OS Boundary-Line OpenData client (GB zip via the OS Downloads API)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._downloads = OSDownloadsClient(
            api_key=api_key,
            base_url=base_url,
            session=session,
        )

    def download_gb_shapefile_zip(
        self,
        output_dir: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Download the full GB Boundary-Line shapefile zip (~700 MB)."""
        downloads = self._downloads.list_downloads(
            GB_AREA,
            product_id=PRODUCT_ID,
            fmt=SHAPEFILE_FORMAT,
        )
        if not downloads:
            raise FileNotFoundError(
                f"No Boundary-Line downloads returned for area {GB_AREA}"
            )

        entry = downloads[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / entry["fileName"]
        if zip_path.exists() and not overwrite:
            logger.info("Using cached Boundary-Line download %s", zip_path)
            return zip_path

        download_url = entry["url"]
        if "redirect" not in download_url:
            separator = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{separator}redirect"

        logger.info(
            "Downloading OS Boundary-Line GB shapefile to %s",
            zip_path,
        )
        response = self._downloads.session.get(
            download_url,
            headers=self._downloads._request_headers(),
            timeout=600,
            stream=True,
        )
        response.raise_for_status()

        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        return zip_path

    @staticmethod
    def extract_high_water_polyline(zip_path: Path, extract_dir: Path) -> Path:
        """Extract only the GB mean-high-water polyline layer from a Boundary-Line zip."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        layer_prefix = f"Data/GB/{HIGH_WATER_LAYER}."
        with zipfile.ZipFile(zip_path, "r") as archive:
            extracted = [
                name
                for name in archive.namelist()
                if name.startswith(layer_prefix)
            ]
            if not extracted:
                raise FileNotFoundError(
                    f"{HIGH_WATER_LAYER} not found in Boundary-Line zip {zip_path}"
                )
            for name in extracted:
                archive.extract(name, extract_dir)

        shapefile = extract_dir / "Data" / "GB" / f"{HIGH_WATER_LAYER}.shp"
        if not shapefile.exists():
            raise FileNotFoundError(
                f"Expected extracted shapefile at {shapefile} from {zip_path}"
            )
        return shapefile


def _find_mhw_shapefile(base_dir: Path) -> Optional[Path]:
    matches = sorted(base_dir.glob(f"**/{HIGH_WATER_LAYER}.shp"))
    return matches[0] if matches else None


def filter_mean_high_water(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep Boundary-Line mean high water (springs) features (CODE 0071)."""
    if "CODE" not in gdf.columns:
        logger.warning(
            "Coastline layer has no CODE column; assuming all features are MHW"
        )
        return gdf

    code = gdf["CODE"].astype(str).str.zfill(4)
    filtered = gdf.loc[code == MHW_CODE].copy()
    if filtered.empty:
        raise ValueError(
            f"No mean high water features with CODE {MHW_CODE} in coastline layer"
        )
    return filtered


def load_mean_high_water_polylines(
    shapefile_path: Path,
    *,
    target_crs: Optional[str] = None,
    clip_bbox: Optional[Tuple[float, float, float, float]] = None,
    simplify_tolerance_m: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Load and optionally clip/simplify Boundary-Line MHW polylines."""
    gdf = gpd.read_file(shapefile_path)
    gdf = filter_mean_high_water(gdf)

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    if clip_bbox is not None:
        minx, miny, maxx, maxy = clip_bbox
        gdf = gdf.cx[minx:maxx, miny:maxy]

    if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance_m)

    return gdf


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    buffer_m: float,
) -> Tuple[float, float, float, float]:
    """Expand an OSGB bbox symmetrically by ``buffer_m`` metres."""
    minx, miny, maxx, maxy = bbox
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def fetch_osm_coastline_overpass(
    bbox_osgb: Tuple[float, float, float, float],
    *,
    target_crs: str = "EPSG:27700",
    session: Optional[requests.Session] = None,
) -> gpd.GeoDataFrame:
    """Fetch OSM coastline ways for an OSGB bbox (fallback when Boundary-Line fails)."""
    session = session or requests.Session()
    minx, miny, maxx, maxy = bbox_osgb
    corners = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([minx, maxx], [miny, maxy]),
        crs=target_crs,
    ).to_crs("EPSG:4326")
    west, south, east, north = corners.total_bounds

    query = f"""
    [out:json][timeout:120];
    way["natural"="coastline"]({south},{west},{north},{east});
    out geom;
    """
    logger.info(
        "Fetching OSM coastline fallback for OSGB bbox %s via Overpass",
        bbox_osgb,
    )
    response = session.post(OVERPASS_URL, data={"data": query}, timeout=180)
    response.raise_for_status()
    payload = response.json()

    lines: list[LineString] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry")
        if not geometry:
            continue
        coords = [(point["lon"], point["lat"]) for point in geometry]
        if len(coords) >= 2:
            lines.append(LineString(coords))

    if not lines:
        raise ValueError(
            f"No OSM coastline features returned for bbox {bbox_osgb}"
        )

    gdf = gpd.GeoDataFrame({"source": ["osm"] * len(lines)}, geometry=lines, crs="EPSG:4326")
    return gdf.to_crs(target_crs)


def _write_osm_fallback_geojson(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def ensure_boundary_line_mhw_shapefile(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    live_download: bool = True,
    overwrite: bool = False,
    client: Optional[BoundaryLineClient] = None,
    fallback_bbox: Optional[Tuple[float, float, float, float]] = None,
    allow_osm_fallback: bool = True,
) -> Path:
    """Ensure Boundary-Line MHW polylines exist locally, downloading if needed."""
    cache_path = Path(cache_dir)
    existing = _find_mhw_shapefile(cache_path)
    if existing is not None and not overwrite:
        logger.info("Using cached Boundary-Line MHW shapefile at %s", existing)
        return existing

    if not live_download:
        raise FileNotFoundError(
            "Boundary-Line mean high water shapefile not found and live_download is "
            f"disabled. Cache under {cache_path} or enable live_download."
        )

    downloader = client or BoundaryLineClient()
    try:
        zip_path = downloader.download_gb_shapefile_zip(
            cache_path / ".downloads",
            overwrite=overwrite,
        )
        return downloader.extract_high_water_polyline(zip_path, cache_path)
    except Exception as exc:
        logger.warning("Boundary-Line live download failed: %s", exc)
        if not allow_osm_fallback or fallback_bbox is None:
            raise

        expanded = expand_bbox(fallback_bbox, OSM_FALLBACK_BUFFER_M)
        logger.info(
            "Attempting OSM coastline fallback for expanded bbox %s (AOI %s)",
            expanded,
            fallback_bbox,
        )
        osm_gdf = fetch_osm_coastline_overpass(expanded)
        _write_osm_fallback_geojson(osm_gdf, OSM_FALLBACK_GEOJSON)
        return OSM_FALLBACK_GEOJSON


def resolve_coastline_source(
    coastline_path: Optional[Path] = None,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    live_download: bool = True,
    overwrite: bool = False,
    client: Optional[BoundaryLineClient] = None,
    fallback_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[Path, CoastlineSourceKind]:
    """Resolve coastline file path and source kind for coastal distance prep.

    Resolution order:
    1. Explicit ``coastline_path`` when it exists.
    2. Cached Boundary-Line ``high_water_polyline.shp`` under ``cache_dir``.
    3. Live Boundary-Line GB download when ``live_download`` is True.
    4. BGS GeoCoast manual drop (coastal-specific override) when present.
    5. OSM Overpass fallback when Boundary-Line download fails (live only).
    """
    if coastline_path is not None:
        path = Path(coastline_path)
        if path.exists():
            kind: CoastlineSourceKind = (
                "bgs_geocoast" if _is_bgs_geocoast_path(path) else "boundary_line"
            )
            logger.info("Using explicit coastline source at %s (%s)", path, kind)
            return path, kind
        if not live_download:
            raise FileNotFoundError(f"Coastline file not found: {path}")

    cache_path = Path(cache_dir)
    cached = _find_mhw_shapefile(cache_path)
    if cached is not None and not overwrite:
        logger.info("Using cached Boundary-Line MHW shapefile at %s", cached)
        return cached, "boundary_line"

    if live_download:
        try:
            path = ensure_boundary_line_mhw_shapefile(
                cache_path,
                live_download=True,
                overwrite=overwrite,
                client=client,
                fallback_bbox=fallback_bbox,
                allow_osm_fallback=fallback_bbox is not None,
            )
            kind = (
                "osm_fallback"
                if path.suffix.lower() == ".geojson"
                else "boundary_line"
            )
            return path, kind
        except Exception:
            if LEGACY_BGS_GEOCOAST.exists():
                logger.info(
                    "Boundary-Line unavailable; using BGS GeoCoast manual file at %s",
                    LEGACY_BGS_GEOCOAST,
                )
                return LEGACY_BGS_GEOCOAST, "bgs_geocoast"
            raise

    if LEGACY_BGS_GEOCOAST.exists():
        logger.info("Using BGS GeoCoast manual file at %s", LEGACY_BGS_GEOCOAST)
        return LEGACY_BGS_GEOCOAST, "bgs_geocoast"

    raise FileNotFoundError(
        "No coastline source found. Enable live_download for OS Boundary-Line, "
        f"place MHW shapefiles under {cache_path}, or provide BGS GeoCoast at "
        f"{LEGACY_BGS_GEOCOAST}."
    )


def _is_bgs_geocoast_path(path: Path) -> bool:
    normalized = str(path).lower()
    return "geocoast" in normalized or path.resolve() == LEGACY_BGS_GEOCOAST.resolve()


def coastline_source_is_polyline(source_kind: CoastlineSourceKind, path: Path) -> bool:
    """Return True when distance should be computed to polylines directly."""
    if source_kind in ("boundary_line", "osm_fallback"):
        return True
    if path.suffix.lower() == ".geojson" and source_kind != "bgs_geocoast":
        return True
    return False
