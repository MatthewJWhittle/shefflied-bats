"""
Client utilities for fetching occurrence data from the NBN Atlas API.

This module provides a thin wrapper around the public NBN Atlas web services:
`https://api.nbnatlas.org/` and, in particular, the records service
`https://records-ws.nbnatlas.org/occurrences/search`.

The main entry point is `fetch_occurrences_from_nbn`, which returns a
GeoDataFrame of occurrences for use elsewhere in the SDM pipeline.

Notes
-----
- This module is intentionally conservative and exposes only a small set of
  convenience parameters (scientific name, taxon ID, arbitrary query string).
  For more advanced usage, you can pass additional query parameters via the
  `extra_params` dict – see the NBN Atlas API documentation for the full set
  of supported parameters.
- Network access is required at runtime; this module does not implement any
  on-disk caching.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import aiohttp
import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

LOGGER = logging.getLogger(__name__)

RECORDS_BASE_URL = "https://records-ws.nbnatlas.org"
OCCURRENCE_SEARCH_PATH = "/occurrences/search"


def _point_radius_to_wkt(
    lat: float,
    lon: float,
    radius_km: float,
    num_segments: int = 36,
) -> str:
    """
    Approximate a circle (lat, lon, radius_km) as a WKT polygon.
    """
    import math

    if radius_km <= 0:
        raise ValueError("radius_km must be positive for point-radius filtering.")

    earth_radius_km = 6371.0
    angular_distance = radius_km / earth_radius_km

    lat_rad = math.radians(lat)

    points: list[str] = []
    for i in range(num_segments):
        angle = 2 * math.pi * i / num_segments
        dx = math.sin(angle) * angular_distance
        dy = math.cos(angle) * angular_distance

        new_lat_rad = lat_rad + dy
        new_lon_rad = math.radians(lon) + dx / math.cos(lat_rad)

        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)
        points.append(f"{new_lon} {new_lat}")

    # Close polygon
    points.append(points[0])
    return f"POLYGON(({', '.join(points)}))"


def _bbox_to_wkt(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
    """
    Convert an axis-aligned bounding box (lon/lat in EPSG:4326) to WKT polygon.
    """
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bbox: require min_lon < max_lon and min_lat < max_lat.")

    coords = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]
    coord_str = ", ".join(f"{x} {y}" for x, y in coords)
    return f"POLYGON(({coord_str}))"


def _build_query(
    scientific_name: Optional[str] = None,
    taxon_id: Optional[str] = None,
    query: Optional[str] = None,
    page_size: int = 1000,
    point: Optional[Tuple[float, float]] = None,
    radius_km: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    polygon: Optional[BaseGeometry] = None,
    wkt: Optional[str] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the base set of query parameters for the NBN Atlas occurrences API.
    """
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    params: Dict[str, Any] = {
        "page": 1,  # NBN Atlas uses 1-based page indexing
        "pageSize": page_size,
    }

    # Construct the primary query
    q_parts: list[str] = []
    if query:
        q_parts.append(query)
    if scientific_name:
        q_parts.append(f"scientificName:\"{scientific_name}\"")
    if taxon_id:
        # Taxon ID field name follows ALA/NBN conventions
        q_parts.append(f"lsid:{taxon_id}")

    if not q_parts:
        raise ValueError(
            "At least one of 'scientific_name', 'taxon_id', or 'query' must be provided."
        )

    params["q"] = " AND ".join(q_parts)

    # Spatial filtering
    # Only one of point+radius, bbox, polygon, or wkt is allowed
    spatial_args = sum(
        int(x is not None)
        for x in (
            (point if point is not None or radius_km is not None else None),
            bbox,
            polygon,
            wkt,
        )
    )
    if spatial_args > 1:
        raise ValueError(
            "Specify only one of (point+radius_km), bbox, polygon, or wkt for spatial filtering."
        )

    # Point-radius filters are approximated as a WKT polygon and sent via `wkt`
    if point is not None or radius_km is not None:
        if point is None or radius_km is None:
            raise ValueError(
                "Both 'point' and 'radius_km' must be provided for point-radius filtering."
            )

        lat, lon = point[1], point[0]
        params["wkt"] = _point_radius_to_wkt(lat=lat, lon=lon, radius_km=radius_km)
    elif bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        params["wkt"] = _bbox_to_wkt(min_lon, min_lat, max_lon, max_lat)
    elif polygon is not None:
        params["wkt"] = polygon.wkt
    elif wkt is not None:
        params["wkt"] = wkt

    if extra_params:
        # Do not let extra_params override core pagination params
        for key, value in extra_params.items():
            if key in {"page", "pageSize"}:
                LOGGER.warning(
                    "Ignoring extra param '%s' as it is controlled internally.", key
                )
                continue
            params[key] = value

    return params


async def _fetch_occurrence_page(
    session: aiohttp.ClientSession,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Fetch a single page of occurrences from the NBN Atlas records API.
    """
    url = f"{RECORDS_BASE_URL}{OCCURRENCE_SEARCH_PATH}"
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_occurrences_async(
    scientific_name: Optional[str] = None,
    taxon_id: Optional[str] = None,
    query: Optional[str] = None,
    page_size: int = 1000,
    max_records: Optional[int] = None,
    point: Optional[Tuple[float, float]] = None,
    radius_km: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    polygon: Optional[BaseGeometry] = None,
    wkt: Optional[str] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Asynchronously fetch one or more pages of occurrence records and return a DataFrame.
    """
    params = _build_query(
        scientific_name=scientific_name,
        taxon_id=taxon_id,
        query=query,
        page_size=page_size,
        point=point,
        radius_km=radius_km,
        bbox=bbox,
        polygon=polygon,
        wkt=wkt,
        extra_params=extra_params,
    )

    all_records: list[Dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        while True:
            LOGGER.info("Requesting NBN occurrences page %s", params["page"])
            payload = await _fetch_occurrence_page(session, params)

            # NBN / ALA style payload: occurrences under 'occurrences' key
            records: Iterable[Mapping[str, Any]] = payload.get("occurrences", [])
            if not records:
                break

            for rec in records:
                all_records.append(dict(rec))
                if max_records is not None and len(all_records) >= max_records:
                    break

            if max_records is not None and len(all_records) >= max_records:
                break

            total_records = payload.get("totalRecords")
            if total_records is not None and len(all_records) >= total_records:
                break

            # Increment page number and continue
            params = dict(params)
            params["page"] = params.get("page", 1) + 1

    if not all_records:
        LOGGER.warning("No occurrence records returned from NBN Atlas.")
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def _fetch_occurrences_sync(**kwargs: Any) -> pd.DataFrame:
    """Run the async NBN fetch from sync code, including notebook event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_fetch_occurrences_async(**kwargs))

    # Jupyter already owns the current thread's event loop. Run the coroutine in
    # a short-lived thread with its own loop so the public sync API remains usable.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(_fetch_occurrences_async(**kwargs)))
        return future.result()


def fetch_occurrences_from_nbn(
    scientific_name: Optional[str] = None,
    taxon_id: Optional[str] = None,
    query: Optional[str] = None,
    page_size: int = 1000,
    max_records: Optional[int] = None,
    polygon: Optional[BaseGeometry] = None,
    extra_params: Optional[Mapping[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch occurrence data from the NBN Atlas API and return a GeoDataFrame.

    Parameters
    ----------
    scientific_name:
        Scientific name to filter by (e.g. ``\"Pipistrellus pipistrellus\"``).
    taxon_id:
        Optional LSID / taxon identifier understood by NBN Atlas.
    query:
        Optional free-form query string in NBN / ALA query syntax.
        If provided alongside `scientific_name` or `taxon_id`, all
        conditions are combined with logical AND.
    page_size:
        Number of records to request per page (default 1000).
    max_records:
        Optional hard cap on the total number of records to download.
        If ``None``, all available records are fetched.
    polygon:
        Optional shapely geometry (typically a ``Polygon``) used for spatial
        filtering. The geometry is converted to WKT and sent via the API's
        ``wkt`` parameter. If omitted, no spatial filter is applied.
    extra_params:
        Optional mapping of additional query parameters to include in the
        request (e.g. spatial filters, date ranges). See NBN Atlas API
        documentation for details.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with occurrence attributes and point geometries in
        EPSG:4326. If no records are returned, an empty GeoDataFrame with
        no geometry is returned.
    """
    df = _fetch_occurrences_sync(
        scientific_name=scientific_name,
        taxon_id=taxon_id,
        query=query,
        page_size=page_size,
        max_records=max_records,
        polygon=polygon,
        extra_params=extra_params,
    )

    if df.empty:
        return gpd.GeoDataFrame()

    # Determine latitude/longitude column names (NBN typically uses decimalLatitude/decimalLongitude)
    lat_candidates = ["decimalLatitude", "latitude", "lat"]
    lon_candidates = ["decimalLongitude", "longitude", "lon"]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find latitude/longitude columns in NBN Atlas response. "
            "Expected one of "
            f"{lat_candidates} for latitude and {lon_candidates} for longitude."
        )

    # Drop records without valid coordinates
    df = df.dropna(subset=[lat_col, lon_col])

    # Cast coordinate columns to float explicitly for geometry creation
    lon_series = df[lon_col].astype(float)
    lat_series = df[lat_col].astype(float)

    geometry = gpd.points_from_xy(lon_series, lat_series)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


