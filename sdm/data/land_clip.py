"""Clip study-area boundaries to land using mean high water (MHW).

Root cause addressed (#78): a uniform buffer around coastal counties expands the
modelling grid into the sea and Humber estuary. Sea is the *seaward* half of an
MHW coastal band (split by the coastline), not the landward half — so inland
edge buffers are kept while seaward lobes are removed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, split, unary_union

from sdm.data.coastline_download import (
    coastline_source_is_polyline,
    load_mean_high_water_polylines,
    resolve_coastline_source,
)

logger = logging.getLogger(__name__)

# Extra margin beyond the study buffer so the sea mask fully covers the lobe.
DEFAULT_SEA_BUFFER_MARGIN_M = 1_000.0


def _as_linestrings(geom: BaseGeometry) -> list[LineString]:
    """Normalize a (multi)line geometry to a list of LineStrings."""
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if not line.is_empty]
    merged = linemerge(geom)
    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return [line for line in merged.geoms if not line.is_empty]
    return []


def _extend_line_across_extent(line: LineString, extent: BaseGeometry) -> LineString:
    """Extend a line beyond ``extent`` so ``split(extent, line)`` bisects it."""
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    minx, miny, maxx, maxy = extent.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5 + 1.0

    def _extend(origin: tuple[float, float], direction: tuple[float, float]) -> tuple[float, float]:
        dx, dy = direction
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return origin
        scale = diag / length
        return (origin[0] + dx * scale, origin[1] + dy * scale)

    start = coords[0]
    start_dir = (coords[0][0] - coords[1][0], coords[0][1] - coords[1][1])
    end = coords[-1]
    end_dir = (coords[-1][0] - coords[-2][0], coords[-1][1] - coords[-2][1])
    return LineString([_extend(start, start_dir), *coords, _extend(end, end_dir)])


def _cutting_coast_lines(coast: BaseGeometry, extent: BaseGeometry) -> BaseGeometry:
    """Coast lines extended to cut ``extent`` for a clean polygon split."""
    lines = [_extend_line_across_extent(line, extent) for line in _as_linestrings(coast)]
    if not lines:
        raise ValueError("MHW coastline has no line geometries to cut with")
    if len(lines) == 1:
        return lines[0]
    return MultiLineString(lines)


def approximate_sea_from_mhw(
    land_gdf: gpd.GeoDataFrame,
    mhw_gdf: gpd.GeoDataFrame,
    *,
    sea_buffer_m: float,
) -> BaseGeometry:
    """Approximate sea near the AOI as the seaward half of an MHW coastal band.

    Extends MHW polylines across the AOI extent, splits that extent, and keeps
    pieces that do not contain land. Intersecting with an MHW buffer limits the
    mask to the coastal strip so distant inland areas stay untouched.
    """
    if land_gdf.empty or land_gdf.geometry.is_empty.all():
        raise ValueError("Land boundary has no geometries")
    if sea_buffer_m <= 0:
        raise ValueError(f"sea_buffer_m must be positive, got {sea_buffer_m}")
    if mhw_gdf.empty or mhw_gdf.geometry.is_empty.all():
        # Inland AOI: no coast in range — return empty sea (caller keeps boundary).
        return GeometryCollection()

    land = land_gdf.union_all()
    coast = unary_union(list(mhw_gdf.geometry))
    extent = box(*land.buffer(sea_buffer_m).bounds)
    cutter = _cutting_coast_lines(coast, extent)

    try:
        split_result = split(extent, cutter)
        pieces: Iterable[BaseGeometry] = list(split_result.geoms)
    except Exception:
        logger.warning(
            "Could not split extent by MHW; falling back to coastal band minus land"
        )
        return coast.buffer(sea_buffer_m).difference(land)

    pieces = [piece for piece in pieces if not piece.is_empty]
    if len(pieces) < 2:
        return coast.buffer(sea_buffer_m).difference(land)

    land_rep = land.representative_point()
    seaward_parts: list[BaseGeometry] = []
    for piece in pieces:
        if piece.contains(land_rep) or piece.intersects(land_rep):
            continue
        # Keep the seaward polygon, restricted to the coastal strip.
        seaward_parts.append(piece.intersection(coast.buffer(sea_buffer_m)))

    if not seaward_parts:
        for piece in pieces:
            outside = piece.difference(land)
            if outside.is_empty:
                continue
            if outside.area >= 0.5 * max(piece.area, 1e-9):
                seaward_parts.append(outside.intersection(coast.buffer(sea_buffer_m)))

    sea = unary_union([part for part in seaward_parts if not part.is_empty])
    if sea.is_empty:
        return coast.buffer(sea_buffer_m).difference(land)
    return sea.difference(land) if sea.intersects(land) else sea


def clip_boundary_to_land(
    boundary_gdf: gpd.GeoDataFrame,
    mhw_gdf: gpd.GeoDataFrame,
    *,
    land_gdf: Optional[gpd.GeoDataFrame] = None,
    sea_buffer_m: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Remove seaward parts of a (possibly buffered) study boundary.

    Parameters
    ----------
    boundary_gdf:
        Boundary to clip (often land buffered for neighbourhood/distance edges).
    mhw_gdf:
        Mean high water polylines (OS Boundary-Line CODE 0071 or equivalent).
    land_gdf:
        Unbuffered land core used to build the sea mask. Defaults to
        ``boundary_gdf`` (correct when the boundary is already land-only).
    sea_buffer_m:
        Buffer applied to MHW when building the sea mask. Defaults to a value
        large enough to cover how far ``boundary_gdf`` extends beyond ``land_gdf``.
    """
    if boundary_gdf.empty:
        raise ValueError("Boundary has no geometries")

    land_core = land_gdf if land_gdf is not None else boundary_gdf
    if land_core.crs is None:
        raise ValueError("land_gdf/boundary must have a CRS set")

    mhw = mhw_gdf
    if mhw.crs is None:
        raise ValueError("MHW coastline must have a CRS set")
    if mhw.crs != land_core.crs:
        mhw = mhw.to_crs(land_core.crs)

    if sea_buffer_m is None:
        land_bounds = land_core.total_bounds
        boundary_bounds = boundary_gdf.to_crs(land_core.crs).total_bounds
        extent_pad = max(
            abs(boundary_bounds[0] - land_bounds[0]),
            abs(boundary_bounds[1] - land_bounds[1]),
            abs(boundary_bounds[2] - land_bounds[2]),
            abs(boundary_bounds[3] - land_bounds[3]),
            0.0,
        )
        sea_buffer_m = max(extent_pad + DEFAULT_SEA_BUFFER_MARGIN_M, DEFAULT_SEA_BUFFER_MARGIN_M)

    sea = approximate_sea_from_mhw(land_core, mhw, sea_buffer_m=sea_buffer_m)
    if sea.is_empty:
        logger.info("Sea mask empty (no MHW in range or no seaward lobe); boundary unchanged")
        return boundary_gdf.copy()

    working = boundary_gdf.to_crs(land_core.crs).copy()
    working["geometry"] = working.geometry.difference(sea)
    working["geometry"] = working.geometry.buffer(0)
    working = working[~working.geometry.is_empty].copy()
    if working.empty:
        raise ValueError(
            "Land clip removed the entire boundary; check MHW alignment and CRS"
        )

    before_area = float(boundary_gdf.to_crs(land_core.crs).union_all().area)
    after_area = float(working.union_all().area)
    if before_area > 0:
        removed_frac = max(0.0, 1.0 - after_area / before_area)
        logger.info(
            "Clipped boundary to land (MHW): removed %.1f%% of buffered area",
            100.0 * removed_frac,
        )
    return working


def load_mhw_for_boundary(
    boundary_gdf: gpd.GeoDataFrame,
    *,
    coastline_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    live_download: bool = True,
    bbox_expand_m: float = 50_000.0,
) -> gpd.GeoDataFrame:
    """Resolve and load MHW polylines covering the boundary bbox.

    The GB Boundary-Line extract is large; features are clipped to the AOI
    bbox expanded by ``bbox_expand_m`` so land clip stays local.
    """
    from sdm.data.coastline_download import DEFAULT_CACHE_DIR

    boundary_osgb = boundary_gdf.to_crs("EPSG:27700")
    minx, miny, maxx, maxy = boundary_osgb.total_bounds
    bbox = (
        minx - bbox_expand_m,
        miny - bbox_expand_m,
        maxx + bbox_expand_m,
        maxy + bbox_expand_m,
    )
    path, kind = resolve_coastline_source(
        coastline_path,
        cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR,
        live_download=live_download,
        fallback_bbox=bbox,
    )
    if not coastline_source_is_polyline(kind, path):
        # BGS GeoCoast polygons: treat exterior rings as coast lines via boundary.
        gdf = gpd.read_file(path, bbox=bbox)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:27700")
        gdf = gdf.to_crs(boundary_gdf.crs)
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.boundary
        return gdf

    try:
        return load_mean_high_water_polylines(
            path,
            target_crs=(
                str(boundary_gdf.crs) if boundary_gdf.crs is not None else "EPSG:27700"
            ),
            clip_bbox=bbox,
        )
    except ValueError as exc:
        # Inland AOIs with no MHW in the expanded bbox: nothing to clip.
        if "No mean high water features" in str(exc):
            logger.info(
                "No MHW features within %.0f m of boundary; skipping land clip mask",
                bbox_expand_m,
            )
            return gpd.GeoDataFrame(geometry=[], crs=boundary_gdf.crs)
        raise


def load_boundary_land_clipped(
    filepath: Union[str, Path],
    *,
    coastline_path: Optional[Path] = None,
    buffer_distance: float = 0,
    target_crs: Union[str, int] = "EPSG:27700",
    cache_dir: Optional[Path] = None,
    live_download: bool = True,
    sea_buffer_m: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Load a boundary, optionally buffer, then clip seaward of MHW."""
    from sdm.utils.io import load_boundary

    land = load_boundary(filepath, buffer_distance=0, target_crs=target_crs)
    if buffer_distance > 0:
        buffered = land.copy()
        buffered["geometry"] = buffered.geometry.buffer(buffer_distance)
    else:
        buffered = land

    mhw = load_mhw_for_boundary(
        land,
        coastline_path=coastline_path,
        cache_dir=cache_dir,
        live_download=live_download,
    )
    resolved_sea_buffer = sea_buffer_m
    if resolved_sea_buffer is None and buffer_distance > 0:
        resolved_sea_buffer = buffer_distance + DEFAULT_SEA_BUFFER_MARGIN_M

    return clip_boundary_to_land(
        buffered,
        mhw,
        land_gdf=land,
        sea_buffer_m=resolved_sea_buffer,
    )
