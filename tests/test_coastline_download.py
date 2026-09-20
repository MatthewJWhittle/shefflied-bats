"""Tests for OS Boundary-Line coastline live download client."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from sdm.data.coastline_download import (
    DEFAULT_CACHE_DIR,
    HIGH_WATER_LAYER,
    LEGACY_BGS_GEOCOAST,
    MHW_CODE,
    WHITBY_SMOKE_BBOX,
    BoundaryLineClient,
    ensure_boundary_line_mhw_shapefile,
    fetch_osm_coastline_overpass,
    filter_mean_high_water,
    load_mean_high_water_polylines,
    resolve_coastline_source,
)


def _make_boundary_line_zip(tmp_path: Path) -> bytes:
    """Build a minimal Boundary-Line-like zip containing high_water_polyline."""
    buffer = io.BytesIO()
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    lines = [
        LineString([(480500, 510500), (481000, 511000)]),
        LineString([(481000, 511000), (481500, 511500)]),
        LineString([(1000, 1000), (2000, 2000)]),
    ]
    codes = [MHW_CODE, MHW_CODE, "0099"]
    gdf = gpd.GeoDataFrame(
        {"CODE": codes, "DESCRIPTIO": ["MHW"] * 3},
        geometry=lines,
        crs="EPSG:27700",
    )
    shp_path = staging / f"{HIGH_WATER_LAYER}.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    with zipfile.ZipFile(buffer, "w") as archive:
        for sidecar in staging.glob(f"{HIGH_WATER_LAYER}.*"):
            archive.write(
                sidecar,
                arcname=f"Data/GB/{sidecar.name}",
            )

    return buffer.getvalue()


def test_filter_mean_high_water_keeps_code_0071() -> None:
    gdf = gpd.GeoDataFrame(
        {"CODE": [MHW_CODE, "0099", 71]},
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3)]),
            LineString([(4, 4), (5, 5)]),
        ],
        crs="EPSG:27700",
    )
    filtered = filter_mean_high_water(gdf)
    assert len(filtered) == 2
    assert all(filtered["CODE"].astype(str).str.zfill(4) == MHW_CODE)


def test_load_mean_high_water_polylines_clips_to_bbox(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"CODE": [MHW_CODE, MHW_CODE]},
        geometry=[
            LineString([(480500, 510500), (481000, 511000)]),
            LineString([(1000, 1000), (2000, 2000)]),
        ],
        crs="EPSG:27700",
    )
    shp = tmp_path / f"{HIGH_WATER_LAYER}.shp"
    gdf.to_file(shp)

    clipped = load_mean_high_water_polylines(
        shp,
        target_crs="EPSG:27700",
        clip_bbox=WHITBY_SMOKE_BBOX,
    )
    assert len(clipped) == 1


def test_boundary_line_client_download_and_extract(tmp_path: Path) -> None:
    zip_bytes = _make_boundary_line_zip(tmp_path)
    session = MagicMock()

    list_response = MagicMock()
    list_response.json.return_value = [
        {
            "fileName": "bdline_essh_gb.zip",
            "url": "https://api.os.uk/downloads/v1/products/BoundaryLine/downloads?area=GB&redirect",
        }
    ]
    download_response = MagicMock()
    download_response.iter_content.return_value = [zip_bytes]
    session.get.side_effect = [list_response, download_response]

    client = BoundaryLineClient(session=session)
    zip_path = client.download_gb_shapefile_zip(tmp_path / "downloads")
    assert zip_path.exists()

    shp_path = client.extract_high_water_polyline(zip_path, tmp_path / "extract")
    assert shp_path.name == f"{HIGH_WATER_LAYER}.shp"
    gdf = gpd.read_file(shp_path)
    assert len(gdf) == 3


def test_ensure_boundary_line_mhw_shapefile_downloads_when_missing(tmp_path: Path) -> None:
    zip_bytes = _make_boundary_line_zip(tmp_path)

    with patch.object(
        BoundaryLineClient,
        "download_gb_shapefile_zip",
        side_effect=lambda output_dir, **kwargs: (
            output_dir.mkdir(parents=True, exist_ok=True),
            (output_dir / "bdline_essh_gb.zip").write_bytes(zip_bytes),
            output_dir / "bdline_essh_gb.zip",
        )[2],
    ):
        shp_path = ensure_boundary_line_mhw_shapefile(
            tmp_path,
            live_download=True,
        )

    assert shp_path.exists()
    assert shp_path.name == f"{HIGH_WATER_LAYER}.shp"


def test_resolve_coastline_source_uses_explicit_path(tmp_path: Path) -> None:
    shp = tmp_path / "custom.shp"
    gdf = gpd.GeoDataFrame(
        {"CODE": [MHW_CODE]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:27700",
    )
    gdf.to_file(shp)

    path, kind = resolve_coastline_source(shp, live_download=False)
    assert path == shp
    assert kind == "boundary_line"


def test_resolve_coastline_source_no_live_download_raises_without_cache(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_coastline_source(
            cache_dir=tmp_path,
            live_download=False,
        )


def test_resolve_coastline_source_falls_back_to_bgs_when_offline(
    tmp_path: Path,
) -> None:
    with patch(
        "sdm.data.coastline_download.LEGACY_BGS_GEOCOAST",
        tmp_path / "geocoast.shp",
    ):
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:27700")
        gdf.to_file(tmp_path / "geocoast.shp")

        path, kind = resolve_coastline_source(cache_dir=tmp_path, live_download=False)
        assert kind == "bgs_geocoast"
        assert path.name == "geocoast.shp"


def test_fetch_osm_coastline_overpass_parses_response() -> None:
    payload = {
        "elements": [
            {
                "type": "way",
                "geometry": [
                    {"lon": -0.5, "lat": 54.4},
                    {"lon": -0.4, "lat": 54.5},
                ],
            }
        ]
    }
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    session.post.return_value = response

    gdf = fetch_osm_coastline_overpass(
        WHITBY_SMOKE_BBOX,
        session=session,
    )
    assert len(gdf) == 1
    assert gdf.crs.to_string() == "EPSG:27700"


@pytest.mark.integration
def test_whitby_coastline_resolution_smoke_artifact(tmp_path: Path) -> None:
    """Write a lightweight smoke artifact documenting Whitby AOI + Boundary-Line API."""
    artifact = {
        "aoi_bbox_osgb": WHITBY_SMOKE_BBOX,
        "product_id": "BoundaryLine",
        "layer": HIGH_WATER_LAYER,
        "mhw_code": MHW_CODE,
        "api_base": "https://api.os.uk/downloads/v1",
        "auth_required_for_opendata": False,
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "manual_bgs_fallback": str(LEGACY_BGS_GEOCOAST),
    }
    artifact_path = tmp_path / "whitby-coastline-smoke.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    assert artifact_path.exists()
