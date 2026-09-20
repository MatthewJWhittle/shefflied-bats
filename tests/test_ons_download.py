"""Tests for ONS counties / unitary authorities live download client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from requests.exceptions import HTTPError

from sdm.data.ons_download import (
    DEFAULT_CACHE_FILE,
    DEFAULT_YORKSHIRE_COUNTY_NAMES,
    LEGACY_MANUAL_FILE,
    ONSBoundariesClient,
    QUERY_URL,
    SKIPTON_COUNTY_NAME,
    YORKSHIRE_SMOKE_COUNTIES,
    counties_where_clause,
    county_name_column,
    resolve_counties_file,
)
from sdm.data.spatial import create_boundary


def _sample_geojson(
    names: tuple[str, ...] = ("Sheffield", "North Yorkshire"),
    name_field: str = "CTYUA24NM",
) -> dict:
    features = []
    for index, name in enumerate(names):
        offset = index * 2
        features.append(
            {
                "type": "Feature",
                "properties": {name_field: name, f"{name_field[:6]}CD": f"E00{index}"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [offset, offset],
                            [offset + 1, offset],
                            [offset + 1, offset + 1],
                            [offset, offset + 1],
                            [offset, offset],
                        ]
                    ],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }


def test_county_name_column_supports_2023_and_2024_vintages() -> None:
    gdf_2024 = gpd.GeoDataFrame({"CTYUA24NM": ["Sheffield"], "geometry": [box(0, 0, 1, 1)]})
    gdf_2023 = gpd.GeoDataFrame({"CTYUA23NM": ["Sheffield"], "geometry": [box(0, 0, 1, 1)]})

    assert county_name_column(gdf_2024) == "CTYUA24NM"
    assert county_name_column(gdf_2023) == "CTYUA23NM"


def test_query_geojson_paginates_when_transfer_limit_exceeded() -> None:
    session = MagicMock()
    page_one = MagicMock()
    page_one.json.return_value = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [0, 0]}}],
        "properties": {"exceededTransferLimit": True},
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }
    page_two = MagicMock()
    page_two.json.return_value = {
        "type": "FeatureCollection",
        "features": [],
        "properties": {"exceededTransferLimit": False},
    }
    session.get.side_effect = [page_one, page_two]

    client = ONSBoundariesClient(session=session)
    payload = client.query_geojson()

    assert len(payload["features"]) == 1
    assert session.get.call_count == 2


def test_download_counties_geojson_writes_cache(tmp_path: Path) -> None:
    output_path = tmp_path / "counties.geojson"
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = _sample_geojson()
    session.get.return_value = response

    client = ONSBoundariesClient(session=session)
    result = client.download_counties_geojson(output_path)

    assert result == output_path
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved["features"]) == 2


def test_resolve_counties_file_uses_legacy_manual_drop(tmp_path: Path) -> None:
    manual = tmp_path / "legacy.geojson"
    manual.write_text(json.dumps(_sample_geojson()), encoding="utf-8")

    with patch("sdm.data.ons_download.LEGACY_MANUAL_FILE", manual):
        resolved = resolve_counties_file(live_download=False)

    assert resolved == manual


def test_counties_where_clause_escapes_quotes() -> None:
    clause = counties_where_clause(["Kingston upon Hull, City of"])
    assert "Kingston upon Hull, City of" in clause
    assert clause.startswith("CTYUA24NM IN (")


def test_resolve_counties_file_falls_back_to_yorkshire_on_gateway_timeout(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "ons" / "counties.geojson"
    response = MagicMock()
    response.status_code = 504
    full_uk_error = HTTPError(response=response)
    fallback_path = cache_file

    with patch("sdm.data.ons_download.LEGACY_MANUAL_FILE", tmp_path / "missing.geojson"), patch.object(
        ONSBoundariesClient,
        "download_counties_geojson",
        side_effect=[full_uk_error, fallback_path],
    ) as download_mock:
        resolved = resolve_counties_file(
            cache_file=cache_file,
            live_download=True,
            county_names=DEFAULT_YORKSHIRE_COUNTY_NAMES,
        )

    assert resolved == fallback_path
    assert download_mock.call_count == 2
    fallback_call = download_mock.call_args_list[1]
    assert "where=" in str(fallback_call)
    assert fallback_call.kwargs.get("overwrite") is True


def test_resolve_counties_file_downloads_when_missing(tmp_path: Path) -> None:
    cache_file = tmp_path / "ons" / "counties.geojson"

    with patch("sdm.data.ons_download.LEGACY_MANUAL_FILE", tmp_path / "missing.geojson"), patch.object(
        ONSBoundariesClient,
        "download_counties_geojson",
        return_value=cache_file,
    ) as download_mock:
        resolved = resolve_counties_file(cache_file=cache_file, live_download=True)

    download_mock.assert_called_once()
    assert resolved == cache_file


def test_create_boundary_uses_manual_file_without_live_download(tmp_path: Path) -> None:
    manual = tmp_path / "manual.geojson"
    payload = _sample_geojson(names=("Sheffield",), name_field="CTYUA23NM")
    manual.write_text(json.dumps(payload), encoding="utf-8")

    mock_gdf = gpd.GeoDataFrame(
        {
            "CTYUA23NM": ["Sheffield"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:4326",
    )

    with patch("sdm.data.spatial.core.resolve_counties_file", return_value=manual):
        with patch("geopandas.read_file", return_value=mock_gdf):
            result = create_boundary(
                counties_file=manual,
                county_names=["Sheffield"],
                live_download=False,
            )

    assert len(result) == 1
    assert result.crs.to_string() == "EPSG:27700"


def test_yorkshire_smoke_counties_present_in_mocked_download(tmp_path: Path) -> None:
    cache_file = tmp_path / "counties.geojson"
    cache_file.write_text(
        json.dumps(_sample_geojson(names=YORKSHIRE_SMOKE_COUNTIES)),
        encoding="utf-8",
    )

    with patch("sdm.data.ons_download.LEGACY_MANUAL_FILE", tmp_path / "missing.geojson"), patch.object(
        ONSBoundariesClient,
        "download_counties_geojson",
        return_value=cache_file,
    ):
        resolved = resolve_counties_file(cache_file=cache_file, live_download=True)

    gdf = gpd.read_file(resolved)
    name_column = county_name_column(gdf)
    assert set(YORKSHIRE_SMOKE_COUNTIES).issubset(set(gdf[name_column]))
    assert SKIPTON_COUNTY_NAME in set(gdf[name_column])


@pytest.mark.integration
def test_skipton_yorkshire_live_query_smoke() -> None:
    """Lightweight live smoke: fetch two Yorkshire CTYUAs and clip North Yorkshire to Skipton AOI."""
    client = ONSBoundariesClient()
    where = " OR ".join(f"CTYUA24NM = '{name}'" for name in YORKSHIRE_SMOKE_COUNTIES)
    payload = client.query_geojson(where=where)
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:27700")

    skipton_aoi = box(397_100, 450_300, 400_100, 453_300)
    north_yorkshire = gdf[gdf["CTYUA24NM"] == SKIPTON_COUNTY_NAME]
    clipped = gpd.clip(north_yorkshire, gpd.GeoDataFrame(geometry=[skipton_aoi], crs="EPSG:27700"))

    assert len(gdf) == 2
    assert not clipped.empty
    assert clipped.geometry.iloc[0].intersects(skipton_aoi)


@pytest.mark.integration
def test_skipton_ons_smoke_artifact(tmp_path: Path) -> None:
    """Write a lightweight smoke artifact documenting the ONS endpoint and Skipton county."""
    artifact = {
        "county_for_skipton": SKIPTON_COUNTY_NAME,
        "smoke_counties": list(YORKSHIRE_SMOKE_COUNTIES),
        "feature_server_query": QUERY_URL,
        "default_cache_file": str(DEFAULT_CACHE_FILE),
        "legacy_manual_file": str(LEGACY_MANUAL_FILE),
        "auth_required": False,
    }
    artifact_path = tmp_path / "skipton-ons-boundary-smoke.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    client = ONSBoundariesClient()
    payload = client.query_geojson(where=f"CTYUA24NM = '{SKIPTON_COUNTY_NAME}'")
    artifact["north_yorkshire_feature_count"] = len(payload.get("features", []))
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    assert artifact["north_yorkshire_feature_count"] == 1
    assert artifact_path.exists()
