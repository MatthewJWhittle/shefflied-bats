"""Tests for OS Vector Map District live download client."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from sdm.data.os import generate_parquets, load_os_shps
from sdm.data.os_download import (
    OSDownloadsClient,
    SKIPTON_AOI_BBOX,
    ensure_os_vector_map_data,
    osgb_100km_tile,
    osgb_100km_tiles_for_bbox,
)


def _make_vmd_zip(tmp_path: Path, layer_prefix: str = "SD") -> bytes:
    """Build a minimal Vector Map District-like zip for mocked downloads."""
    buffer = io.BytesIO()
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buffer, "w") as archive:
        for layer in ("Building", "Water", "Woodland", "Road"):
            gdf = gpd.GeoDataFrame(
                {"id": [1], "geometry": [box(397500, 451500, 397600, 451600)]},
                crs="EPSG:27700",
            )
            if layer == "Road":
                gdf["CLASSIFICA"] = ["A Road"]
            shp_path = staging / f"{layer_prefix}_{layer}.shp"
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            for sidecar in staging.glob(f"{layer_prefix}_{layer}.*"):
                archive.write(
                    sidecar,
                    arcname=f"{layer_prefix}/data/{sidecar.name}",
                )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("easting", "northing", "expected"),
    [
        (398_600, 451_800, "SD"),
        (400_100, 451_800, "SE"),
    ],
)
def test_osgb_100km_tile(easting: float, northing: float, expected: str) -> None:
    assert osgb_100km_tile(easting, northing) == expected


def test_skipton_bbox_tiles() -> None:
    assert osgb_100km_tiles_for_bbox(*SKIPTON_AOI_BBOX) == ["SD", "SE"]


def test_list_downloads_uses_api_key_header() -> None:
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = [{"fileName": "vmdvec_sd.zip", "url": "http://example"}]
    session.get.return_value = response

    client = OSDownloadsClient(api_key="test-key", session=session)
    downloads = client.list_downloads("SD")

    assert downloads[0]["fileName"] == "vmdvec_sd.zip"
    session.get.assert_called_once()
    _, kwargs = session.get.call_args
    assert kwargs["headers"] == {"key": "test-key"}
    assert kwargs["params"]["area"] == "SD"


def test_download_and_extract_vector_map_tile(tmp_path: Path) -> None:
    zip_bytes = _make_vmd_zip(tmp_path, "SD")
    session = MagicMock()

    list_response = MagicMock()
    list_response.json.return_value = [
        {
            "fileName": "vmdvec_sd.zip",
            "url": "https://api.os.uk/downloads/v1/products/VectorMapDistrict/downloads?area=SD&redirect",
        }
    ]

    download_response = MagicMock()
    download_response.iter_content.return_value = [zip_bytes]

    session.get.side_effect = [list_response, download_response]

    client = OSDownloadsClient(session=session)
    zip_path = client.download_vector_map_district_tile("SD", tmp_path / "downloads")
    assert zip_path.exists()

    extract_dir = tmp_path / "SD"
    client.extract_shapefile_zip(zip_path, extract_dir)
    assert list(extract_dir.glob("**/*Building*.shp"))


def test_ensure_os_vector_map_data_downloads_missing_tiles(tmp_path: Path) -> None:
    zip_bytes = _make_vmd_zip(tmp_path, "SD")

    with patch.object(
        OSDownloadsClient,
        "list_downloads",
        return_value=[
            {
                "fileName": "vmdvec_sd.zip",
                "url": "https://example.test/download?redirect",
            }
        ],
    ), patch.object(
        OSDownloadsClient,
        "download_vector_map_district_tile",
        side_effect=lambda area, output_dir, **kwargs: (
            output_dir.mkdir(parents=True, exist_ok=True),
            (output_dir / "vmdvec_sd.zip").write_bytes(zip_bytes),
            output_dir / "vmdvec_sd.zip",
        )[2],
    ):
        tiles = ensure_os_vector_map_data(
            SKIPTON_AOI_BBOX,
            tmp_path,
            datasets=["Building", "Water", "Woodland", "Road"],
        )

    assert tiles == ["SD", "SE"]
    assert list(tmp_path.glob("**/*Road*.shp"))


def test_load_os_shps_uses_manual_drop_without_live_download(tmp_path: Path) -> None:
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    gdf = gpd.GeoDataFrame(
        {"id": [1], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:27700",
    )
    gdf.to_file(manual_dir / "Building.shp")

    with patch("sdm.data.os.ensure_os_vector_map_data") as ensure_mock:
        result = load_os_shps(
            ["Building"],
            dir=manual_dir,
            bbox=SKIPTON_AOI_BBOX,
            live_download=False,
        )

    ensure_mock.assert_not_called()
    assert len(result["Building"]) == 1


def test_generate_parquets_triggers_live_download_when_needed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    boundary = box(*SKIPTON_AOI_BBOX)
    zip_bytes = _make_vmd_zip(tmp_path, "SD")

    with patch.object(
        OSDownloadsClient,
        "list_downloads",
        return_value=[
            {
                "fileName": "vmdvec_sd.zip",
                "url": "https://example.test/download?redirect",
            }
        ],
    ), patch.object(
        OSDownloadsClient,
        "download_vector_map_district_tile",
        side_effect=lambda area, output_dir, **kwargs: (
            output_dir.mkdir(parents=True, exist_ok=True),
            (output_dir / f"vmdvec_{area.lower()}.zip").write_bytes(zip_bytes),
            output_dir / f"vmdvec_{area.lower()}.zip",
        )[2],
    ):
        paths = generate_parquets(
            ["Building", "Water", "Woodland", "Road"],
            dir=str(processed_dir),
            raw_dir=raw_dir,
            boundary=boundary,
            overwrite=True,
        )

    assert len(paths) == 4
    assert all(path.exists() for path in paths)


@pytest.mark.integration
def test_skipton_tile_resolution_smoke_artifact(tmp_path: Path) -> None:
    """Write a lightweight smoke artifact without downloading large SD/SE tiles."""
    artifact = {
        "aoi_bbox_osgb": SKIPTON_AOI_BBOX,
        "tiles": osgb_100km_tiles_for_bbox(*SKIPTON_AOI_BBOX),
        "api": "https://api.os.uk/downloads/v1/products/VectorMapDistrict/downloads",
        "auth_required_for_opendata": False,
    }
    artifact_path = tmp_path / "skipton-os-smoke.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))

    assert artifact["tiles"] == ["SD", "SE"]
    assert artifact_path.exists()
