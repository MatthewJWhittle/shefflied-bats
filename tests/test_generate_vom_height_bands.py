"""Generator wiring for VOM height bands (#121)."""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Polygon

from sdm.commands.data_preparation.environmental.generate_vom_height_bands import (
    generate_vom_height_bands,
)
from sdm.data.wcs_tilearray import EA_VOM_COVERAGE_ID
from sdm.raster.vom_height_bands import CELL_LAYER_STEMS, summarise_height_to_grid


@pytest.fixture
def sample_boundary():
    boundary_polygon = Polygon(
        [
            (420000, 380000),
            (421000, 380000),
            (421000, 381000),
            (420000, 381000),
            (420000, 380000),
        ]
    )
    return gpd.GeoDataFrame(
        {"id": [1], "geometry": [boundary_polygon]},
        crs="EPSG:27700",
    )


@pytest.fixture
def temp_boundary_file(tmp_path, sample_boundary):
    boundary_file = tmp_path / "test_boundary.geojson"
    sample_boundary.to_file(boundary_file)
    return boundary_file


def test_summarise_height_to_grid_matches_stems() -> None:
    h = np.full((100, 100), 8.0, dtype=np.float64)
    layers = summarise_height_to_grid(h)
    assert set(layers) == set(CELL_LAYER_STEMS)
    assert layers["woody_cover_share"][0, 0] == pytest.approx(1.0)


def test_generate_vom_height_bands_writes_geotiff(
    temp_boundary_file, tmp_path
) -> None:
    output_dir = tmp_path / "vom"
    # 200×200 at 1 m → two 100 m cells per side after trim.
    yy = np.linspace(381000.5, 380800.5, 200)
    xx = np.linspace(420000.5, 420200.5, 200)
    data = np.full((200, 200), 3.0, dtype=np.float32)
    data[100:, 100:] = 18.0
    data[0, 0] = 70.0  # capped
    vom = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": yy, "x": xx},
        name=EA_VOM_COVERAGE_ID,
    ).rio.write_crs("EPSG:27700")

    with patch(
        "sdm.commands.data_preparation.environmental.generate_vom_height_bands.fetch_wcs_dataarray",
        return_value=vom,
    ):
        result = generate_vom_height_bands(
            output_dir=output_dir,
            boundary_path=temp_boundary_file,
            wcs_download_resolution_m=1.0,
            summary_target_resolution_m=100,
            max_concurrent_downloads=1,
            verbose=False,
        )

    assert result.exists()
    notes = output_dir / "vom_height_bands_notes.json"
    assert notes.exists()
    with rxr.open_rasterio(result) as ds:
        assert ds.sizes["band"] >= len(CELL_LAYER_STEMS)
        # Woody neighbourhood adds at least one extra band.
        assert ds.sizes["band"] >= len(CELL_LAYER_STEMS) + 1
