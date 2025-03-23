from pathlib import Path

import pytest
import xarray as xr
import numpy as np

from generate_evs.ingestion.get_terrain_data import init_wcs_downloaders, main
from generate_evs.ingestion.ogc import WCSDownloader


@pytest.mark.asyncio
async def test_init_wcs_downloaders():
    """
    Test that the init_wcs_downloaders function creates WCS downloaders for each layer.
    """
    wcs_downloaders = init_wcs_downloaders()
    assert isinstance(wcs_downloaders, dict)
    dtm_downloader = wcs_downloaders["dtm"]
    dsm_downloader = wcs_downloaders["dsm"]

    assert dtm_downloader.native_crs is not None
    assert dsm_downloader.native_crs is not None

    bbox = 422558, 391118, 422658, 391218

    dtm_data = await dtm_downloader.get_coverage(bbox=bbox, resolution=10)
    dsm_data = await dsm_downloader.get_coverage(bbox=bbox, resolution=10)

    assert dtm_data is not None
    assert dsm_data is not None

    assert isinstance(dtm_data, xr.Dataset)
    assert isinstance(dsm_data, xr.Dataset)


@pytest.fixture
def params(tmp_path) -> dict:
    return {
        "boundary_path": "src/ingestion/tests/data/aoi.geojson",
        "output_dir": tmp_path,
    }


def test_main(params):
    main(**params, buffer_distance=0)
    output_dir = Path(params["output_dir"])
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.tif"))) > 0


@pytest.mark.asyncio
async def test_value_range():
    dtm_downloader = WCSDownloader(
        endpoint="https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        coverage_id="13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        fill_value=np.nan,
    )
    dataset = await dtm_downloader.get_coverage(
        bbox=(403273, 515629, 405059, 517603),
        resolution=1,
    )

    array = dataset[dtm_downloader.coverage_id]
    assert array.min() >= -10
    assert array.max() <= 1000


@pytest.mark.asyncio
async def test_value_range_use_disk():
    dtm_downloader = WCSDownloader(
        endpoint="https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        coverage_id="13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        fill_value=np.nan,
        use_temp_storage=True,
    )
    dataset = await dtm_downloader.get_coverage(
        bbox=(403273, 515629, 405059, 517603),
        resolution=1,
    )

    array = dataset[dtm_downloader.coverage_id]
    assert array.min() >= -10
    assert array.max() <= 1000
