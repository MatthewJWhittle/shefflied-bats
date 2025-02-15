from pathlib import Path

import pytest
import xarray as xr

from src.ingestion.get_terrain_data import init_wcs_downloaders, main


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

    assert isinstance(dtm_data, xr.DataArray)
    assert isinstance(dsm_data, xr.DataArray)


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
    

