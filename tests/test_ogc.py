import pytest
import xarray as xr
import numpy as np

from sdm.data.terrain.core import WCSDownloader


@pytest.fixture
def wcs_test_params() -> dict:
    """Return standard WCS parameters for testing."""
    return {
        "endpoint": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        "coverage_id": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        "request_tile_pixels": (100, 100),
    }


@pytest.fixture
def test_bbox() -> tuple:
    """Return a small bounding box in British National Grid."""
    xmin, ymin = 422558, 391118
    size = 100  # 100m square area
    return xmin, ymin, xmin + size, ymin + size


@pytest.fixture
def test_downloader(wcs_test_params):
    """Return a WCSDownloader instance for testing."""
    return WCSDownloader(**wcs_test_params)


def test_downloader_initialization(wcs_test_params):
    """Test basic downloader initialization."""
    downloader = WCSDownloader(**wcs_test_params)
    assert downloader.endpoint == wcs_test_params["endpoint"]
    assert downloader.coverage_id == wcs_test_params["coverage_id"]
    # Check that the downloader was initialized with correct attributes
    assert hasattr(downloader, 'endpoint')
    assert hasattr(downloader, 'coverage_id')
    # Check tile dimensions
    assert hasattr(downloader, 'tile_width')
    assert hasattr(downloader, 'tile_height')


@pytest.mark.asyncio
async def test_memory_download(test_downloader, test_bbox):
    """Test basic download functionality using memory storage."""
    result = await test_downloader.get_coverage(
        bbox=test_bbox,
        resolution=10.0,
    )

    assert isinstance(result, xr.Dataset)
    assert result.sizes["x"] > 0 and result.sizes["y"] > 0
    assert not np.all(np.isnan(result[test_downloader.coverage_id].values))

    result_box = result[test_downloader.coverage_id].rio.bounds()
    assert result_box[0] <= test_bbox[0]
    assert result_box[1] <= test_bbox[1]
    assert result_box[2] >= test_bbox[2]
    assert result_box[3] >= test_bbox[3]


@pytest.mark.asyncio
async def test_download_resolution(test_downloader, test_bbox):
    """Test download functionality with a specified resolution."""
    resolution = 10.0
    result = await test_downloader.get_coverage(
        bbox=test_bbox,
        resolution=resolution,
    )

    result_res = result.rio.resolution()
    assert result_res[0] == resolution
    assert result_res[1] == -resolution


@pytest.mark.asyncio
async def test_download_to_temp_storage(test_bbox, wcs_test_params):
    """Test download functionality using temporary storage."""
    temp_downloader = WCSDownloader(
        endpoint=wcs_test_params["endpoint"],
        coverage_id=wcs_test_params["coverage_id"],
        request_tile_pixels=(10, 10),
        use_temp_storage=True
    )
    
    resolution = 10.0
    result = await temp_downloader.get_coverage(
        bbox=test_bbox,
        resolution=resolution,
    )

    assert isinstance(result, xr.Dataset)
    # assert it has returned a chunked array
    assert result.chunks is not None

    # check the resolution
    result_res = result.rio.resolution()
    assert result_res[0] == resolution
    assert result_res[1] == -resolution

    # check the bounding box
    result_box = result[temp_downloader.coverage_id].rio.bounds()
    assert result_box[0] <= test_bbox[0]
    assert result_box[1] <= test_bbox[1]
    assert result_box[2] >= test_bbox[2]
    assert result_box[3] >= test_bbox[3]
    


