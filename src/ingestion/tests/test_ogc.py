import pytest
import xarray as xr
import rioxarray as rxr
import numpy as np
from pathlib import Path

from src.ingestion.ogc import WCSDownloader

@pytest.fixture
def wcs_params() -> dict:
    """Return standard WCS parameters for testing."""
    return {
        "endpoint": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        "coverage_id": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        "request_tile_pixels" : (100, 100),
    }

@pytest.fixture
def small_bbox() -> tuple:
    """Return a small bounding box in British National Grid."""
    xmin, ymin = 422558, 391118
    size = 100  # 100m square area
    return xmin, ymin, xmin + size, ymin + size

@pytest.fixture
def dtm_downloader(wcs_params):
    """Return a WCSDownloader instance for testing."""
    return WCSDownloader(**wcs_params)

def test_downloader_initialization(wcs_params):
    """Test basic downloader initialization."""
    downloader = WCSDownloader(**wcs_params)
    assert downloader.endpoint == wcs_params["endpoint"]
    assert downloader.coverage_id == wcs_params["coverage_id"]
    assert isinstance(downloader.axis_labels, list)
    assert isinstance(downloader.native_crs, str)

@pytest.mark.asyncio
async def test_memory_download(dtm_downloader, small_bbox):
    """Test basic download functionality using memory storage."""
    result = await dtm_downloader.get_coverage(
        bbox=small_bbox,
        resolution=10.0,
    )
    
    assert isinstance(result, xr.Dataset)
    assert result.dims["x"] > 0 and result.dims["y"] > 0
    assert not np.all(np.isnan(result[dtm_downloader.coverage_id].values))

    result_box = result[dtm_downloader.coverage_id].rio.bounds()
    assert result_box[0] <= small_bbox[0]
    assert result_box[1] <= small_bbox[1]
    assert result_box[2] >= small_bbox[2]
    assert result_box[3] >= small_bbox[3]


@pytest.mark.asyncio
async def test_download_resolution(dtm_downloader, small_bbox):
    """Test download functionality with a specified resolution."""
    resolution = 10.0
    result = await dtm_downloader.get_coverage(
        bbox=small_bbox,
        resolution=10.0,
    )

    result_res = result.rio.resolution()
    assert result_res[0] == resolution
    assert result_res[1] == -resolution

