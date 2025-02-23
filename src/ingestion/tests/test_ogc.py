import pytest
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from src.ingestion.ogc import WCSDownloader

@pytest.fixture
def aoi() -> gpd.GeoDataFrame:
    """Return a simple AOI as a GeoDataFrame."""
    gdf = gpd.read_file("src/ingestion/tests/data/aoi.geojson")
    gdf = gdf.to_crs("EPSG:27700")
    return gdf

@pytest.fixture
def small_bbox() -> tuple:
    """Return a small bounding box."""
    xmin, ymin = 422558 , 391118
    size = 100
    return xmin, ymin, xmin + size, ymin + size
    


def test_wcs_downloader_init():
    """
    Test that the WCSDownloader class initializes properly.
    """
    base_url = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
    coverage_id = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
    downloader = WCSDownloader(
        endpoint=base_url,
        coverage_id=coverage_id,
    )
    assert downloader.endpoint == base_url
    assert downloader.coverage_id == coverage_id
    assert isinstance(downloader.axis_labels, list)
    assert len(downloader.axis_labels) > 0
    assert isinstance(downloader.native_crs, str)
    assert downloader.native_crs != ""


@pytest.fixture
def dtm_downloader():
    """
    Return a WCSDownloader instance for the DTM coverage.
    """
    base_url = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
    coverage_id = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"
    return WCSDownloader(
        endpoint=base_url,
        coverage_id=coverage_id,
    )


@pytest.mark.asyncio
async def test_download_coverage(small_bbox, dtm_downloader):
    """Test that download_coverage returns an xarray.Dataset with the proper dimensions and attributes."""
    da = await dtm_downloader.get_coverage(small_bbox, resolution=10, tile_size=(50, 50))
    assert isinstance(da, xr.Dataset)
    assert da.dims["x"] > 0
    assert da.dims["y"] > 0
    assert "transform" in da.attrs
    assert "crs" in da.attrs
    assert "coverage_id" in da.attrs
    assert "source_url" in da.attrs
    assert da.attrs["source_url"] == dtm_downloader.endpoint


@pytest.fixture
def mock_wcs_response():
    """Create a mock GeoTIFF response."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    transform = from_bounds(0, 0, 20, 20, 2, 2)
    
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=2, width=2,
            count=1,
            dtype=np.float32,
            transform=transform,
            crs='EPSG:27700'
        ) as dataset:
            dataset.write(data, 1)
        return memfile.read()

@pytest.fixture
def mock_wcs_server(mock_wcs_response, monkeypatch):
    """Mock the WCS server responses."""
    async def mock_fetch_tile(*args, **kwargs):
        return mock_wcs_response
    
    monkeypatch.setattr(WCSDownloader, '_fetch_tile', mock_fetch_tile)
    
    def mock_fetch_description(*args, **kwargs):
        return ['x', 'y'], 'EPSG:27700'
    
    monkeypatch.setattr(WCSDownloader, '_fetch_coverage_description', mock_fetch_description)

@pytest.mark.asyncio
async def test_wcs_invalid_bbox(dtm_downloader):
    """Test that WCSDownloader raises an error for invalid bounding boxes."""
    invalid_bbox = (0, 0, 0)  # Invalid bbox with only 3 elements
    with pytest.raises(ValueError, match="bbox must be a tuple"):
        await dtm_downloader.get_coverage(invalid_bbox, resolution=100)

@pytest.mark.asyncio
async def test_wcs_invalid_resolution(dtm_downloader, small_bbox):
    """Test that WCSDownloader raises an error for invalid resolutions."""
    bbox = small_bbox
    invalid_resolution = -10  # Negative resolution
    with pytest.raises(ValueError, match="resolution must be positive"):
        await dtm_downloader.get_coverage(bbox, resolution=invalid_resolution)


@pytest.mark.asyncio
async def test_wcs_different_tile_sizes(mock_wcs_server, small_bbox):
    """Test that WCSDownloader handles different tile sizes correctly."""
    downloader = WCSDownloader(
        endpoint="http://mock-server/wcs",
        coverage_id="test_coverage"
    )
    
    bbox = small_bbox
    
    # Download with small tile size
    result_small_tiles = await downloader.get_coverage(bbox, resolution=10.0, tile_size=(5, 5))
    assert isinstance(result_small_tiles, xr.Dataset)
    
    # Download with large tile size
    result_large_tiles = await downloader.get_coverage(bbox, resolution=10.0, tile_size=(20, 20))
    assert isinstance(result_large_tiles, xr.Dataset)
    
    # Results should be identical
    xr.testing.assert_equal(result_small_tiles, result_large_tiles)

@pytest.mark.asyncio
async def test_wcs_invalid_tile_size(dtm_downloader):
    """Test that WCSDownloader raises an error for invalid tile sizes."""
    bbox = (0, 0, 10, 10)
    invalid_tile_size = (-10, 10)  # Negative tile width
    with pytest.raises(ValueError, match="tile_size must be positive"):
        await dtm_downloader.get_coverage(bbox, resolution=100, tile_size=invalid_tile_size)
