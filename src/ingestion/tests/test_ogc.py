import pytest
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from src.ingestion.ogc import WCSDownloader

@pytest.fixture
def aoi() -> gpd.GeoDataFrame:
    """Return a simple AOI as a GeoDataFrame."""
    gdf = gpd.read_file("src/ingestion/tests/data/aoi.geojson")
    gdf = gdf.to_crs("EPSG:27700")
    return gdf


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
async def test_download_coverage(aoi, dtm_downloader):
    """
    Test that download_coverage returns an xarray.DataArray with the proper dimensions and attributes.
    """

    # Use a moderate resolution and max output size.
    bounds = tuple(aoi.total_bounds)
    da = await dtm_downloader.get_coverage(bounds, resolution=100, tile_size=(1000, 1000))
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("y", "x")
    # Check dimensions are non-zero.
    assert da.shape[0] > 0 and da.shape[1] > 0
    # Check that required attributes are present.
    assert "transform" in da.attrs
    assert "crs" in da.attrs
    assert "coverage_id" in da.attrs
    assert "source_url" in da.attrs

    assert da.attrs["source_url"] == dtm_downloader.endpoint   
