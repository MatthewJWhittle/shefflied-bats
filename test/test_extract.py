import pytest
import geopandas as gpd
from sdm.extract import download_feature_layer, ImageTileDownloader
from shapely.geometry import box
import xarray as xr


def test_get_image_tiles():
    bottom_left = (426268,385241)
    top_right = (438544,390389)
    bbox = box(bottom_left[0], bottom_left[1], top_right[0], top_right[1])
    url = "https://environment.data.gov.uk/image/rest/services/SURVEY/VegetationObjectModel/ImageServer"

    downloader = ImageTileDownloader(url, cache_folder="test/data/test_cache")
    downloader.clear_cache()

    image = downloader.download_image(
        bbox,
        target_resolution=10
    )

    assert isinstance(image, xr.Dataset)
    assert image.rio.resolution()[0] == 10
    assert image.rio.resolution()[1] == -10

def test_get_image_tiles_at_resolution():
    bottom_left = (430000,380000)
    top_right = (437000,387000)
    bbox = box(bottom_left[0], bottom_left[1], top_right[0], top_right[1])
    url_vom = "https://environment.data.gov.uk/image/rest/services/SURVEY/VegetationObjectModel/ImageServer"
    url_dtm = "https://environment.data.gov.uk/image/rest/services/SURVEY/LIDAR_Composite_2m_DTM_2022_Elevation/ImageServer"

    # This has a resolution of 1m on the image server
    downloader_vom = ImageTileDownloader(url_vom, cache_folder="test/data/test_cache")
    downloader_vom.clear_cache()

    # This has a resolution of 2m on the image server
    downloader_dtm = ImageTileDownloader(url_dtm, cache_folder="test/data/test_cache")
    downloader_dtm.clear_cache()


    image_vom = downloader_vom.download_tiles(
        bbox,
        target_resolution=10
    )

    image_dtm = downloader_dtm.download_tiles(
        bbox,
        target_resolution=10
    )

    # Bo0th should acheive the correct target resolution
    assert image_vom.rio.resolution()[0] == 10
    assert image_vom.rio.resolution()[1] == -10

    assert image_dtm.rio.resolution()[0] == 10
    assert image_dtm.rio.resolution()[1] == -10
