"""Unit tests for climate loader geometry clipping and reprojection."""

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from shapely.geometry import box

from sdm.data.climate import reproject_climate_datasets
from sdm.data.loaders.climate import ClimateData


@pytest.fixture
def sample_boundary():
    geometry = box(0, 0, 1000, 1000)
    return gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:27700")


@pytest.fixture
def cached_climate_raster(tmp_path):
    data = xr.DataArray(
        np.ones((1, 10, 10)),
        dims=["band", "y", "x"],
        coords={"band": [1], "y": range(10), "x": range(10)},
    )
    data = data.rio.write_crs("EPSG:27700")
    data = data.rio.write_transform(Affine(100, 0, 0, 0, -100, 1000))

    cache_path = tmp_path / "tavg.tif"
    data.rio.to_raster(cache_path)
    return tmp_path


def test_get_dataset_clips_with_shapely_geometries(cached_climate_raster, sample_boundary):
    """Clip accepts a list of Shapely geometries, not a GeoSeries FeatureCollection."""
    climate = ClimateData(cache_folder=cached_climate_raster)

    result = climate.get_dataset("tavg", aoi=sample_boundary)

    assert isinstance(result, xr.DataArray)
    assert result.rio.crs == sample_boundary.crs


@patch("sdm.data.climate.reproject_data")
def test_reproject_climate_datasets_uses_two_step_reproject(mock_reproject_data):
    """Reprojection delegates to reproject_data instead of passing transform+resolution together."""
    source = xr.DataArray(
        np.ones((1, 5, 5)),
        dims=["band", "y", "x"],
        coords={"band": [1], "y": range(5), "x": range(5)},
    ).rio.write_crs("EPSG:4326")
    mock_reproject_data.return_value = source

    target_transform = Affine(100, 0, 0, 0, -100, 1000)
    result = reproject_climate_datasets(
        datasets={"tavg": source},
        target_crs="EPSG:27700",
        target_transform=target_transform,
        target_resolution=100,
    )

    mock_reproject_data.assert_called_once_with(
        source,
        crs="EPSG:27700",
        transform=target_transform,
        resolution=100,
    )
    assert "tavg" in result
