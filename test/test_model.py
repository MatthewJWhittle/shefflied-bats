import pytest
from sdm.models import S2Dataset, SpatialTransformer
import numpy as np
import torch
import xarray as xr

@pytest.fixture
def dataset():
    train_dataset = S2Dataset()
    return train_dataset

    

def test_dataset(dataset):
    for i in range(0, 10):

        item = dataset[i]
        assert item[0].shape == (7, dataset.image_size[0], dataset.image_size[1])
        assert item[1].shape == (dataset.image_size[0], dataset.image_size[1])
        assert item[2].shape == (1,)


def test_dataset_normalisation(dataset):
    # Take a random sample from the dataset
    i = 43
    item = dataset[i]
    # Check that the input tensor is normalised. Round to deal with floating point errors
    assert item[0].max().round(decimals=3) <= 1
    assert item[0].min().round(decimals=3) >= 0

def test_dataset_no_nulls(dataset):
    # Take a random sample of 100 indices from the dataset
    indices = np.random.randint(0, len(dataset), 100)
    for i in indices:
        data = dataset[i]
        assert not np.isnan(data[0]).any()
        assert not np.isnan(data[1]).any()


def test_spatial_info(dataset):
    # Take a random sample from the dataset
    i = 43
    item = dataset[i]
    # Check that the spatial information is collected correctly
    assert dataset.spatial_info[i]["bbox"] == dataset.bbox_tile_list[i]
    assert dataset.spatial_info[i]["crs"] == dataset.crs


def test_spatial_transformer(dataset):
    i = 43
    item = dataset[i]
    # Create a spatial transformer
    transformer = SpatialTransformer(**dataset.spatial_info[i])
    # Create a fake prediction based upon the height and width of the input tensor
    prediction = torch.rand(item[0].shape[-2], item[0].shape[-1])
    # Transform the prediction
    transformed_prediction = transformer(prediction)
    # Check that the output has the same CRS as the input
    assert transformed_prediction.rio.crs == dataset.crs
    assert isinstance(transformed_prediction, xr.DataArray)


