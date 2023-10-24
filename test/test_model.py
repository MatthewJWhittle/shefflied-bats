import pytest
from sdm.models import S2Dataset

@pytest.fixture
def dataset():
    input_vars = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14
        ]
    n_input_bands = len(input_vars)

    train_dataset = S2Dataset(
        "data/processed/s2-lidar/s2-lidar-stack.tif",
        input_bands=input_vars,
        target_band=[15],
        tile_size=1024,
    )
    return train_dataset

def test_dataset(dataset):
    item = dataset[0]
    assert item[0].shape == (14, 1024, 1024)
    assert item[1].shape == (1024, 1024)
    assert item[2].shape == (1024, 1024)