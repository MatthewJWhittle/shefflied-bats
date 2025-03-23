import pytest
import numpy as np

from data_prep.generate_evs.ingestion.geo_utils import BoxTiler


@pytest.fixture
def bbox():
    return (0.1, 1.2, 93.3, 100.1)

def test_pad_and_align(bbox):
    result = BoxTiler(tile_size=(10, 10), origin=(0, 0)).pad_and_align(bbox)
    assert result == (0, 0, 100, 110)

    result_2 = BoxTiler(tile_size=(50, 50), origin=(5, 5)).pad_and_align(bbox)
    assert result_2 == (-45, -45, 105, 105)


def test_aligning_many_tiles():
    tiler = BoxTiler(tile_size=(100, 100), origin=(0, 0))

    n_boxes = 100
    seed = 42
    np.random.seed(seed)
    # generate random bounding boxes that are not aligned 
    boxes = np.random.rand(n_boxes, 4) * 100
    boxes[:, 2] += 10 * np.random.rand(n_boxes)
    boxes[:, 3] += 10 * np.random.rand(n_boxes)

    for box in boxes:
        result = tiler.pad_and_align(box)
        xmin, ymin, xmax, ymax = result
        # check that the aligned box is always enclosing the original box
        assert xmin <= box[0]
        assert ymin <= box[1]
        assert xmax >= box[2]
        assert ymax >= box[3]



def test_tile_bbox(bbox):

    result = BoxTiler(tile_size=(50, 50), origin=(0, 0)).tile_bbox(bbox)
    xmins = np.array([box[0] for box in result])
    ymins = np.array([box[1] for box in result])
    xmaxs = np.array([box[2] for box in result])
    ymaxs = np.array([box[3] for box in result])

    assert xmins.min() == 0
    assert ymins.min() == 0
    assert xmaxs.max() == 100
    assert ymaxs.max() == 150

    # check they are all multiples of 50
    assert np.all(xmins % 50 == 0)
    assert np.all(ymins % 50 == 0)
    assert np.all(xmaxs % 50 == 0)
    assert np.all(ymaxs % 50 == 0)





def test_align_big_box():
    bbox = (412392.528344397, 447395.6376247584, 414044.85501631664, 449042.5684995359)
    result = BoxTiler(tile_size=(100, 100), origin=(0, 0)).pad_and_align(bbox)

