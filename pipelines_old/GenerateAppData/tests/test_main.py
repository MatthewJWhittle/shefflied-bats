import pytest
import numpy as np
import geopandas as gpd

from pipelines.GenerateAppData.main import align_to_grid, points_to_grid_squares

@pytest.fixture
def bbox():
    return (500001.2, 600000.3, 500000.3, 600000.1)

@pytest.fixture
def origin(bbox):
    return (0, 0)

@pytest.fixture
def n_points():
    return 100

@pytest.fixture
def points(bbox, n_points):
    xy = np.random.random((n_points, 2)) * bbox[2]
    return {
        "x" : xy[:, 0],
        "y" : xy[:, 1]
    }

@pytest.fixture
def grid_resolution():
    return 250

@pytest.fixture
def points_series(points):
    return gpd.GeoSeries(gpd.points_from_xy(points["x"], points["y"]))

def test_align_to_grid(bbox, points, grid_resolution, origin):
    x, y = points["x"], points["y"]
    n_points = len(x)
    x_aligned, y_aligned = align_to_grid(x = x, y =y, origin = origin, resolution=grid_resolution)
    assert len(x_aligned) == n_points
    assert len(y_aligned) == n_points
    assert all([x_ % grid_resolution == 0 for x_ in x_aligned])
    assert all([y_ % grid_resolution == 0 for y_ in y_aligned])
    # All aligned points should be less than or equal to the original points
    assert (np.array(x_aligned) <= x).all()
    assert (np.array(y_aligned) <= y).all()

    # All points should be within the grid resolution of the original points
    assert (np.abs(np.array(x_aligned) - x) <= grid_resolution).all()
    assert (np.abs(np.array(y_aligned) - y) <= grid_resolution).all()


def test_points_to_grid_squares(points_series, grid_resolution, origin):
    grid_squares = points_to_grid_squares(points_series, grid_resolution, origin=origin)
    assert len(grid_squares) == len(points_series)

    points_in_squares = points_series.within(grid_squares.unary_union)
    assert points_in_squares.all()

    # assert no geometries are null
    assert not grid_squares.isnull().any()


def test_points_to_grid_squares_subset(points_series, grid_resolution, origin):
    choices = np.random.choice(len(points_series), 50, replace=False)
    points_series = points_series[choices]
    grid_squares = points_to_grid_squares(points_series, grid_resolution, origin=origin)
    assert len(grid_squares) == len(points_series)

    points_in_squares = points_series.within(grid_squares.unary_union)
    assert points_in_squares.all()

    # assert no geometries are null
    assert not grid_squares.isnull().any()

