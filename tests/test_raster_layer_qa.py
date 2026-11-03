"""Synthetic tests for distance-limited edge nodata fill (#74)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sdm.raster.layer_qa import (
    DEFAULT_NODATA_FILL_MAX_CELLS,
    fill_nodata_nearest,
    fill_xarray_nodata_nearest,
)


class TestFillNodataNearest:
    def test_fills_thin_edge_ring_from_nearest(self):
        # Valid interior, one-cell nodata ring (thin edge gap).
        data = np.full((5, 5), np.nan, dtype=np.float32)
        data[1:4, 1:4] = 10.0
        data[2, 2] = 20.0
        out = fill_nodata_nearest(data, max_search_distance_cells=2.0)
        assert np.isfinite(out).all()
        assert out[0, 0] == pytest.approx(10.0)
        assert out[2, 2] == pytest.approx(20.0)

    def test_leaves_far_sea_block_unfilled(self):
        # Land on the left; a wide nodata block on the right (sea / coverage hole).
        data = np.full((4, 10), np.nan, dtype=np.float32)
        data[:, :2] = 5.0
        out = fill_nodata_nearest(
            data, max_search_distance_cells=DEFAULT_NODATA_FILL_MAX_CELLS
        )
        assert np.isfinite(out[:, : 2 + int(DEFAULT_NODATA_FILL_MAX_CELLS)]).all()
        # Far columns beyond the search radius stay nodata.
        assert np.isnan(out[:, -1]).all()

    def test_all_valid_unchanged(self):
        data = np.arange(9, dtype=np.float32).reshape(3, 3)
        out = fill_nodata_nearest(data, max_search_distance_cells=3.0)
        np.testing.assert_array_equal(out, data)

    def test_all_nodata_stays_nodata(self):
        data = np.full((3, 3), np.nan, dtype=np.float32)
        out = fill_nodata_nearest(data, max_search_distance_cells=5.0)
        assert np.isnan(out).all()

    def test_dataarray_multiband_fills_each_band(self):
        band0 = np.full((3, 3), np.nan, dtype=np.float32)
        band0[1, 1] = 1.0
        band1 = np.full((3, 3), np.nan, dtype=np.float32)
        band1[1, 1] = 2.0
        da = xr.DataArray(
            np.stack([band0, band1]),
            dims=("band", "y", "x"),
            coords={"band": [1, 2], "y": [0, 1, 2], "x": [0, 1, 2]},
        )
        out = fill_xarray_nodata_nearest(da, max_search_distance_cells=2.0)
        assert np.isfinite(out.values).all()
        assert out.values[0].max() == pytest.approx(1.0)
        assert out.values[1].max() == pytest.approx(2.0)
