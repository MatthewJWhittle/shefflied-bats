"""Unit tests for the tilearray WCS adapter (no live network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from sdm.data.wcs_tilearray import (
    EA_DTM_COVERAGE_ID,
    EA_DTM_WCS_ENDPOINT,
    ea_wcs_config,
    fetch_wcs_dataarray,
    fetch_wcs_dataset,
)


def test_ea_wcs_config_uses_ea_dsp_preset() -> None:
    with patch("sdm.data.wcs_tilearray.WCSConfig.for_ea_dsp") as mock_for_ea:
        mock_for_ea.return_value = MagicMock(name="cfg")
        cfg = ea_wcs_config(
            EA_DTM_WCS_ENDPOINT,
            EA_DTM_COVERAGE_ID,
            chunk_size=(512, 512),
            resolution_m=10.0,
            max_concurrent=4,
        )
        assert cfg is mock_for_ea.return_value
        kwargs = mock_for_ea.call_args.kwargs
        assert kwargs["coverage_id"] == EA_DTM_COVERAGE_ID
        assert kwargs["chunk_size"] == (512, 512)
        assert kwargs["resolution"] == 10.0
        assert kwargs["max_concurrent_requests"] == 4


def test_fetch_wcs_dataset_names_coverage_and_computes() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    lazy = xr.DataArray(values, dims=("y", "x"), name=None)
    computed = lazy.rename(EA_DTM_COVERAGE_ID)

    with (
        patch("sdm.data.wcs_tilearray.ea_wcs_config", return_value=MagicMock()),
        patch("sdm.data.wcs_tilearray.create_array", return_value=lazy) as mock_create,
        patch(
            "sdm.data.wcs_tilearray.compute_with_policy", return_value=computed
        ) as mock_compute,
    ):
        ds = fetch_wcs_dataset(
            endpoint=EA_DTM_WCS_ENDPOINT,
            coverage_id=EA_DTM_COVERAGE_ID,
            bbox=(0.0, 0.0, 40.0, 30.0),
            resolution_m=10.0,
            chunk_size=(1024, 1024),
            max_concurrent=3,
            compute=True,
        )
        mock_create.assert_called_once()
        mock_compute.assert_called_once()
        assert EA_DTM_COVERAGE_ID in ds.data_vars
        np.testing.assert_array_equal(ds[EA_DTM_COVERAGE_ID].values, values)


def test_fetch_wcs_dataarray_can_stay_lazy() -> None:
    lazy = xr.DataArray(
        np.ones((2, 2), dtype=np.float32), dims=("y", "x"), name="tmp"
    )
    with (
        patch("sdm.data.wcs_tilearray.ea_wcs_config", return_value=MagicMock()),
        patch("sdm.data.wcs_tilearray.create_array", return_value=lazy),
        patch("sdm.data.wcs_tilearray.compute_with_policy") as mock_compute,
    ):
        out = fetch_wcs_dataarray(
            endpoint=EA_DTM_WCS_ENDPOINT,
            coverage_id=EA_DTM_COVERAGE_ID,
            bbox=(0.0, 0.0, 20.0, 20.0),
            resolution_m=10.0,
            compute=False,
        )
        mock_compute.assert_not_called()
        assert out is lazy
