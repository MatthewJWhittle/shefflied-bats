"""Unit tests for the tilearray WCS adapter (no live network)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from tilearray import WCSConfig, compute_with_policy
from tilearray.fetch import AdaptiveConcurrencyGate, reset_adaptive_gates
from tilearray.types import TileResponse

from sdm.data.wcs_tilearray import (
    EA_DTM_COVERAGE_ID,
    EA_DTM_WCS_ENDPOINT,
    _clamp_concurrency,
    ea_wcs_config,
    fetch_wcs_dataarray,
    fetch_wcs_dataset,
)


def _geotiff_bytes(
    *,
    width: int,
    height: int,
    value: float = 12.5,
    transform=None,
) -> bytes:
    """Minimal in-memory GeoTIFF for stubbed WCS tile responses."""
    if transform is None:
        transform = from_origin(0.0, float(height), 1.0, 1.0)
    buf = BytesIO()
    data = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return buf.getvalue()


def test_clamp_concurrency_keeps_aimd_ordered() -> None:
    clamped = _clamp_concurrency(4)
    assert clamped["max_concurrent_requests"] == 4
    assert (
        clamped["min_concurrent_requests"]
        <= clamped["initial_concurrent_requests"]
        <= clamped["max_concurrent_requests"]
    )
    # Gate construction must succeed (this is what live max=4 crashed on).
    AdaptiveConcurrencyGate(
        initial=clamped["initial_concurrent_requests"],
        minimum=clamped["min_concurrent_requests"],
        maximum=clamped["max_concurrent_requests"],
        additive_increase=1,
        multiplicative_decrease=0.75,
    )


def test_ea_wcs_config_builds_real_config_resolution_tuple() -> None:
    """Unmocked WCSConfig: resolution is (x, y); max=4 does not trip AIMD."""
    reset_adaptive_gates()
    cfg = ea_wcs_config(
        EA_DTM_WCS_ENDPOINT,
        EA_DTM_COVERAGE_ID,
        chunk_size=(64, 64),
        resolution_m=10.0,
        max_concurrent=4,
    )
    assert isinstance(cfg, WCSConfig)
    assert cfg.resolution == (10.0, 10.0)
    assert cfg.max_concurrent_requests == 4
    assert cfg.initial_concurrent_requests <= 4
    assert cfg.min_concurrent_requests <= cfg.initial_concurrent_requests


def test_create_array_compute_with_policy_one_chunk_stubbed_http() -> None:
    """Real config + create_array path; HTTP stubbed so CI catches gate/schema bugs."""
    reset_adaptive_gates()
    cfg = ea_wcs_config(
        EA_DTM_WCS_ENDPOINT,
        EA_DTM_COVERAGE_ID,
        chunk_size=(32, 32),
        resolution_m=10.0,
        max_concurrent=4,
    )
    # One chunk: 32×32 @ 10 m → 320 m box.
    bbox = (400000.0, 300000.0, 400320.0, 300320.0)
    tiff = _geotiff_bytes(
        width=32,
        height=32,
        value=42.0,
        transform=from_origin(400000.0, 300320.0, 10.0, 10.0),
    )

    def _fake_fetch(request, policy=None, progress=None):
        return TileResponse(
            data=tiff,
            content_type="image/tiff",
            status_code=200,
            headers={},
            url=request.url,
            success=True,
        )

    with patch("tilearray.array.fetch_tile", side_effect=_fake_fetch):
        from tilearray import create_array
        from tilearray.types import CRS

        lazy = create_array(
            cfg,
            bbox=bbox,
            crs=CRS.EPSG_27700,
            compute=False,
            dtype="float32",
        )
        out = compute_with_policy(lazy)

    assert out.ndim >= 2
    assert float(np.nanmean(np.asarray(out.values))) == pytest.approx(42.0, abs=0.1)


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
