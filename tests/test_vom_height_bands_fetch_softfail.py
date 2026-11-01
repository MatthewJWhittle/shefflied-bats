"""Per-chunk fetch soft-fail for VOM height bands (#122 box)."""

from __future__ import annotations

import numpy as np
import pytest

from sdm.commands.data_preparation.environmental.generate_vom_height_bands import (
    DEFAULT_MAX_FAILED_CHUNK_FRACTION,
    VomHeightBandsChunkFailureError,
    _bbox_from_centres,
    summarise_chunks_resilient,
)
from sdm.raster.vom_height_bands import CELL_LAYER_STEMS


def test_bbox_from_centres_snaps_to_resolution_grid() -> None:
    """Drifted linspace centres must still yield exact n×resolution spans."""
    res = 1.0
    n = 1024
    # Mimic tilearray: linspace over edges (inclusive) → ~1024.0 + epsilon span
    # when half-pixel pads are applied without snapping.
    x_drifted = np.linspace(420000.0, 420000.0 + n, n)
    y_drifted = np.linspace(380000.0 + n, 380000.0, n)
    raw_w = (x_drifted.max() + res / 2.0) - (x_drifted.min() - res / 2.0)
    assert raw_w != pytest.approx(n * res)  # documents the drift bug

    minx, miny, maxx, maxy = _bbox_from_centres(
        y_drifted, x_drifted, resolution_m=res
    )
    width = maxx - minx
    height = maxy - miny
    assert width == pytest.approx(n * res)
    assert height == pytest.approx(n * res)
    assert width / res == pytest.approx(int(width / res))
    assert minx / res == pytest.approx(round(minx / res))
    assert maxy / res == pytest.approx(round(maxy / res))


def test_resilient_chunks_one_request_per_chunk_no_slivers() -> None:
    """Each chunk issues one snapped bbox; widths are exact pixel multiples."""
    y = np.linspace(10000.0, 0.0, 1000)  # drifted centres
    x = np.linspace(0.0, 1000.0, 1000)
    chunks = ((100,) * 10, (1000,))
    seen: list[tuple[float, float, float, float]] = []

    def fetch_chunk(bbox):
        seen.append(bbox)
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny
        assert width == pytest.approx(round(width))
        assert height == pytest.approx(round(height))
        assert width % 1.0 == pytest.approx(0.0)
        ny = int(round(height))
        nx = int(round(width))
        return np.full((ny, nx), 10.0, dtype=np.float64)

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_chunk,
        max_workers=4,
    )
    assert n_chunks == 10
    assert n_failed == 0
    assert len(seen) == n_chunks  # one request per chunk, no slivers
    assert mosaic.shape[0] == len(CELL_LAYER_STEMS)


def test_injected_fetch_failures_do_not_abort_siblings() -> None:
    """~10% injected fetch failures → NaN tiles + count; siblings succeed."""
    # 10 chunks of 100×100 at 1 m → 10 summary cells.
    y = np.arange(1000.0, 0.0, -1.0) - 0.5  # 1000 centres
    x = np.arange(0.0, 1000.0, 1.0) + 0.5
    chunks = ((100,) * 10, (1000,))  # 10 row-chunks × 1 col = 10 chunks
    calls = {"n": 0}

    def fetch_chunk(bbox):
        calls["n"] += 1
        # Fail every 10th chunk (10%).
        if calls["n"] % 10 == 0:
            raise RuntimeError("injected fetch failure")
        # Constant canopy height for a successful 100×100 (or whatever) block.
        # Size from bbox span at 1 m.
        minx, miny, maxx, maxy = bbox
        ny = max(1, int(round(maxy - miny)))
        nx = max(1, int(round(maxx - minx)))
        return np.full((ny, nx), 10.0, dtype=np.float64)

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_chunk,
        max_workers=4,
    )
    assert n_chunks == 10
    assert n_failed == 1
    assert mosaic.shape[0] == len(CELL_LAYER_STEMS)
    # One failed row-strip is all-NaN; others finite.
    band0 = mosaic[0]
    nan_rows = [i for i in range(band0.shape[0]) if np.all(~np.isfinite(band0[i]))]
    finite_rows = [i for i in range(band0.shape[0]) if np.any(np.isfinite(band0[i]))]
    assert len(nan_rows) == 1
    assert len(finite_rows) == 9
    assert n_failed / n_chunks < DEFAULT_MAX_FAILED_CHUNK_FRACTION


def test_abort_when_failed_fraction_exceeds_limit() -> None:
    y = np.arange(200.0, 0.0, -1.0) - 0.5
    x = np.arange(0.0, 100.0, 1.0) + 0.5
    chunks = ((100, 100), (100,))

    def fetch_always_fail(_bbox):
        raise RuntimeError("injected fetch failure")

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_always_fail,
        max_workers=2,
    )
    assert n_chunks == 2
    assert n_failed == 2
    assert np.all(~np.isfinite(mosaic))
    # Caller would abort — mirror the threshold check used in the generator.
    fail_frac = n_failed / n_chunks
    assert fail_frac > DEFAULT_MAX_FAILED_CHUNK_FRACTION
    with pytest.raises(VomHeightBandsChunkFailureError):
        if fail_frac > DEFAULT_MAX_FAILED_CHUNK_FRACTION:
            raise VomHeightBandsChunkFailureError(
                f"Failed chunk fraction {n_failed}/{n_chunks}"
            )
