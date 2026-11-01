"""Unit tests for simple VOM height bands (#121) — no woodland/TOW/linear masks."""

from __future__ import annotations

import numpy as np
import pytest

from sdm.raster.vom_height_bands import (
    CELL_LAYER_STEMS,
    canopy_mask,
    height_band_masks,
    pairwise_pearson,
    summarise_height_block,
    summarise_height_to_grid,
)


def _synthetic_height(n: int = 200) -> np.ndarray:
    """200×200 1 m grid: open, shrub, mid trees, tall canopy, a mast."""
    h = np.full((n, n), np.nan, dtype=np.float64)
    h[0:40, 0:40] = 1.0  # below VOM threshold
    h[40:80, 40:80] = 3.5  # 2.5–5
    h[80:140, 80:140] = 10.0  # 5–15
    h[140:200, 140:200] = 22.0  # >15
    h[10, 10] = 55.0  # mast (capped by caller / block kernel)
    return h


def test_canopy_and_band_masks_partition_woody_pixels() -> None:
    h = _synthetic_height(100)
    canopy = canopy_mask(h)
    low, mid, high = height_band_masks(h, canopy=canopy)
    assert canopy.dtype == bool
    # Bands are mutually exclusive and cover canopy.
    assert not np.any(low & mid)
    assert not np.any(low & high)
    assert not np.any(mid & high)
    np.testing.assert_array_equal(canopy, low | mid | high)
    assert int(low.sum()) > 0 and int(mid.sum()) > 0 and int(high.sum()) > 0


def test_summarise_height_to_grid_shares_and_std() -> None:
    h = np.full((100, 100), 12.0, dtype=np.float64)
    h[50:100, 50:100] = 20.0
    layers = summarise_height_to_grid(h, fine_resolution_m=1.0, summary_resolution_m=100)
    assert set(layers) == set(CELL_LAYER_STEMS)
    assert layers["woody_cover_share"].shape == (1, 1)
    # All canopy: three quarters mid (12 m), one quarter tall (20 m).
    assert layers["woody_cover_share"][0, 0] == pytest.approx(1.0)
    assert layers["height_band_5_15_share"][0, 0] == pytest.approx(0.75)
    assert layers["height_band_gt15_share"][0, 0] == pytest.approx(0.25)
    assert layers["height_band_2p5_5_share"][0, 0] == pytest.approx(0.0)
    assert layers["canopy_height_std"][0, 0] > 0.0


def test_summarise_height_block_caps_and_tolerates_bad_input() -> None:
    h = np.full((100, 100), 10.0, dtype=np.float64)
    h[0, 0] = 80.0  # above 40 m cap
    stacked = summarise_height_block(h, fine_resolution_m=1.0, summary_resolution_m=100)
    assert stacked.shape == (len(CELL_LAYER_STEMS), 1, 1)
    assert np.all(np.isfinite(stacked))

    # Non-array-like path: empty → all-NaN failure tile (no raise).
    bad = summarise_height_block(
        np.zeros((0, 0)), fine_resolution_m=1.0, summary_resolution_m=100
    )
    assert bad.shape[0] == len(CELL_LAYER_STEMS)
    assert np.all(np.isnan(bad))


def test_pairwise_pearson_flags_close_pairs() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a * 2.0
    c = np.array([4.0, 3.0, 2.0, 1.0])
    corr = pairwise_pearson({"a": a, "b": b, "c": c})
    assert corr[("a", "b")] == pytest.approx(1.0)
    assert corr[("a", "c")] == pytest.approx(-1.0)


def test_bands_sum_to_woody_cover() -> None:
    h = _synthetic_height(200)
    # Cap mast like the live path.
    h = np.clip(h, None, 40.0)
    layers = summarise_height_to_grid(h, summary_resolution_m=100)
    band_sum = (
        layers["height_band_2p5_5_share"]
        + layers["height_band_5_15_share"]
        + layers["height_band_gt15_share"]
    )
    np.testing.assert_allclose(band_sum, layers["woody_cover_share"], atol=1e-6)
