"""Synthetic-grid tests for VOM canopy masks and 100 m summaries."""

from __future__ import annotations

import numpy as np
import pytest

from sdm.commands.data_preparation.rebuild_steps import (
    OPT_IN_REBUILD_STEPS,
    resolve_active_steps,
)
from sdm.raster.vom_canopy import (
    CANOPY_MIN_HEIGHT_M,
    COMPONENT_LAYER_IDS,
    build_canopy_masks,
    canopy_mask,
    compare_narrow_canopy_retention,
    height_band_masks,
    linear_woody_mask,
    summarise_masks_to_grid,
    woodland_from_canopy,
)


def test_vom_canopy_is_opt_in_not_default() -> None:
    assert "vom_canopy" in OPT_IN_REBUILD_STEPS
    active = resolve_active_steps()
    assert "vom_canopy" not in active
    assert "vom" in active
    assert resolve_active_steps(only=["vom_canopy"]) == frozenset({"vom_canopy"})


def test_canopy_mask_threshold() -> None:
    height = np.array(
        [[0.0, 2.4, 2.5, 10.0], [np.nan, 5.0, 2.49, 40.0]],
        dtype=np.float64,
    )
    mask = canopy_mask(height)
    expected = np.array(
        [[False, False, True, True], [False, True, False, True]],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)
    assert CANOPY_MIN_HEIGHT_M == 2.5


def test_height_band_masks() -> None:
    height = np.array(
        [[3.0, 6.0, 16.0, 1.0], [4.9, 15.0, 15.1, np.nan]],
        dtype=np.float64,
    )
    low, mid, high = height_band_masks(height)
    np.testing.assert_array_equal(
        low, np.array([[True, False, False, False], [True, False, False, False]])
    )
    np.testing.assert_array_equal(
        mid, np.array([[False, True, False, False], [False, True, False, False]])
    )
    np.testing.assert_array_equal(
        high, np.array([[False, False, True, False], [False, False, True, False]])
    )


def test_woodland_from_canopy_requires_core_and_area() -> None:
    # 100×100 m solid block at 1 m → 1 ha; with 12 m erosion a core remains.
    solid = np.ones((100, 100), dtype=bool)
    woodland = woodland_from_canopy(solid, resolution_m=1.0, erosion_m=12.0)
    assert woodland.any()
    assert woodland.sum() == solid.sum()

    # Small 10×10 patch (0.01 ha) cannot be woodland.
    small = np.zeros((50, 50), dtype=bool)
    small[20:30, 20:30] = True
    assert not woodland_from_canopy(small, resolution_m=1.0).any()


def test_linear_woody_detects_narrow_strip() -> None:
    grid = np.zeros((80, 80), dtype=bool)
    # 4 m wide × 60 m long hedge-like strip (well under woodland area).
    grid[40:44, 10:70] = True
    linear = linear_woody_mask(grid, woodland=np.zeros_like(grid), resolution_m=1.0)
    assert linear.sum() > 0
    assert linear.sum() == grid.sum()


def test_summarise_masks_to_grid_shares_and_lengths() -> None:
    # 20×20 at 1 m → one 10 m summary cell when summarising to 10 m.
    height = np.full((20, 20), 0.0, dtype=np.float64)
    height[0:10, 0:10] = 8.0  # quarter canopy in top-left of full 20×20
    height[5, 0:20] = 4.0  # thin east–west line
    masks = build_canopy_masks(height, resolution_m=1.0)
    summaries = summarise_masks_to_grid(
        masks, height, fine_resolution_m=1.0, summary_resolution_m=10
    )
    for name in COMPONENT_LAYER_IDS:
        assert name in summaries
        assert summaries[name].shape == (2, 2)
    # Top-left 10×10 is fully canopy from the block (line overlaps).
    assert summaries["vom_canopy_share"][0, 0] == pytest.approx(1.0)
    assert summaries["vom_canopy_edge_length_ha"][0, 0] > 0
    assert np.isfinite(summaries["vom_canopy_height_std"][0, 0])


def test_canopy_component_docs_are_available() -> None:
    from sdm.predictors import load_layer

    doc = load_layer("vom_canopy_share")
    assert doc.source_id == "ea_vom"
    assert doc.pipeline is not None
    assert doc.pipeline.rebuild_step == "vom_canopy"
    derived = load_layer("vom_canopy_share_500m")
    assert derived.derived_from == ["vom_canopy_share"]


def test_compare_narrow_canopy_retention_reports_loss() -> None:
    """A 3 m-wide hedge is retained at 1 m but diluted by 10 m block-mean."""
    height = np.full((100, 100), 0.0, dtype=np.float64)
    height[50:53, 10:90] = 6.0  # 3 m × 80 m hedge
    stats = compare_narrow_canopy_retention(height, resolution_1m=1.0)
    assert stats["linear_pixels_1m"] > 0
    assert stats["linear_lost_frac"] == pytest.approx(1.0) or stats[
        "linear_retained_at_10m_frac"
    ] < 0.5
    # Block mean of a 3 m strip inside 10×10 is 3/100 * 6 = 0.18 m ≪ 2.5 m.
    assert stats["linear_retained_at_10m_frac"] == pytest.approx(0.0)
    assert stats["linear_lost_frac"] == pytest.approx(1.0)
