"""Synthetic-grid tests for VOM canopy masks and 100 m summaries."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from sdm.commands.data_preparation.rebuild_steps import (
    OPT_IN_REBUILD_STEPS,
    resolve_active_steps,
)
from sdm.raster.vom_canopy import (
    CANOPY_MIN_HEIGHT_M,
    COMPONENT_LAYER_IDS,
    WOODLAND_CLOSE_WINDOW_M,
    WOODLAND_MIN_FRAC,
    WOODLAND_WINDOW_M,
    CanopyMasks,
    build_canopy_masks,
    canopy_edge_mask,
    canopy_mask,
    compare_narrow_canopy_retention,
    height_band_masks,
    linear_woody_mask,
    mask_overlap_halo_px,
    summarise_masks_by_coordinates,
    summarise_masks_to_grid,
    woodland_from_canopy,
)

# Legacy labelled algorithm (pre-opening heuristics) for agreement checks only.
_LEGACY_WOODLAND_MIN_AREA_M2 = 5_000.0
_LEGACY_LINEAR_MIN_ASPECT = 3.0
_LEGACY_LINEAR_MIN_AREA_M2 = 50.0


def _legacy_woodland(
    canopy: np.ndarray,
    *,
    resolution_m: float = 1.0,
    erosion_m: float = 12.0,
    min_area_m2: float = _LEGACY_WOODLAND_MIN_AREA_M2,
) -> np.ndarray:
    """Former EDT-core + labelled area filter (agreement baseline)."""
    canopy_bool = np.asarray(canopy, dtype=bool)
    if not canopy_bool.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    radius_px = max(1, int(round(erosion_m / resolution_m)))
    core = ndimage.distance_transform_edt(canopy_bool) > radius_px
    if not core.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    labeled, n_labels = ndimage.label(canopy_bool)
    if n_labels == 0:
        return np.zeros_like(canopy_bool, dtype=bool)
    core_labels = np.unique(labeled[core])
    core_labels = core_labels[core_labels > 0]
    areas = ndimage.sum(
        canopy_bool.astype(np.float64), labeled, index=np.arange(1, n_labels + 1)
    )
    cell_area = float(resolution_m) ** 2
    woodland = np.zeros_like(canopy_bool, dtype=bool)
    for lab in core_labels:
        if areas[lab - 1] * cell_area >= min_area_m2:
            woodland |= labeled == lab
    return woodland


def _legacy_linear(
    canopy: np.ndarray,
    *,
    woodland: np.ndarray,
    resolution_m: float = 1.0,
    max_width_m: float = 20.0,
    min_aspect: float = _LEGACY_LINEAR_MIN_ASPECT,
    min_area_m2: float = _LEGACY_LINEAR_MIN_AREA_M2,
) -> np.ndarray:
    """Former per-label mean-width + aspect filter (agreement baseline)."""
    canopy_bool = np.asarray(canopy, dtype=bool)
    candidates = canopy_bool & ~np.asarray(woodland, dtype=bool)
    if not candidates.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    labeled, n_labels = ndimage.label(candidates)
    if n_labels == 0:
        return np.zeros_like(canopy_bool, dtype=bool)
    dist = ndimage.distance_transform_edt(candidates) * float(resolution_m)
    cell_area = float(resolution_m) ** 2
    linear = np.zeros_like(canopy_bool, dtype=bool)
    for lab in range(1, n_labels + 1):
        comp = labeled == lab
        area_m2 = float(comp.sum()) * cell_area
        if area_m2 < min_area_m2:
            continue
        mean_width = 2.0 * float(dist[comp].mean())
        if mean_width > max_width_m:
            continue
        ys, xs = np.nonzero(comp)
        if ys.size < 2:
            continue
        coords = np.column_stack(
            (xs.astype(np.float64) * resolution_m, ys.astype(np.float64) * resolution_m)
        )
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.ndim < 2:
            continue
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 1e-12, None)
        aspect = float(np.sqrt(eig[-1] / eig[0]))
        if aspect >= min_aspect:
            linear |= comp
    return linear


def _mask_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    denom = 2.0 * tp + fp + fn
    if denom <= 0:
        return 1.0
    return 2.0 * tp / denom


def _corr_and_mad(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    xv, yv = x[ok], y[ok]
    if float(xv.std()) < 1e-12 and float(yv.std()) < 1e-12:
        corr = 1.0
    else:
        corr = float(np.corrcoef(xv, yv)[0, 1])
    mad = float(np.mean(np.abs(xv - yv)))
    return corr, mad


def _sample_height(seed: int = 0, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = np.zeros((size, size), dtype=np.float64)
    scattered = rng.random((size, size)) > 0.78
    height[scattered] = rng.uniform(3.0, 22.0, size=int(scattered.sum()))
    # Woodland-scale blocks
    height[40 : size // 2, 40 : size // 2] = 14.0
    height[size // 2 + 20 : size - 20, size // 3 : size - 30] = 11.0
    # Narrow hedges
    height[8:12, 10 : size - 10] = 5.0
    height[size // 3 : 2 * size // 3, 16:20] = 6.0
    return height


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


def test_woodland_box_keeps_thick_core() -> None:
    solid = np.ones((100, 100), dtype=bool)
    woodland = woodland_from_canopy(solid, resolution_m=1.0)
    assert woodland.any()
    # Dense core + dilate recovers a solid block.
    assert woodland.sum() == solid.sum()

    small = np.zeros((50, 50), dtype=bool)
    small[20:30, 20:30] = True
    assert not woodland_from_canopy(small, resolution_m=1.0).any()


def test_linear_woody_detects_narrow_strip() -> None:
    grid = np.zeros((80, 80), dtype=bool)
    grid[40:44, 10:70] = True
    linear = linear_woody_mask(grid, woodland=np.zeros_like(grid), resolution_m=1.0)
    assert linear.sum() > 0
    # Elongation windows soften strip ends; still recover most of the hedge.
    assert _mask_f1(linear, grid) >= 0.85
    assert linear.sum() >= int(0.75 * grid.sum())


def test_canopy_edge_mask_is_8_connected() -> None:
    canopy = np.zeros((5, 5), dtype=bool)
    canopy[2, 2] = True
    canopy[1, 1] = True
    edge = canopy_edge_mask(canopy)
    assert edge[2, 2] and edge[1, 1]
    block = np.zeros((7, 7), dtype=bool)
    block[2:5, 2:5] = True
    edge_block = canopy_edge_mask(block)
    assert not edge_block[3, 3]
    assert edge_block[2, 2]


def test_mask_overlap_halo_covers_box_filters() -> None:
    halo = mask_overlap_halo_px(resolution_m=1.0)
    win_r = int(round(WOODLAND_WINDOW_M)) // 2
    linear_r = (41 + 1) // 2
    assert halo == win_r + linear_r
    assert WOODLAND_MIN_FRAC >= 0.9
    assert WOODLAND_CLOSE_WINDOW_M == 0.0


def test_local_heuristics_agree_with_legacy_on_sample_tiles() -> None:
    """Pixel F1 vs labelled baseline; 100 m summaries corr + MAD.

    Target band is ~0.90–0.95; box-filter heuristics trade exact match for
    local uniform_filter ops (no EDT / labelling).
    """
    f1_wood: list[float] = []
    f1_linear: list[float] = []
    f1_tow: list[float] = []
    corr_shares: list[float] = []
    mad_shares: list[float] = []

    for seed in (0, 1, 2):
        height = _sample_height(seed=seed, size=256)
        canopy = canopy_mask(height)
        masks = build_canopy_masks(height, resolution_m=1.0)
        wood_ref = _legacy_woodland(canopy)
        lin_ref = _legacy_linear(canopy, woodland=wood_ref)
        tow_ref = canopy & ~wood_ref

        f1_wood.append(_mask_f1(masks.woodland, wood_ref))
        f1_linear.append(_mask_f1(masks.linear, lin_ref))
        f1_tow.append(_mask_f1(masks.trees_outside_woodland, tow_ref))

        # Legacy summaries from legacy masks (same summary recipe).
        legacy_masks = CanopyMasks(
            canopy=canopy,
            band_2p5_5=masks.band_2p5_5,
            band_5_15=masks.band_5_15,
            band_gt15=masks.band_gt15,
            woodland=wood_ref,
            trees_outside_woodland=tow_ref,
            linear=lin_ref,
            canopy_edge=masks.canopy_edge,
        )
        got = summarise_masks_to_grid(
            masks, height, fine_resolution_m=1.0, summary_resolution_m=100
        )
        ref = summarise_masks_to_grid(
            legacy_masks, height, fine_resolution_m=1.0, summary_resolution_m=100
        )
        for key in (
            "vom_woodland_from_canopy_share",
            "vom_trees_outside_woodland_share",
            "vom_linear_woody_length_ha",
        ):
            corr, mad = _corr_and_mad(got[key], ref[key])
            corr_shares.append(corr)
            # MAD on fraction shares only (length units are O(10–100)/ha).
            if key.endswith("_share"):
                mad_shares.append(mad)

    assert float(np.min(f1_wood)) >= 0.90
    assert float(np.min(f1_tow)) >= 0.90
    assert float(np.min(f1_linear)) >= 0.90
    assert float(np.nanmin(corr_shares)) >= 0.90
    assert float(np.nanmean(mad_shares)) < 0.05


def test_summarise_masks_by_coordinates_mismatched_shapes_raise() -> None:
    """Coord/mask shape mismatch → clear ValueError (not numpy bincount)."""
    height = np.zeros((16, 16), dtype=np.float64)
    height[4:12, 4:12] = 8.0
    masks = build_canopy_masks(height, resolution_m=1.0)
    x_coords = np.arange(16, dtype=np.float64) + 0.5
    y_coords = np.arange(8, dtype=np.float64) + 0.5
    with pytest.raises(ValueError, match=r"mismatched:.*row.*\(8, 16\)") as exc:
        summarise_masks_by_coordinates(
            masks,
            height,
            x_coords=x_coords,
            y_coords=y_coords,
            fine_resolution_m=1.0,
            summary_resolution_m=10,
            tile_id="unit-mismatch",
        )
    msg = str(exc.value)
    assert "tile unit-mismatch" in msg
    assert "(16, 16)" in msg


def test_summarise_masks_by_coordinates_cli_tile_1d_coords() -> None:
    """1 m WCS-sized tile with 1-D x/y vectors as the tiled CLI passes."""
    ny = nx = 1024
    height = np.zeros((ny, nx), dtype=np.float64)
    height[200:220, 100:900] = 6.0
    height[400:520, 400:520] = 12.0
    x_coords = 500000.5 + np.arange(nx, dtype=np.float64)
    y_coords = 400000.5 - np.arange(ny, dtype=np.float64)
    canopy = height >= CANOPY_MIN_HEIGHT_M
    empty = np.zeros_like(canopy, dtype=bool)
    masks = CanopyMasks(
        canopy=canopy,
        band_2p5_5=canopy & (height < 5.0),
        band_5_15=canopy & (height >= 5.0) & (height <= 15.0),
        band_gt15=empty,
        woodland=empty,
        trees_outside_woodland=canopy,
        linear=canopy.copy(),
        canopy_edge=canopy.copy(),
    )
    got = summarise_masks_by_coordinates(
        masks,
        height,
        x_coords=x_coords,
        y_coords=y_coords,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        tile_id="wcs-1024",
    )
    assert got
    canopy_pixels_recovered = sum(
        vals["vom_canopy_share"] * 10_000.0 for vals in got.values()
    )
    assert canopy_pixels_recovered == pytest.approx(float(canopy.sum()), rel=0, abs=1e-6)


def test_summarise_masks_by_coordinates_matches_brute_force_64x64() -> None:
    """Real-shaped 2-D tile: broadcast keys so bincount weights match ny×nx."""
    rng = np.random.default_rng(0)
    ny = nx = 64
    height = rng.uniform(0.0, 20.0, size=(ny, nx)).astype(np.float64)
    height[height < 2.0] = 0.0
    x_coords = 420000.5 + np.arange(nx, dtype=np.float64)
    y_coords = 381000.5 - np.arange(ny, dtype=np.float64)
    masks = build_canopy_masks(height, resolution_m=1.0)
    summary_res = 10
    got = summarise_masks_by_coordinates(
        masks,
        height,
        x_coords=x_coords,
        y_coords=y_coords,
        fine_resolution_m=1.0,
        summary_resolution_m=summary_res,
    )
    assert got, "expected at least one summary cell"

    expected_counts: dict[tuple[int, int], int] = {}
    for iy in range(ny):
        for ix in range(nx):
            key = (
                int(np.floor(y_coords[iy] / summary_res)),
                int(np.floor(x_coords[ix] / summary_res)),
            )
            if masks.canopy[iy, ix]:
                expected_counts[key] = expected_counts.get(key, 0) + 1

    expected_pixels = float((summary_res / 1.0) ** 2)
    for key, count in expected_counts.items():
        assert key in got
        n_in_cell = sum(
            1
            for iy in range(ny)
            for ix in range(nx)
            if (
                int(np.floor(y_coords[iy] / summary_res)),
                int(np.floor(x_coords[ix] / summary_res)),
            )
            == key
        )
        denom = expected_pixels if n_in_cell >= expected_pixels * 0.99 else float(n_in_cell)
        assert got[key]["vom_canopy_share"] == pytest.approx(count / denom)


def test_summarise_masks_to_grid_shares_and_lengths() -> None:
    height = np.full((20, 20), 0.0, dtype=np.float64)
    height[0:10, 0:10] = 8.0
    height[5, 0:20] = 4.0
    masks = build_canopy_masks(height, resolution_m=1.0)
    summaries = summarise_masks_to_grid(
        masks, height, fine_resolution_m=1.0, summary_resolution_m=10
    )
    for name in COMPONENT_LAYER_IDS:
        assert name in summaries
        assert summaries[name].shape == (2, 2)
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
    height[50:53, 10:90] = 6.0
    stats = compare_narrow_canopy_retention(height, resolution_1m=1.0)
    assert stats["linear_pixels_1m"] > 0
    assert stats["linear_retained_at_10m_frac"] == pytest.approx(0.0)
    assert stats["linear_lost_frac"] == pytest.approx(1.0)


def test_vom_canopy_fails_loudly_when_most_chunks_fail(tmp_path, monkeypatch) -> None:
    """High chunk-failure fraction → non-zero path, no GeoTIFF written."""
    from sdm.commands.data_preparation.environmental import (
        generate_vom_canopy_features as mod,
    )

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    async def _boom(*_args, **_kwargs):
        raise mod.VomCanopyChunkFailureError(
            "VOM canopy abort: 10/10 chunks failed (100% > allowed 5%); "
            "no output written"
        )

    monkeypatch.setattr(mod, "fetch_and_process_vom_canopy", _boom)
    with pytest.raises(mod.VomCanopyChunkFailureError, match="no output written"):
        mod.generate_vom_canopy_features(out_dir, boundary)
    assert not (out_dir / "vom_canopy_features_100m.tif").exists()


def test_max_failed_chunk_fraction_constant() -> None:
    from sdm.commands.data_preparation.environmental.generate_vom_canopy_features import (
        MAX_FAILED_CHUNK_FRACTION,
    )

    assert 0.0 < MAX_FAILED_CHUNK_FRACTION <= 0.1
