"""Composable VOM canopy masks and 100 m summaries (EA VOM only; no CEH/OS).

Masks are local array ops only: thresholds and ``scipy.ndimage.uniform_filter``
box means. No Euclidean distance transforms and no connected-component
labelling — safe under Dask ``map_overlap`` with a fixed halo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import xarray as xr
from scipy import ndimage
from skimage.morphology import skeletonize

# Official VOM inclusion threshold (metres) — top-of-canopy objects above this.
CANOPY_MIN_HEIGHT_M = 2.5

# Height bands (metres) applied only where canopy is present.
HEIGHT_BAND_LOW = (2.5, 5.0)
HEIGHT_BAND_MID = (5.0, 15.0)
# Open upper bound: canopy cells with height > 15 m.
HEIGHT_BAND_HIGH_MIN = 15.0

# Woodland-from-canopy: near-solid box-mean core (~25 m ≈ 2·12+1), then
# binary_propagation under canopy (component fill without labelling / EDT).
WOODLAND_CLOSE_WINDOW_M = 0.0
WOODLAND_CLOSE_MIN_FRAC = 0.0
WOODLAND_WINDOW_M = 25.0
WOODLAND_MIN_FRAC = 0.99

# Linear woody: non-woodland canopy via H/V vs square box-mean elongation.
LINEAR_LINE_WINDOW_PX = (5, 41)  # (short, long) for H/V uniform filters
LINEAR_BLOB_WINDOW_PX = 9
LINEAR_NARROW_WINDOW_PX = 21
LINEAR_NARROW_MAX_DENSITY = 0.55
LINEAR_LINE_MIN_DENSITY = 0.45
LINEAR_BLOB_MAX_DENSITY = 0.70

# Default summary cell size (model grid).
DEFAULT_SUMMARY_RESOLUTION_M = 100


def _window_px(window_m: float, resolution_m: float) -> int:
    """Odd pixel window covering ``window_m`` at ``resolution_m`` (≥1)."""
    return max(1, int(round(float(window_m) / float(resolution_m))))


def woodland_halo_px(
    resolution_m: float = 1.0,
    *,
    close_window_m: float = WOODLAND_CLOSE_WINDOW_M,
    window_m: float = WOODLAND_WINDOW_M,
) -> int:
    """``map_overlap`` halo (px) for woodland core box-mean.

    Propagation under canopy is applied after cores are assembled so a fixed
    halo only needs to cover the core window (not whole woodland extent).
    """
    close_px = _window_px(close_window_m, resolution_m) if close_window_m > 0 else 0
    win_px = _window_px(window_m, resolution_m)
    return (close_px // 2) + (win_px // 2)


def linear_halo_px(resolution_m: float = 1.0) -> int:
    """Extra halo (px) for H/V and narrow box-mean windows."""
    del resolution_m  # windows are defined in pixels at 1 m; scale with res if needed
    return (max(LINEAR_LINE_WINDOW_PX) + 1) // 2


def mask_overlap_halo_px(
    resolution_m: float = 1.0,
    *,
    close_window_m: float = WOODLAND_CLOSE_WINDOW_M,
    window_m: float = WOODLAND_WINDOW_M,
) -> int:
    """Halo covering woodland then linear local ops in one ``map_overlap``."""
    return woodland_halo_px(
        resolution_m, close_window_m=close_window_m, window_m=window_m
    ) + linear_halo_px(resolution_m)


@dataclass(frozen=True)
class CanopyMasks:
    """Boolean masks at the native VOM resolution."""

    canopy: np.ndarray
    band_2p5_5: np.ndarray
    band_5_15: np.ndarray
    band_gt15: np.ndarray
    woodland: np.ndarray
    trees_outside_woodland: np.ndarray
    linear: np.ndarray
    canopy_edge: np.ndarray


def _finite_height(height: np.ndarray) -> np.ndarray:
    arr = np.asarray(height, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def canopy_mask(
    height: np.ndarray,
    *,
    min_height_m: float = CANOPY_MIN_HEIGHT_M,
) -> np.ndarray:
    """Binary canopy: finite heights at or above the VOM threshold."""
    h = _finite_height(height)
    return np.isfinite(h) & (h >= min_height_m)


def height_band_masks(
    height: np.ndarray,
    *,
    canopy: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height-band masks restricted to canopy cells (2.5–5, 5–15, >15 m)."""
    h = _finite_height(height)
    if canopy is None:
        canopy = canopy_mask(h)
    # Bands are contiguous: [2.5, 5), [5, 15], (15, ∞) so 15 m sits in mid.
    low = canopy & (h >= HEIGHT_BAND_LOW[0]) & (h < HEIGHT_BAND_LOW[1])
    mid = canopy & (h >= HEIGHT_BAND_MID[0]) & (h <= HEIGHT_BAND_MID[1])
    high = canopy & (h > HEIGHT_BAND_HIGH_MIN)
    return low, mid, high


def woodland_core_mask(
    canopy: np.ndarray,
    *,
    resolution_m: float = 1.0,
    close_window_m: float = WOODLAND_CLOSE_WINDOW_M,
    close_min_frac: float = WOODLAND_CLOSE_MIN_FRAC,
    window_m: float = WOODLAND_WINDOW_M,
    min_frac: float = WOODLAND_MIN_FRAC,
) -> np.ndarray:
    """Near-solid box-mean core of canopy (local; safe under ``map_overlap``)."""
    canopy_bool = np.asarray(canopy, dtype=bool)
    if not canopy_bool.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    dens = canopy_bool.astype(np.float64)
    close_px = _window_px(close_window_m, resolution_m) if close_window_m > 0 else 0
    if close_px > 1 and close_min_frac > 0:
        dens = (
            ndimage.uniform_filter(dens, size=close_px) >= float(close_min_frac)
        ).astype(np.float64)
    win_px = _window_px(window_m, resolution_m)
    return ndimage.uniform_filter(dens, size=win_px) >= float(min_frac)


def woodland_from_canopy(
    canopy: np.ndarray,
    *,
    resolution_m: float = 1.0,
    close_window_m: float = WOODLAND_CLOSE_WINDOW_M,
    close_min_frac: float = WOODLAND_CLOSE_MIN_FRAC,
    window_m: float = WOODLAND_WINDOW_M,
    min_frac: float = WOODLAND_MIN_FRAC,
    core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Woodland via dense box-mean core + propagation under canopy.

    No EDT and no connected-component labelling: ``binary_propagation`` fills
    canopy reachable from the core (same components the labelled path kept).
    """
    canopy_bool = np.asarray(canopy, dtype=bool)
    if not canopy_bool.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    if core is None:
        core = woodland_core_mask(
            canopy_bool,
            resolution_m=resolution_m,
            close_window_m=close_window_m,
            close_min_frac=close_min_frac,
            window_m=window_m,
            min_frac=min_frac,
        )
    core_bool = np.asarray(core, dtype=bool) & canopy_bool
    if not core_bool.any():
        return np.zeros_like(canopy_bool, dtype=bool)
    return ndimage.binary_propagation(
        core_bool,
        mask=canopy_bool,
        structure=np.ones((3, 3), dtype=bool),
    )


def linear_woody_mask(
    canopy: np.ndarray,
    *,
    woodland: Optional[np.ndarray] = None,
    resolution_m: float = 1.0,
) -> np.ndarray:
    """Narrow elongated non-woodland canopy via box-mean density windows."""
    del resolution_m  # windows are pixel-sized at the fine grid
    canopy_bool = np.asarray(canopy, dtype=bool)
    if woodland is None:
        woodland = woodland_from_canopy(canopy_bool)
    candidates = canopy_bool & ~np.asarray(woodland, dtype=bool)
    if not candidates.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    dens = candidates.astype(np.float64)
    local = ndimage.uniform_filter(dens, size=LINEAR_NARROW_WINDOW_PX)
    narrow = candidates & (local > 0.0) & (local <= LINEAR_NARROW_MAX_DENSITY)
    if not narrow.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    short_px, long_px = LINEAR_LINE_WINDOW_PX
    line_h = ndimage.uniform_filter(dens, size=(short_px, long_px))
    line_v = ndimage.uniform_filter(dens, size=(long_px, short_px))
    blob = ndimage.uniform_filter(dens, size=LINEAR_BLOB_WINDOW_PX)
    elongated = (
        (np.maximum(line_h, line_v) >= LINEAR_LINE_MIN_DENSITY)
        & (blob <= LINEAR_BLOB_MAX_DENSITY)
    )
    return narrow & elongated


def canopy_edge_mask(canopy: np.ndarray) -> np.ndarray:
    """Canopy pixels with a non-canopy neighbour in the 3×3 box (8-connected)."""
    canopy_bool = np.asarray(canopy, dtype=bool)
    # Mean < 1 ⇒ at least one of the 9 cells in the 3×3 is False.
    neigh = ndimage.uniform_filter(canopy_bool.astype(np.float64), size=3)
    return canopy_bool & (neigh < 1.0 - 1e-12)


def build_canopy_masks(
    height: np.ndarray,
    *,
    resolution_m: float = 1.0,
) -> CanopyMasks:
    """Derive all 1 m canopy / patch / band masks from a height array."""
    canopy = canopy_mask(height)
    low, mid, high = height_band_masks(height, canopy=canopy)
    woodland = woodland_from_canopy(canopy, resolution_m=resolution_m)
    linear = linear_woody_mask(
        canopy, woodland=woodland, resolution_m=resolution_m
    )
    tow = canopy & ~woodland
    edge = canopy_edge_mask(canopy)
    return CanopyMasks(
        canopy=canopy,
        band_2p5_5=low,
        band_5_15=mid,
        band_gt15=high,
        woodland=woodland,
        trees_outside_woodland=tow,
        linear=linear,
        canopy_edge=edge,
    )


def _block_reduce_sum(mask: np.ndarray, factor: int) -> np.ndarray:
    """Sum boolean/float mask into non-overlapping blocks (trim incomplete edge)."""
    arr = np.asarray(mask, dtype=np.float64)
    ny, nx = arr.shape
    ny_c = (ny // factor) * factor
    nx_c = (nx // factor) * factor
    if ny_c == 0 or nx_c == 0:
        return np.zeros((0, 0), dtype=np.float64)
    trimmed = arr[:ny_c, :nx_c]
    reshaped = trimmed.reshape(ny_c // factor, factor, nx_c // factor, factor)
    return reshaped.sum(axis=(1, 3))


def _block_reduce_canopy_std(
    height: np.ndarray,
    canopy: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Per-block standard deviation of height within canopy cells only."""
    h = _finite_height(height)
    can = np.asarray(canopy, dtype=bool)
    ny, nx = h.shape
    ny_c = (ny // factor) * factor
    nx_c = (nx // factor) * factor
    out_ny, out_nx = ny_c // factor, nx_c // factor
    out = np.full((out_ny, out_nx), np.nan, dtype=np.float64)
    if ny_c == 0 or nx_c == 0:
        return out
    vals = np.where(can & np.isfinite(h), h, np.nan)[:ny_c, :nx_c]
    blocks = vals.reshape(out_ny, factor, out_nx, factor)
    flat = blocks.reshape(out_ny, out_nx, factor * factor)
    finite = np.isfinite(flat)
    n = np.sum(finite, axis=2)
    filled = np.where(finite, flat, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(filled, axis=2) / np.maximum(n, 1)
        mean_sq = np.sum(filled * filled, axis=2) / np.maximum(n, 1)
        var = mean_sq - mean * mean
    out = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), out)
    out = np.where(n == 1, 0.0, out)
    return out


def summarise_masks_to_grid(
    masks: CanopyMasks,
    height: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
) -> dict[str, np.ndarray]:
    """Aggregate 1 m masks to summary cells via reshape-mean (no Python loops).

    Share layers are the fraction of the summary cell covered by the mask.
    Length layers count skeleton (linear) or edge pixels × fine resolution,
    divided by cell area in hectares.

    Assumes the fine array origin aligns so blocks map to a regular summary
    grid (use :func:`summarise_masks_by_coordinates` when tile origins may
    fall off the national 100 m lattice).
    """
    if summary_resolution_m % int(fine_resolution_m) != 0:
        raise ValueError(
            f"summary_resolution_m ({summary_resolution_m}) must be a multiple "
            f"of fine_resolution_m ({fine_resolution_m})"
        )
    factor = int(round(summary_resolution_m / fine_resolution_m))
    cell_pixels = float(factor * factor)
    cell_area_ha = (summary_resolution_m ** 2) / 10_000.0

    shares = {
        "vom_canopy_share": _block_reduce_sum(masks.canopy, factor) / cell_pixels,
        "vom_woodland_from_canopy_share": (
            _block_reduce_sum(masks.woodland, factor) / cell_pixels
        ),
        "vom_trees_outside_woodland_share": (
            _block_reduce_sum(masks.trees_outside_woodland, factor) / cell_pixels
        ),
        "vom_canopy_height_band_2p5_5_share": (
            _block_reduce_sum(masks.band_2p5_5, factor) / cell_pixels
        ),
        "vom_canopy_height_band_5_15_share": (
            _block_reduce_sum(masks.band_5_15, factor) / cell_pixels
        ),
        "vom_canopy_height_band_gt15_share": (
            _block_reduce_sum(masks.band_gt15, factor) / cell_pixels
        ),
    }

    linear_skel = skeletonize(np.asarray(masks.linear, dtype=bool))
    linear_length_m = _block_reduce_sum(linear_skel, factor) * fine_resolution_m
    edge_length_m = (
        _block_reduce_sum(masks.canopy_edge, factor) * fine_resolution_m
    )
    shares["vom_linear_woody_length_ha"] = linear_length_m / cell_area_ha
    shares["vom_canopy_edge_length_ha"] = edge_length_m / cell_area_ha
    shares["vom_canopy_height_std"] = _block_reduce_canopy_std(
        height, masks.canopy, factor
    )
    return shares


def summarise_masks_by_coordinates(
    masks: CanopyMasks,
    height: np.ndarray,
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    tile_id: Optional[str] = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Aggregate fine masks into national-grid 100 m cells by coordinates.

    Keys are ``(floor(y / res), floor(x / res))`` in CRS metres. Partial tile
    overlap is normalised by observed fine pixels; full cells use the expected
    pixel count so shares stay on ``[0, 1]``.

    After broadcasting 1-D coordinate vectors to the fine grid, ``row`` / ``col``
    and every mask or weights array must share the same ``(ny, nx)`` shape, and
    flattened index length must match before ``np.bincount``. Mismatches raise
    ``ValueError`` with shapes (and optional ``tile_id``) rather than numpy's
    opaque weights-length error.
    """
    res = float(summary_resolution_m)
    fine = float(fine_resolution_m)
    cell_area_ha = (res ** 2) / 10_000.0
    xs = np.asarray(x_coords, dtype=np.float64)
    ys = np.asarray(y_coords, dtype=np.float64)
    h = _finite_height(height)
    canopy = np.asarray(masks.canopy, dtype=bool)
    expected_shape = canopy.shape
    if h.shape != expected_shape:
        where = f" (tile {tile_id})" if tile_id else ""
        raise ValueError(
            f"height shape {h.shape} != canopy shape {expected_shape}{where}"
        )

    col = np.floor(xs[None, :] / res).astype(np.int64)
    row = np.floor(ys[:, None] / res).astype(np.int64)
    # Broadcast (ny, 1) × (1, nx) to (ny, nx) before flatten — otherwise
    # keys/inv have ny (or nx) entries and bincount weights fail on real tiles.
    row, col = np.broadcast_arrays(row, col)

    linear_skel = skeletonize(np.asarray(masks.linear, dtype=bool))
    woodland = np.asarray(masks.woodland, dtype=bool)
    tow = np.asarray(masks.trees_outside_woodland, dtype=bool)
    band_low = np.asarray(masks.band_2p5_5, dtype=bool)
    band_mid = np.asarray(masks.band_5_15, dtype=bool)
    band_high = np.asarray(masks.band_gt15, dtype=bool)
    edge = np.asarray(masks.canopy_edge, dtype=bool)
    canopy_vals = np.where(canopy & np.isfinite(h), h, np.nan)
    canopy_weight = np.isfinite(canopy_vals).astype(np.float64)
    h_filled = np.where(np.isfinite(canopy_vals), canopy_vals, 0.0)

    shape_sources: dict[str, tuple[int, ...]] = {
        "row": row.shape,
        "col": col.shape,
        "height": h.shape,
        "canopy": canopy.shape,
        "woodland": woodland.shape,
        "trees_outside_woodland": tow.shape,
        "band_2p5_5": band_low.shape,
        "band_5_15": band_mid.shape,
        "band_gt15": band_high.shape,
        "linear_skel": linear_skel.shape,
        "canopy_edge": edge.shape,
        "canopy_weight": canopy_weight.shape,
        "h_filled": h_filled.shape,
    }
    mismatched = {
        name: shape
        for name, shape in shape_sources.items()
        if shape != expected_shape
    }
    if mismatched:
        where = f" (tile {tile_id})" if tile_id else ""
        raise ValueError(
            "summarise_masks_by_coordinates: after broadcast, row/col and all "
            f"mask/weights arrays must share shape {expected_shape}; "
            f"mismatched: {mismatched}{where}"
        )

    keys = np.stack([row.ravel(), col.ravel()], axis=1)
    n_pixels = int(np.prod(expected_shape))
    if keys.shape[0] != n_pixels:
        where = f" (tile {tile_id})" if tile_id else ""
        raise ValueError(
            "summarise_masks_by_coordinates: flattened index length "
            f"{keys.shape[0]} != fine-grid size {n_pixels} "
            f"(shape {expected_shape}){where}"
        )

    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = uniq.shape[0]
    if inv.size != n_pixels:
        where = f" (tile {tile_id})" if tile_id else ""
        raise ValueError(
            "summarise_masks_by_coordinates: bincount index length "
            f"{inv.size} != fine-grid size {n_pixels}{where}"
        )
    if inv.size and (int(inv.min()) < 0 or int(inv.max()) >= n_cells):
        where = f" (tile {tile_id})" if tile_id else ""
        raise ValueError(
            "summarise_masks_by_coordinates: bincount indices outside "
            f"[0, {n_cells}) (min={int(inv.min())}, max={int(inv.max())})"
            f"{where}"
        )

    def _sum(mask: np.ndarray) -> np.ndarray:
        weights = np.asarray(mask, dtype=np.float64).ravel()
        if weights.size != inv.size:
            where = f" (tile {tile_id})" if tile_id else ""
            raise ValueError(
                "summarise_masks_by_coordinates: bincount weights length "
                f"{weights.size} != index length {inv.size}{where}"
            )
        return np.bincount(inv, weights=weights, minlength=n_cells)

    n_pix = np.bincount(inv, minlength=n_cells).astype(np.float64)
    sum_canopy = _sum(canopy)
    sum_wood = _sum(woodland)
    sum_tow = _sum(tow)
    sum_low = _sum(band_low)
    sum_mid = _sum(band_mid)
    sum_high = _sum(band_high)
    sum_skel = _sum(linear_skel) * fine
    sum_edge = _sum(edge) * fine
    h_n = _sum(canopy_weight)
    h_sum = _sum(h_filled)
    h_sumsq = _sum(h_filled ** 2)

    expected_pixels = (res / fine) ** 2
    # Vectorised cell stats (no per-cell Python body beyond zip into dict).
    denom = np.where(
        n_pix >= expected_pixels * 0.99, expected_pixels, n_pix
    ).astype(np.float64)
    valid = denom > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.full(n_cells, np.nan, dtype=np.float64)
        ok2 = h_n >= 2
        mean = np.where(ok2, h_sum / np.maximum(h_n, 1), 0.0)
        std = np.where(
            ok2,
            np.sqrt(np.maximum(h_sumsq / np.maximum(h_n, 1) - mean * mean, 0.0)),
            std,
        )
        std = np.where(h_n == 1, 0.0, std)

    out: dict[tuple[int, int], dict[str, float]] = {}
    for i in np.flatnonzero(valid):
        d = float(denom[i])
        key = (int(uniq[i, 0]), int(uniq[i, 1]))
        out[key] = {
            "vom_canopy_share": float(sum_canopy[i] / d),
            "vom_woodland_from_canopy_share": float(sum_wood[i] / d),
            "vom_trees_outside_woodland_share": float(sum_tow[i] / d),
            "vom_canopy_height_band_2p5_5_share": float(sum_low[i] / d),
            "vom_canopy_height_band_5_15_share": float(sum_mid[i] / d),
            "vom_canopy_height_band_gt15_share": float(sum_high[i] / d),
            "vom_linear_woody_length_ha": float(sum_skel[i] / cell_area_ha),
            "vom_canopy_edge_length_ha": float(sum_edge[i] / cell_area_ha),
            "vom_canopy_height_std": float(std[i]),
        }
    return out


COMPONENT_LAYER_IDS: tuple[str, ...] = (
    "vom_canopy_share",
    "vom_woodland_from_canopy_share",
    "vom_trees_outside_woodland_share",
    "vom_linear_woody_length_ha",
    "vom_canopy_edge_length_ha",
    "vom_canopy_height_band_2p5_5_share",
    "vom_canopy_height_band_5_15_share",
    "vom_canopy_height_band_gt15_share",
    "vom_canopy_height_std",
)


def compare_narrow_canopy_retention(
    height_1m: np.ndarray,
    *,
    resolution_1m: float = 1.0,
    coarse_resolution_m: float = 10.0,
) -> Mapping[str, float]:
    """Measure how much narrow/linear canopy is lost when coarsening to 10 m.

    Simulates a coarser WCS download by block-mean of 1 m heights, then
    re-thresholding at 2.5 m. Reports retention of linear-woody and all-canopy
    pixels relative to the 1 m masks. Used for Implementation-notes numbers.
    """
    if coarse_resolution_m % resolution_1m != 0:
        raise ValueError("coarse_resolution_m must be a multiple of resolution_1m")
    factor = int(round(coarse_resolution_m / resolution_1m))
    masks_1m = build_canopy_masks(height_1m, resolution_m=resolution_1m)

    h = _finite_height(height_1m)
    ny, nx = h.shape
    ny_c = (ny // factor) * factor
    nx_c = (nx // factor) * factor
    if ny_c == 0 or nx_c == 0:
        return {
            "canopy_pixels_1m": 0.0,
            "linear_pixels_1m": 0.0,
            "canopy_retained_at_10m_frac": float("nan"),
            "linear_retained_at_10m_frac": float("nan"),
            "linear_lost_frac": float("nan"),
        }

    h_trim = h[:ny_c, :nx_c]
    blocks = h_trim.reshape(ny_c // factor, factor, nx_c // factor, factor)
    with np.errstate(all="ignore"):
        coarse = np.nanmean(blocks, axis=(1, 3))

    coarse_canopy = canopy_mask(coarse)
    coarse_canopy_1m = np.repeat(np.repeat(coarse_canopy, factor, axis=0), factor, axis=1)

    canopy_1m = masks_1m.canopy[:ny_c, :nx_c]
    linear_1m = masks_1m.linear[:ny_c, :nx_c]
    n_canopy = int(canopy_1m.sum())
    n_linear = int(linear_1m.sum())
    canopy_retained = (
        float((canopy_1m & coarse_canopy_1m).sum()) / n_canopy if n_canopy else float("nan")
    )
    linear_retained = (
        float((linear_1m & coarse_canopy_1m).sum()) / n_linear if n_linear else float("nan")
    )
    return {
        "canopy_pixels_1m": float(n_canopy),
        "linear_pixels_1m": float(n_linear),
        "canopy_retained_at_10m_frac": canopy_retained,
        "linear_retained_at_10m_frac": linear_retained,
        "linear_lost_frac": (
            1.0 - linear_retained if np.isfinite(linear_retained) else float("nan")
        ),
        "coarse_factor": float(factor),
    }


def masks_to_dataarrays(
    summaries: Mapping[str, np.ndarray],
    *,
    y: np.ndarray,
    x: np.ndarray,
    crs,
    transform,
) -> xr.Dataset:
    """Wrap summary arrays as an ``xr.Dataset`` on the model grid."""
    data_vars = {}
    for name, arr in summaries.items():
        da = xr.DataArray(arr, coords={"y": y, "x": x}, dims=("y", "x"), name=name)
        da.attrs["long_name"] = name
        data_vars[name] = da
    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    for name in ds.data_vars:
        ds[name].rio.write_nodata(np.nan, inplace=True)
    return ds


__all__ = [
    "CANOPY_MIN_HEIGHT_M",
    "COMPONENT_LAYER_IDS",
    "DEFAULT_SUMMARY_RESOLUTION_M",
    "HEIGHT_BAND_HIGH_MIN",
    "HEIGHT_BAND_LOW",
    "HEIGHT_BAND_MID",
    "WOODLAND_CLOSE_MIN_FRAC",
    "WOODLAND_CLOSE_WINDOW_M",
    "WOODLAND_MIN_FRAC",
    "WOODLAND_WINDOW_M",
    "CanopyMasks",
    "build_canopy_masks",
    "canopy_edge_mask",
    "canopy_mask",
    "compare_narrow_canopy_retention",
    "height_band_masks",
    "linear_halo_px",
    "linear_woody_mask",
    "mask_overlap_halo_px",
    "masks_to_dataarrays",
    "summarise_masks_by_coordinates",
    "summarise_masks_to_grid",
    "woodland_core_mask",
    "woodland_from_canopy",
    "woodland_halo_px",
]
