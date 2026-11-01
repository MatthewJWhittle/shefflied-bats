"""Composable VOM canopy masks and 100 m summaries (EA VOM only; no CEH/OS).

Building blocks operate on fine-resolution height arrays (typically 1 m).
Component layers for modelling are short recipes over these masks/summaries.
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

# Woodland-from-canopy: Euclidean morphological opening radius (~10–15 m).
WOODLAND_EROSION_M = 12.0

# Linear woody: non-woodland canopy thinner than this width + local elongation.
LINEAR_MAX_WIDTH_M = 20.0
# Directional density windows (px) for cheap elongation vs compact TOW blobs.
LINEAR_LINE_WINDOW_PX = (5, 41)  # (short, long) for H/V uniform filters
LINEAR_BLOB_WINDOW_PX = 9
LINEAR_LINE_MIN_DENSITY = 0.45
LINEAR_BLOB_MAX_DENSITY = 0.70

# Default summary cell size (model grid).
DEFAULT_SUMMARY_RESOLUTION_M = 100


def woodland_halo_px(resolution_m: float = 1.0, erosion_m: float = WOODLAND_EROSION_M) -> int:
    """``map_overlap`` halo (px) for woodland opening: erode + dilate by ``r``."""
    r = max(1, int(round(erosion_m / resolution_m)))
    return 2 * r


def linear_halo_px(
    resolution_m: float = 1.0,
    max_width_m: float = LINEAR_MAX_WIDTH_M,
) -> int:
    """Extra halo (px) for candidate EDT width + elongation windows."""
    width_r = max(1, int(round((max_width_m / 2.0) / resolution_m)))
    elong_r = (max(LINEAR_LINE_WINDOW_PX) + 1) // 2
    return max(width_r, elong_r)


def mask_overlap_halo_px(
    resolution_m: float = 1.0,
    erosion_m: float = WOODLAND_EROSION_M,
    max_width_m: float = LINEAR_MAX_WIDTH_M,
) -> int:
    """Halo covering woodland then linear local ops in one ``map_overlap``."""
    return woodland_halo_px(resolution_m, erosion_m) + linear_halo_px(
        resolution_m, max_width_m
    )


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


def canopy_distance_px(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) from each True cell to the nearest False."""
    return ndimage.distance_transform_edt(np.asarray(mask, dtype=bool))


def _euclidean_opening(
    mask: np.ndarray,
    radius_px: int,
    *,
    distance_px: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Binary opening with a Euclidean disk via two EDT thresholds.

    ``core = edt(mask) > r``; grow back with ``edt(~core) <= r``; intersect mask.
    Local fixed-radius op — safe under ``map_overlap`` with halo ``2 * r``.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if radius_px < 1 or not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    if distance_px is None:
        distance_px = canopy_distance_px(mask_bool)
    core = distance_px > radius_px
    if not core.any():
        return np.zeros_like(mask_bool, dtype=bool)
    return mask_bool & (canopy_distance_px(~core) <= radius_px)


def woodland_from_canopy(
    canopy: np.ndarray,
    *,
    resolution_m: float = 1.0,
    erosion_m: float = WOODLAND_EROSION_M,
    distance_px: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Woodland via Euclidean morphological opening of canopy (no labelling).

    ``core = edt(canopy) > r`` then grow back by ``r`` and intersect canopy.
    Thick joined canopy keeps a core after ~10–15 m inward erosion; thin
    patches and hedges do not. Pass ``distance_px`` to reuse a canopy EDT.
    """
    radius_px = max(1, int(round(erosion_m / float(resolution_m))))
    return _euclidean_opening(
        canopy, radius_px, distance_px=distance_px
    )


def linear_woody_mask(
    canopy: np.ndarray,
    *,
    woodland: Optional[np.ndarray] = None,
    resolution_m: float = 1.0,
    max_width_m: float = LINEAR_MAX_WIDTH_M,
    canopy_distance_px_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Narrow elongated non-woodland canopy (hedges / tree lines).

    Local ops only (no labelling): candidates are canopy outside woodland;
    keep pixels with ``2 * edt(candidates) <= max_width_m`` that also look
    elongated under H/V vs square uniform-density windows.
    """
    canopy_bool = np.asarray(canopy, dtype=bool)
    if woodland is None:
        woodland = woodland_from_canopy(
            canopy_bool,
            resolution_m=resolution_m,
            distance_px=canopy_distance_px_arr,
        )
    candidates = canopy_bool & ~np.asarray(woodland, dtype=bool)
    if not candidates.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    half_width_m = canopy_distance_px(candidates) * float(resolution_m)
    narrow = candidates & ((2.0 * half_width_m) <= float(max_width_m))
    if not narrow.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    short_px, long_px = LINEAR_LINE_WINDOW_PX
    dens = candidates.astype(np.float64)
    line_h = ndimage.uniform_filter(dens, size=(short_px, long_px))
    line_v = ndimage.uniform_filter(dens, size=(long_px, short_px))
    blob = ndimage.uniform_filter(dens, size=LINEAR_BLOB_WINDOW_PX)
    elongated = (
        (np.maximum(line_h, line_v) >= LINEAR_LINE_MIN_DENSITY)
        & (blob <= LINEAR_BLOB_MAX_DENSITY)
    )
    return narrow & elongated


def canopy_edge_mask(canopy: np.ndarray) -> np.ndarray:
    """Canopy pixels that touch a non-canopy neighbour (8-connected edge).

    Uses a 3×3 all-ones structuring element (diagonal neighbours count).
    Docstring previously said 4-connected; code was always 8-connected —
    docstring updated to match.
    """
    canopy_bool = np.asarray(canopy, dtype=bool)
    eroded = ndimage.binary_erosion(
        canopy_bool, structure=np.ones((3, 3), dtype=bool)
    )
    return canopy_bool & ~eroded


def build_canopy_masks(
    height: np.ndarray,
    *,
    resolution_m: float = 1.0,
) -> CanopyMasks:
    """Derive all 1 m canopy / patch / band masks from a height array.

    Reuses one canopy EDT for woodland opening; linear uses a second EDT on
    non-woodland candidates. Both are local fixed-radius ops.
    """
    canopy = canopy_mask(height)
    low, mid, high = height_band_masks(height, canopy=canopy)
    canopy_edt = canopy_distance_px(canopy)
    woodland = woodland_from_canopy(
        canopy, resolution_m=resolution_m, distance_px=canopy_edt
    )
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
    """Aggregate 1 m masks to summary cells (share or length per hectare).

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

    # Linear length: skeleton of linear mask × resolution / hectares.
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
    out: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(n_cells):
        denom = (
            expected_pixels
            if n_pix[i] >= expected_pixels * 0.99
            else float(n_pix[i])
        )
        if denom <= 0:
            continue
        if h_n[i] >= 2:
            mean = h_sum[i] / h_n[i]
            std = float(np.sqrt(max(h_sumsq[i] / h_n[i] - mean * mean, 0.0)))
        elif h_n[i] == 1:
            std = 0.0
        else:
            std = float("nan")
        key = (int(uniq[i, 0]), int(uniq[i, 1]))
        out[key] = {
            "vom_canopy_share": float(sum_canopy[i] / denom),
            "vom_woodland_from_canopy_share": float(sum_wood[i] / denom),
            "vom_trees_outside_woodland_share": float(sum_tow[i] / denom),
            "vom_canopy_height_band_2p5_5_share": float(sum_low[i] / denom),
            "vom_canopy_height_band_5_15_share": float(sum_mid[i] / denom),
            "vom_canopy_height_band_gt15_share": float(sum_high[i] / denom),
            "vom_linear_woody_length_ha": float(sum_skel[i] / cell_area_ha),
            "vom_canopy_edge_length_ha": float(sum_edge[i] / cell_area_ha),
            "vom_canopy_height_std": std,
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
    # Nodata-aware block mean: nanmean so empty blocks stay empty.
    blocks = h_trim.reshape(ny_c // factor, factor, nx_c // factor, factor)
    with np.errstate(all="ignore"):
        coarse = np.nanmean(blocks, axis=(1, 3))

    # Upsample coarse canopy decision back to 1 m for pixel-wise retention.
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
    "LINEAR_MAX_WIDTH_M",
    "WOODLAND_EROSION_M",
    "CanopyMasks",
    "build_canopy_masks",
    "canopy_distance_px",
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
    "woodland_from_canopy",
    "woodland_halo_px",
]
