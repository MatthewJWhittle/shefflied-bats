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
from skimage.morphology import disk, erosion, skeletonize

# Official VOM inclusion threshold (metres) — top-of-canopy objects above this.
CANOPY_MIN_HEIGHT_M = 2.5

# Height bands (metres) applied only where canopy is present.
HEIGHT_BAND_LOW = (2.5, 5.0)
HEIGHT_BAND_MID = (5.0, 15.0)
# Open upper bound: canopy cells with height > 15 m.
HEIGHT_BAND_HIGH_MIN = 15.0

# Woodland-from-canopy: joined canopy surviving inward erosion, ~0.5 ha.
WOODLAND_EROSION_M = 12.0  # mid of ~10–15 m inward erosion
WOODLAND_MIN_AREA_M2 = 5_000.0  # 0.5 ha

# Linear woody: narrow elongated canopy (hedges / tree lines).
LINEAR_MAX_WIDTH_M = 20.0
LINEAR_MIN_ASPECT = 3.0
LINEAR_MIN_AREA_M2 = 50.0

# Default summary cell size (model grid).
DEFAULT_SUMMARY_RESOLUTION_M = 100


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


def woodland_from_canopy(
    canopy: np.ndarray,
    *,
    resolution_m: float = 1.0,
    erosion_m: float = WOODLAND_EROSION_M,
    min_area_m2: float = WOODLAND_MIN_AREA_M2,
) -> np.ndarray:
    """Woodland mask from canopy alone via inward erosion + area filter.

    Connected canopy that still has a core after ~10–15 m erosion, and whose
    full joined canopy area is at least ``min_area_m2`` (~0.5 ha), is treated
    as woodland. Remaining canopy is trees-outside-woodland / small patches.
    """
    canopy_bool = np.asarray(canopy, dtype=bool)
    if not canopy_bool.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    radius_px = max(1, int(round(erosion_m / resolution_m)))
    core = erosion(canopy_bool, disk(radius_px))
    if not core.any():
        return np.zeros_like(canopy_bool, dtype=bool)

    # Label the original canopy; keep components that intersect a surviving core
    # and meet the area threshold.
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


def linear_woody_mask(
    canopy: np.ndarray,
    *,
    woodland: Optional[np.ndarray] = None,
    resolution_m: float = 1.0,
    max_width_m: float = LINEAR_MAX_WIDTH_M,
    min_aspect: float = LINEAR_MIN_ASPECT,
    min_area_m2: float = LINEAR_MIN_AREA_M2,
) -> np.ndarray:
    """Narrow elongated canopy (hedges / tree lines), excluding woodland cores.

    Components of non-woodland canopy with small mean width (via EDT) and high
    aspect ratio are marked linear.
    """
    canopy_bool = np.asarray(canopy, dtype=bool)
    if woodland is None:
        woodland = woodland_from_canopy(canopy_bool, resolution_m=resolution_m)
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
        # Aspect from covariance of coordinates (major/minor axis ratio).
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


def canopy_edge_mask(canopy: np.ndarray) -> np.ndarray:
    """Canopy pixels that touch a non-canopy neighbour (4-connected edge)."""
    canopy_bool = np.asarray(canopy, dtype=bool)
    eroded = ndimage.binary_erosion(canopy_bool, structure=np.ones((3, 3), dtype=bool))
    return canopy_bool & ~eroded


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
    out = np.full((ny_c // factor, nx_c // factor), np.nan, dtype=np.float64)
    if ny_c == 0 or nx_c == 0:
        return out
    for iy in range(out.shape[0]):
        for ix in range(out.shape[1]):
            y0, y1 = iy * factor, (iy + 1) * factor
            x0, x1 = ix * factor, (ix + 1) * factor
            vals = h[y0:y1, x0:x1][can[y0:y1, x0:x1]]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                out[iy, ix] = float(vals.std(ddof=0))
            elif vals.size == 1:
                out[iy, ix] = 0.0
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
) -> dict[tuple[int, int], dict[str, float]]:
    """Aggregate fine masks into national-grid 100 m cells by coordinates.

    Keys are ``(floor(y / res), floor(x / res))`` in CRS metres. Partial tile
    overlap is normalised by observed fine pixels; full cells use the expected
    pixel count so shares stay on ``[0, 1]``.
    """
    res = float(summary_resolution_m)
    fine = float(fine_resolution_m)
    cell_area_ha = (res ** 2) / 10_000.0
    xs = np.asarray(x_coords, dtype=np.float64)
    ys = np.asarray(y_coords, dtype=np.float64)
    col = np.floor(xs[None, :] / res).astype(np.int64)
    row = np.floor(ys[:, None] / res).astype(np.int64)
    keys = np.stack([row.ravel(), col.ravel()], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = uniq.shape[0]

    def _sum(mask: np.ndarray) -> np.ndarray:
        return np.bincount(
            inv, weights=np.asarray(mask, dtype=np.float64).ravel(), minlength=n_cells
        )

    canopy = np.asarray(masks.canopy, dtype=bool)
    h = _finite_height(height)
    canopy_vals = np.where(canopy & np.isfinite(h), h, np.nan)
    canopy_weight = np.isfinite(canopy_vals).astype(np.float64)
    h_filled = np.where(np.isfinite(canopy_vals), canopy_vals, 0.0)

    linear_skel = skeletonize(np.asarray(masks.linear, dtype=bool))
    n_pix = np.bincount(inv, minlength=n_cells).astype(np.float64)
    sum_canopy = _sum(canopy)
    sum_wood = _sum(masks.woodland)
    sum_tow = _sum(masks.trees_outside_woodland)
    sum_low = _sum(masks.band_2p5_5)
    sum_mid = _sum(masks.band_5_15)
    sum_high = _sum(masks.band_gt15)
    sum_skel = _sum(linear_skel) * fine
    sum_edge = _sum(masks.canopy_edge) * fine
    h_n = np.bincount(inv, weights=canopy_weight.ravel(), minlength=n_cells)
    h_sum = np.bincount(inv, weights=h_filled.ravel(), minlength=n_cells)
    h_sumsq = np.bincount(inv, weights=(h_filled ** 2).ravel(), minlength=n_cells)

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
    "LINEAR_MIN_AREA_M2",
    "LINEAR_MIN_ASPECT",
    "WOODLAND_EROSION_M",
    "WOODLAND_MIN_AREA_M2",
    "CanopyMasks",
    "build_canopy_masks",
    "canopy_edge_mask",
    "canopy_mask",
    "compare_narrow_canopy_retention",
    "height_band_masks",
    "linear_woody_mask",
    "masks_to_dataarrays",
    "summarise_masks_by_coordinates",
    "summarise_masks_to_grid",
    "woodland_from_canopy",
]
