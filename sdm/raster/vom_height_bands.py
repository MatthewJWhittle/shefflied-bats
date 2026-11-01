"""Simple VOM height-band and woody-cover summaries for 100 m cells (#121).

Array ops only: threshold masks, coarsen shares / std, optional box-mean
neighbourhood. No woodland/TOW labelling, distance transforms, or patch loops.
Safe under Dask ``map_blocks`` with no halo when summarising aligned chunks.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import xarray as xr

# Official VOM inclusion threshold (metres) — top-of-canopy objects above this.
CANOPY_MIN_HEIGHT_M = 2.5

# Contiguous height bands (metres) on canopy cells: [2.5, 5), [5, 15], (15, ∞).
HEIGHT_BAND_LOW = (2.5, 5.0)
HEIGHT_BAND_MID = (5.0, 15.0)
HEIGHT_BAND_HIGH_MIN = 15.0

DEFAULT_SUMMARY_RESOLUTION_M = 100

# Short band stems written to the GeoTIFF; merge prefix ``vom`` yields final ids.
CELL_LAYER_STEMS: tuple[str, ...] = (
    "woody_cover_share",
    "height_band_2p5_5_share",
    "height_band_5_15_share",
    "height_band_gt15_share",
    "canopy_height_std",
)

LAYER_IDS: tuple[str, ...] = tuple(f"vom_{stem}" for stem in CELL_LAYER_STEMS)


def canopy_mask(
    height: np.ndarray,
    *,
    min_height_m: float = CANOPY_MIN_HEIGHT_M,
) -> np.ndarray:
    """Binary canopy: finite heights at or above the VOM threshold."""
    h = np.asarray(height, dtype=np.float64)
    return np.isfinite(h) & (h >= min_height_m)


def height_band_masks(
    height: np.ndarray,
    *,
    canopy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height-band masks on canopy cells (2.5–5, 5–15, >15 m)."""
    h = np.asarray(height, dtype=np.float64)
    if canopy is None:
        canopy = canopy_mask(h)
    low = canopy & (h >= HEIGHT_BAND_LOW[0]) & (h < HEIGHT_BAND_LOW[1])
    mid = canopy & (h >= HEIGHT_BAND_MID[0]) & (h <= HEIGHT_BAND_MID[1])
    high = canopy & (h > HEIGHT_BAND_HIGH_MIN)
    return low, mid, high


def _block_reduce_sum(mask: np.ndarray, factor: int) -> np.ndarray:
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
    h = np.asarray(height, dtype=np.float64)
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
    finite = np.isfinite(blocks)
    n = np.sum(finite, axis=(1, 3))
    filled = np.where(finite, blocks, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(filled, axis=(1, 3)) / np.maximum(n, 1)
        mean_sq = np.sum(filled * filled, axis=(1, 3)) / np.maximum(n, 1)
        var = mean_sq - mean * mean
    out = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), out)
    out = np.where(n == 1, 0.0, out)
    return out


def summarise_height_to_grid(
    height: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
) -> dict[str, np.ndarray]:
    """Coarsen 1 m heights to cell shares and canopy-height std (no loops)."""
    if summary_resolution_m % int(fine_resolution_m) != 0:
        raise ValueError(
            f"summary_resolution_m ({summary_resolution_m}) must be a multiple "
            f"of fine_resolution_m ({fine_resolution_m})"
        )
    factor = int(round(summary_resolution_m / fine_resolution_m))
    cell_pixels = float(factor * factor)
    canopy = canopy_mask(height)
    low, mid, high = height_band_masks(height, canopy=canopy)
    return {
        "woody_cover_share": _block_reduce_sum(canopy, factor) / cell_pixels,
        "height_band_2p5_5_share": _block_reduce_sum(low, factor) / cell_pixels,
        "height_band_5_15_share": _block_reduce_sum(mid, factor) / cell_pixels,
        "height_band_gt15_share": _block_reduce_sum(high, factor) / cell_pixels,
        "canopy_height_std": _block_reduce_canopy_std(height, canopy, factor),
    }


def summarise_height_block(
    block: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    max_height_m: float = 40.0,
) -> np.ndarray:
    """``map_blocks`` kernel: height → stacked float32 summaries (or all-NaN).

    Failures are swallowed so one bad chunk does not abort siblings; the caller
    counts all-NaN outputs as failed chunks.
    """
    try:
        arr = np.asarray(block, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        finite = np.isfinite(arr)
        capped = arr.copy()
        capped[finite & (capped > max_height_m)] = max_height_m
        layers = summarise_height_to_grid(
            capped,
            fine_resolution_m=fine_resolution_m,
            summary_resolution_m=summary_resolution_m,
        )
        stacked = np.stack(
            [layers[stem].astype(np.float32) for stem in CELL_LAYER_STEMS],
            axis=0,
        )
        return stacked
    except Exception:
        factor = int(round(summary_resolution_m / fine_resolution_m))
        ny = max(0, (np.asarray(block).shape[-2] // factor))
        nx = max(0, (np.asarray(block).shape[-1] // factor))
        return np.full((len(CELL_LAYER_STEMS), ny, nx), np.nan, dtype=np.float32)


def summaries_to_dataset(
    layers: Mapping[str, np.ndarray],
    *,
    y: np.ndarray,
    x: np.ndarray,
    crs: str,
    transform,
) -> xr.Dataset:
    """Build a CRS-aware Dataset from stem → 2-D arrays."""
    data_vars = {
        stem: (("y", "x"), np.asarray(values, dtype=np.float32))
        for stem, values in layers.items()
    }
    ds = xr.Dataset(data_vars, coords={"y": y, "x": x})
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)
    for stem in ds.data_vars:
        ds[stem].attrs["long_name"] = stem
        ds[stem].rio.write_nodata(np.nan, inplace=True)
    return ds


def pairwise_pearson(
    layers: Mapping[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Pearson r for every unordered pair (finite cells only)."""
    names = list(layers)
    flat = {
        name: np.asarray(values, dtype=np.float64).ravel() for name, values in layers.items()
    }
    out: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            mask = np.isfinite(flat[left]) & np.isfinite(flat[right])
            if int(mask.sum()) < 3:
                out[(left, right)] = float("nan")
                continue
            out[(left, right)] = float(np.corrcoef(flat[left][mask], flat[right][mask])[0, 1])
    return out


__all__ = [
    "CANOPY_MIN_HEIGHT_M",
    "CELL_LAYER_STEMS",
    "DEFAULT_SUMMARY_RESOLUTION_M",
    "HEIGHT_BAND_HIGH_MIN",
    "HEIGHT_BAND_LOW",
    "HEIGHT_BAND_MID",
    "LAYER_IDS",
    "canopy_mask",
    "height_band_masks",
    "pairwise_pearson",
    "summaries_to_dataset",
    "summarise_height_block",
    "summarise_height_to_grid",
]
