"""Simple VOM height-band and woody-cover summaries for 100 m cells (#121).

Array ops only: threshold masks, xarray ``coarsen`` shares / std, optional
box-mean neighbourhood. No woodland/TOW labelling, distance transforms, or
patch loops. Safe under Dask when chunks are multiples of the coarsen factor.
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


def _as_yx_dataarray(height: np.ndarray) -> xr.DataArray:
    arr = np.asarray(height, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return xr.DataArray(arr, dims=("y", "x"))


def summarise_height_to_grid(
    height: np.ndarray | xr.DataArray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
) -> dict[str, np.ndarray]:
    """Coarsen 1 m heights to cell shares and canopy-height std via xarray.

    ``canopy_height_std`` is 0 where woody cover share is 0 (no canopy in the
    cell) so MaxEnt does not see NaNs on open land.
    """
    if summary_resolution_m % int(fine_resolution_m) != 0:
        raise ValueError(
            f"summary_resolution_m ({summary_resolution_m}) must be a multiple "
            f"of fine_resolution_m ({fine_resolution_m})"
        )
    factor = int(round(summary_resolution_m / fine_resolution_m))
    h = height if isinstance(height, xr.DataArray) else _as_yx_dataarray(height)
    if h.dims != ("y", "x"):
        h = h.transpose("y", "x")

    ny = (int(h.sizes["y"]) // factor) * factor
    nx = (int(h.sizes["x"]) // factor) * factor
    if ny == 0 or nx == 0:
        empty = {
            stem: np.zeros((0, 0), dtype=np.float64) for stem in CELL_LAYER_STEMS
        }
        empty["canopy_height_std"] = np.full((0, 0), np.nan, dtype=np.float64)
        return empty
    h = h.isel(y=slice(0, ny), x=slice(0, nx))

    canopy = (h.notnull() & (h >= CANOPY_MIN_HEIGHT_M)).astype("float64")
    low = (
        canopy
        * (h >= HEIGHT_BAND_LOW[0]).astype("float64")
        * (h < HEIGHT_BAND_LOW[1]).astype("float64")
    )
    mid = (
        canopy
        * (h >= HEIGHT_BAND_MID[0]).astype("float64")
        * (h <= HEIGHT_BAND_MID[1]).astype("float64")
    )
    high = canopy * (h > HEIGHT_BAND_HIGH_MIN).astype("float64")

    coarse = dict(y=factor, x=factor)
    woody = canopy.coarsen(coarse, boundary="trim").mean()
    band_low = low.coarsen(coarse, boundary="trim").mean()
    band_mid = mid.coarsen(coarse, boundary="trim").mean()
    band_high = high.coarsen(coarse, boundary="trim").mean()
    # Std only among canopy cells (non-canopy → NaN before coarsen).
    h_canopy = h.where(canopy > 0)
    # ddof=0 so a single canopy pixel yields 0 (not NaN / warning).
    std = h_canopy.coarsen(coarse, boundary="trim").std(skipna=True, ddof=0)
    # Open land (no woody cover): treat structural variability as 0, not NaN.
    std = std.where(woody > 0, 0.0)

    return {
        "woody_cover_share": np.asarray(woody.values, dtype=np.float64),
        "height_band_2p5_5_share": np.asarray(band_low.values, dtype=np.float64),
        "height_band_5_15_share": np.asarray(band_mid.values, dtype=np.float64),
        "height_band_gt15_share": np.asarray(band_high.values, dtype=np.float64),
        "canopy_height_std": np.asarray(std.values, dtype=np.float64),
    }


def summarise_height_block(
    block: np.ndarray,
    *,
    fine_resolution_m: float = 1.0,
    summary_resolution_m: int = DEFAULT_SUMMARY_RESOLUTION_M,
    max_height_m: float = 40.0,
) -> np.ndarray:
    """Per-chunk kernel: height → stacked float32 summaries (or all-NaN).

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
        return np.stack(
            [layers[stem].astype(np.float32) for stem in CELL_LAYER_STEMS],
            axis=0,
        )
    except Exception:
        factor = int(round(summary_resolution_m / fine_resolution_m))
        shape = np.asarray(block).shape
        ny = max(0, (shape[-2] // factor) if len(shape) >= 2 else 0)
        nx = max(0, (shape[-1] // factor) if len(shape) >= 2 else 0)
        return np.full((len(CELL_LAYER_STEMS), ny, nx), np.nan, dtype=np.float32)


def fill_canopy_std_where_no_woody(
    canopy_height_std: np.ndarray,
    woody_cover_share: np.ndarray,
) -> np.ndarray:
    """Set canopy SD to 0 where woody cover share is 0 (open land)."""
    std = np.asarray(canopy_height_std, dtype=np.float64).copy()
    woody = np.asarray(woody_cover_share, dtype=np.float64)
    open_land = np.isfinite(woody) & (woody <= 0.0)
    std[open_land] = 0.0
    return std


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
        name: np.asarray(values, dtype=np.float64).ravel()
        for name, values in layers.items()
    }
    out: dict[tuple[str, str], float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            mask = np.isfinite(flat[left]) & np.isfinite(flat[right])
            if int(mask.sum()) < 3:
                out[(left, right)] = float("nan")
                continue
            out[(left, right)] = float(
                np.corrcoef(flat[left][mask], flat[right][mask])[0, 1]
            )
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
    "fill_canopy_std_where_no_woody",
    "height_band_masks",
    "pairwise_pearson",
    "summaries_to_dataset",
    "summarise_height_block",
    "summarise_height_to_grid",
]
