"""Post-process helpers for environmental-layer data-quality fixes.

Distance-limited nearest fill for thin climate / vegetation-height edge
nodata (#74). Large sea / coverage holes stay empty: they are owned by
the land-clip (#78) and NE-gap (#79) work, not by unbounded interpolation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import rioxarray  # noqa: F401 — enables DataArray.rio
import xarray as xr
from scipy.ndimage import distance_transform_edt

# Thin land-edge / coast mismatch only (~300 m at 100 m cells). Wider sea
# or NE coverage holes stay nodata for #78 / #79.
DEFAULT_NODATA_FILL_MAX_CELLS: float = 3.0

ArrayLike = np.ndarray
XArrayLike = Union[xr.DataArray, xr.Dataset]


def fill_nodata_nearest(
    data: ArrayLike,
    *,
    max_search_distance_cells: float = DEFAULT_NODATA_FILL_MAX_CELLS,
) -> np.ndarray:
    """Fill NaN cells from the nearest finite neighbour, distance-limited.

    Cells farther than ``max_search_distance_cells`` from any valid cell
    remain NaN so multi-kilometre sea buffers and coverage holes are not
    invented from distant land values.
    """
    if max_search_distance_cells < 0:
        raise ValueError("max_search_distance_cells must be >= 0")

    arr = np.array(data, dtype=np.float32, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"fill_nodata_nearest expects a 2-D array, got shape {arr.shape}")

    valid = np.isfinite(arr)
    if not valid.any() or valid.all():
        return arr

    dist, (iy, ix) = distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True,
    )
    fill = (~valid) & (dist <= float(max_search_distance_cells))
    arr[fill] = arr[iy[fill], ix[fill]]
    return arr


def _fill_ndarray(values: np.ndarray, max_search_distance_cells: float) -> np.ndarray:
    """Fill 2-D or banded (band, y, x) arrays."""
    if values.ndim == 2:
        return fill_nodata_nearest(
            values, max_search_distance_cells=max_search_distance_cells
        )
    if values.ndim == 3:
        bands = [
            fill_nodata_nearest(
                values[i], max_search_distance_cells=max_search_distance_cells
            )
            for i in range(values.shape[0])
        ]
        return np.stack(bands, axis=0)
    raise ValueError(
        f"Unsupported array ndim {values.ndim}; expected 2 (y,x) or 3 (band,y,x)"
    )


def fill_xarray_nodata_nearest(
    obj: XArrayLike,
    *,
    max_search_distance_cells: float = DEFAULT_NODATA_FILL_MAX_CELLS,
) -> XArrayLike:
    """Distance-limited nearest fill for a DataArray or every Dataset variable."""
    if isinstance(obj, xr.DataArray):
        filled = _fill_ndarray(np.asarray(obj.values), max_search_distance_cells)
        out = obj.copy(data=filled)
        if obj.rio.crs is not None:
            out = out.rio.write_crs(obj.rio.crs)
        nodata = obj.rio.nodata
        if nodata is not None:
            out = out.rio.write_nodata(nodata)
        else:
            out = out.rio.write_nodata(np.nan)
        return out

    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Expected DataArray or Dataset, got {type(obj)!r}")

    out_ds = obj.copy(deep=True)
    for name in list(out_ds.data_vars):
        out_ds[name] = fill_xarray_nodata_nearest(
            out_ds[name],
            max_search_distance_cells=max_search_distance_cells,
        )
    return out_ds
