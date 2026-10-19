"""Neighbourhood / multiscale rolling must stay aligned on north-up rasters."""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

from sdm.raster.processing import calculate_multiscale_variables

RESOLUTION_M = 100.0
# Production multiscale window (10 cells → 500 m suffix at 100 m resolution).
WINDOW_CELLS = 10
WINDOW_MEAN = 1.0 / (WINDOW_CELLS * WINDOW_CELLS)


def _hot_cell_dataset(*, north_up: bool, size: int = 31, hot: int = 15) -> tuple[xr.Dataset, float, float]:
    """Synthetic single-hot-cell grid; north-up vs south-up differ only in row order."""
    x = np.arange(size, dtype=np.float64) * RESOLUTION_M + 400_000.0
    y_south_to_north = np.arange(size, dtype=np.float64) * RESOLUTION_M + 400_000.0
    data_south_to_north = np.zeros((size, size), dtype=np.float64)
    data_south_to_north[hot, hot] = 1.0
    hot_y = float(y_south_to_north[hot])
    hot_x = float(x[hot])

    if north_up:
        y = y_south_to_north[::-1].copy()
        data = data_south_to_north[::-1].copy()
        transform = Affine(
            RESOLUTION_M,
            0.0,
            float(x[0] - RESOLUTION_M / 2),
            0.0,
            -RESOLUTION_M,
            float(y[0] + RESOLUTION_M / 2),
        )
    else:
        y = y_south_to_north.copy()
        data = data_south_to_north.copy()
        transform = Affine(
            RESOLUTION_M,
            0.0,
            float(x[0] - RESOLUTION_M / 2),
            0.0,
            RESOLUTION_M,
            float(y[0] - RESOLUTION_M / 2),
        )

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="cover",
    )
    da = da.rio.write_crs("EPSG:27700")
    da = da.rio.write_transform(transform)
    da.rio.write_nodata(np.nan, inplace=True)
    return da.to_dataset(), hot_y, hot_x


def _positive_dy_dx(result: xr.DataArray, hot_y: float, hot_x: float) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(result.values > 0)
    dy = result.y.values[ys] - hot_y
    dx = result.x.values[xs] - hot_x
    return np.asarray(dy, dtype=float), np.asarray(dx, dtype=float)


def test_neighbourhood_mean_keeps_hot_cell_on_north_up() -> None:
    """North-up impulse must stay on the hot cell after the neighbourhood mean.

    ``rolling(..., center=True)`` with an even window is half-cell asymmetric toward
    higher array indices. On a north-up raster (descending ``y``) that bias points
    south, so the neighbourhood pattern shifts one row unless the shared helper
    normalises orientation first.
    """
    dataset, hot_y, hot_x = _hot_cell_dataset(north_up=True)
    out = calculate_multiscale_variables(dataset, WINDOW_CELLS)
    band = f"cover_{WINDOW_CELLS // 2 * int(RESOLUTION_M)}m"
    result = out[band]

    assert float(result.sel(y=hot_y, x=hot_x)) == WINDOW_MEAN

    dy, dx = _positive_dy_dx(result, hot_y, hot_x)
    # Hot cell itself must receive the impulse (same cell, not a neighbour only).
    assert 0.0 in dy
    assert 0.0 in dx
    # Even-window convention (ascending-y / historical south-up): one extra cell
    # toward increasing coordinates — north and east — not toward the south.
    assert dy.max() == -dy.min() + RESOLUTION_M
    assert dx.max() == -dx.min() + RESOLUTION_M


def test_neighbourhood_mean_north_up_matches_south_up() -> None:
    """North-up and south-up inputs must yield the same geographically aligned result."""
    north, hot_y, hot_x = _hot_cell_dataset(north_up=True)
    south, _, _ = _hot_cell_dataset(north_up=False)

    band = f"cover_{WINDOW_CELLS // 2 * int(RESOLUTION_M)}m"
    out_north = calculate_multiscale_variables(north, WINDOW_CELLS)[band]
    out_south = calculate_multiscale_variables(south, WINDOW_CELLS)[band]

    aligned_south = out_south.reindex(y=out_north.y, x=out_north.x)
    np.testing.assert_allclose(out_north.values, aligned_south.values, equal_nan=True)

    assert float(out_north.sel(y=hot_y, x=hot_x)) == float(
        out_south.sel(y=hot_y, x=hot_x)
    )
