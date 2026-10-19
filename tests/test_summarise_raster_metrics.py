"""Tests for summarise_raster_metrics (VOM 100 m window stats)."""

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from rasterio.transform import Affine, from_origin

from sdm.raster.processing import summarise_raster_metrics


def _synthetic_10m_raster(
    *,
    height: int = 30,
    width: int = 30,
    resolution: float = 10.0,
    origin_x: float = 1000.0,
    origin_y: float = 2000.0,
) -> xr.DataArray:
    """Build a synthetic EPSG:27700 raster at ``resolution`` metres."""
    transform = from_origin(origin_x, origin_y, resolution, resolution)
    data = np.arange(height * width, dtype=np.float64).reshape(height, width)
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={
            "x": origin_x + (np.arange(width) + 0.5) * resolution,
            "y": origin_y - (np.arange(height) + 0.5) * resolution,
        },
        name="height",
    )
    return da.rio.write_crs("EPSG:27700").rio.write_transform(transform)


def _plain_block_stats(values: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    """Window mean/min/max/std over non-overlapping ``scale``×``scale`` blocks."""
    n_y, n_x = values.shape
    assert n_y % scale == 0 and n_x % scale == 0
    blocks = values.reshape(n_y // scale, scale, n_x // scale, scale)
    return {
        "mean": blocks.mean(axis=(1, 3)),
        "min": blocks.min(axis=(1, 3)),
        "max": blocks.max(axis=(1, 3)),
        "std": blocks.std(axis=(1, 3)),
    }


def test_summarise_raster_metrics_100m_matches_block_stats_and_transform():
    """100 m coarsened stats must match plain block stats with a 100 m transform.

    Rebuild failure mode (#38): ``summarise_raster_metrics`` stamped the
    coarsened Dataset with the source 10 m transform (rioxarray's cached
    affine after ``coarsen``). Later resample to the model grid then smeared
    the window stats. The output transform must be recalculated from the
    coarsened coordinates (e.g. ``rio.transform(recalc=True)``).
    """
    source = _synthetic_10m_raster()
    scale = 10
    target_resolution = 100

    summary = summarise_raster_metrics(
        source,
        target_resolution=target_resolution,
        var_name_prefix="vom_height",
        boundary_handling="trim",
    )

    expected = _plain_block_stats(np.asarray(source.values), scale)
    for stat, block_values in expected.items():
        var = f"vom_height_{stat}_{target_resolution}m"
        assert var in summary.data_vars
        np.testing.assert_allclose(
            summary[var].values,
            block_values,
            err_msg=f"{var} must equal plain {scale}×{scale} block {stat}",
        )

    expected_transform = Affine(
        target_resolution,
        0.0,
        1000.0,
        0.0,
        -target_resolution,
        2000.0,
    )
    assert summary.rio.transform() == expected_transform, (
        f"expected 100 m transform {expected_transform}, "
        f"got {summary.rio.transform()} "
        f"(recalc={summary.rio.transform(recalc=True)})"
    )
    assert summary.rio.resolution() == (float(target_resolution), -float(target_resolution))
