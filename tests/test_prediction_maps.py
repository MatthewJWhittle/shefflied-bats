"""``plot_prediction_map`` must build a colorbar from the AxesImage mappable.

``rasterio.plot.show(..., ax=ax)`` returns the Axes. Passing that Axes to
``plt.colorbar`` raises ``AttributeError: 'Axes' object has no property 'cmap'``.
Synthetic raster only; Agg backend (no display).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sdm.viz.prediction_maps import _mappable_from_axes, plot_prediction_map


def _write_synthetic_suitability(path: Path, size: int = 8) -> None:
    """Tiny GeoTIFF with values in [0, 1] — no real locations."""
    data = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    transform = from_origin(0.0, float(size), 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:27700",
        "transform": transform,
        "nodata": -9999.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_plot_prediction_map_writes_png_with_colorbar(tmp_path: Path) -> None:
    raster = tmp_path / "suitability.tif"
    out = tmp_path / "preview.png"
    _write_synthetic_suitability(raster)

    plot_prediction_map(raster, output_path=out, title="synthetic")

    assert out.is_file()
    assert out.stat().st_size > 0


def test_mappable_from_axes_requires_image() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    with pytest.raises(RuntimeError, match="No image artists"):
        _mappable_from_axes(ax)
    matplotlib.pyplot.close(fig)
