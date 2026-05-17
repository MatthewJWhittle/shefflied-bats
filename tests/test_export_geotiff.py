"""Tests for ``export_geotiff`` and ``export_raster_paths``."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

from sdm.commands.modelling.export_rasters import export_raster_paths
from sdm.raster.io import export_geotiff


@pytest.fixture
def tiny_gtiff_27700(tmp_path: Path) -> Path:
    path = tmp_path / "demo.tif"
    h, w = 8, 10
    data = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    transform = from_bounds(
        400_000,
        500_000,
        400_000 + w * 100,
        500_000 + h * 100,
        w,
        h,
    )
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:27700",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_export_geotiff_cog_same_crs(tiny_gtiff_27700: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out_cog.tif"
    export_geotiff(tiny_gtiff_27700, dst, as_cog=True)
    with rasterio.open(dst) as ds:
        assert ds.crs.to_epsg() == 27700
    ok, _, errs = cog_validate(dst, quiet=True)
    assert ok, errs


def test_export_geotiff_warp_3857_plain(tiny_gtiff_27700: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out_warp.tif"
    export_geotiff(tiny_gtiff_27700, dst, dst_crs="EPSG:3857", as_cog=False)
    with rasterio.open(dst) as ds:
        assert ds.crs.to_epsg() == 3857


def test_export_geotiff_warp_3857_cog(tiny_gtiff_27700: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out_both.tif"
    export_geotiff(tiny_gtiff_27700, dst, dst_crs="EPSG:3857", as_cog=True)
    with rasterio.open(dst) as ds:
        assert ds.crs.to_epsg() == 3857
    ok, _, errs = cog_validate(dst, quiet=True)
    assert ok, errs


def test_export_geotiff_same_src_dst_raises(tiny_gtiff_27700: Path) -> None:
    with pytest.raises(ValueError, match="Destination must differ"):
        export_geotiff(tiny_gtiff_27700, tiny_gtiff_27700, as_cog=True)


def test_export_raster_paths_requires_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Nothing to do"):
        export_raster_paths([], tmp_path / "out")


def test_export_raster_paths_duplicate_stems(tiny_gtiff_27700: Path, tmp_path: Path) -> None:
    other = tmp_path / "nested" / "demo.tif"
    other.parent.mkdir()
    shutil.copy2(tiny_gtiff_27700, other)
    out_dir = tmp_path / "exports"
    paths = export_raster_paths(
        [tiny_gtiff_27700, other],
        out_dir,
        output_crs="EPSG:3857",
        as_cog=False,
    )
    assert len(paths) == 2
    assert paths[0].name == "demo_EPSG3857.tif"
    assert paths[1].name == "demo_EPSG3857_2.tif"
