"""Tests for the artificial light at night (ALAN / VIIRS) layer (#37)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from shapely.geometry import box

from sdm.commands.data_preparation.environmental.generate_alan_data import (
    generate_alan_data,
)
from sdm.data.alan import (
    ALAN_BAND_NAME,
    ALAN_OUTPUT_NAME,
    load_alan_source,
    reproject_alan_to_model_grid,
)
from sdm.raster.utils import construct_transform_shift_bounds, grid_shape_from_bounds


def _write_coarse_viirs(path: Path, *, crs: str = "EPSG:27700") -> None:
    """Synthetic ~500 m radiance grid covering 400000–402000 east/north."""
    height = width = 4
    values = np.array(
        [
            [0.1, 0.2, 0.2, 0.1],
            [0.2, 8.0, 7.5, 0.2],
            [0.2, 7.0, 6.5, 0.2],
            [0.1, 0.2, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    # North-up: origin at NW corner (400000 E, 402000 N).
    transform = Affine(500.0, 0.0, 400_000.0, 0.0, -500.0, 402_000.0)
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": np.arange(height) * -500.0 + 402_000.0 - 250.0,
            "x": np.arange(width) * 500.0 + 400_000.0 + 250.0,
        },
        name=ALAN_BAND_NAME,
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)
    da.attrs["long_name"] = ALAN_BAND_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path)


def test_load_alan_source_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tif"
    with pytest.raises(FileNotFoundError, match="ALAN source not found"):
        load_alan_source(missing)


def test_reproject_alan_matches_model_grid(tmp_path: Path) -> None:
    source_path = tmp_path / "viirs.tif"
    _write_coarse_viirs(source_path)
    source = load_alan_source(source_path)

    # Same footprint as the synthetic VIIRS tile (must overlap).
    bounds = (400_000.0, 400_000.0, 402_000.0, 402_000.0)
    resolution = 100.0
    transform, snapped = construct_transform_shift_bounds(bounds, resolution)
    shape = grid_shape_from_bounds(snapped, resolution)

    out = reproject_alan_to_model_grid(
        source,
        target_crs="EPSG:27700",
        target_transform=transform,
        target_resolution=resolution,
        target_bounds=snapped,
        resampling=Resampling.bilinear,
    )

    assert out.rio.crs.to_string() == "EPSG:27700"
    assert out.shape == shape
    assert out.rio.transform() == transform
    assert out.attrs.get("long_name") == ALAN_BAND_NAME
    # Bright centre cells stay brighter than the dark corners after upsample.
    assert float(out.max()) > float(out.min())
    assert float(out.max()) > 1.0


def test_generate_alan_data_writes_aligned_layer(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "viirs_annual.tif"
    _write_coarse_viirs(source_path)
    boundary_path = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(
        geometry=[box(400_100, 400_100, 401_900, 401_900)],
        crs="EPSG:27700",
    ).to_file(boundary_path)

    # Keep the model grid tiny and deterministic for the test.
    def _fake_load(boundary_path, buffer_distance=7000):  # noqa: ARG001
        boundary = gpd.read_file(boundary_path)
        bounds = (400_000.0, 400_000.0, 402_000.0, 402_000.0)
        transform, snapped = construct_transform_shift_bounds(bounds, 100.0)
        return boundary, transform, snapped, {"crs": "EPSG:27700", "resolution": 100}

    monkeypatch.setattr(
        "sdm.commands.data_preparation.environmental.generate_alan_data.load_boundary_and_transform",
        _fake_load,
    )

    output_dir = tmp_path / "evs"
    written = generate_alan_data(
        output_dir=output_dir,
        boundary_path=boundary_path,
        alan_source_path=source_path,
        buffer_distance_m=0,
    )
    assert written == output_dir / ALAN_OUTPUT_NAME
    assert written.exists()

    result = rxr.open_rasterio(written, masked=True)
    if "band" in result.dims:
        result = result.squeeze("band", drop=True)
    assert result.rio.crs.to_string() == "EPSG:27700"
    assert result.shape == (20, 20)
    assert float(np.nanmax(result.values)) > float(np.nanmin(result.values))
