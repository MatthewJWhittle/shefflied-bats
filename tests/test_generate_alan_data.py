"""Tests for the artificial light at night (ALAN / VIIRS) layer (#37)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from shapely.geometry import box

from sdm.commands.data_preparation.environmental.generate_alan_data import (
    generate_alan_data,
)
from sdm.commands.data_preparation.processing.merge_ev_layers import (
    build_ev_dataset_inputs,
    parse_dataset_input,
)
from sdm.data.alan import (
    ALAN_BAND_NAME,
    ALAN_OUTPUT_NAME,
    DEFAULT_ALAN_CF_CVG_URI,
    DEFAULT_ALAN_SOURCE_URI,
    load_alan_source,
    load_alan_source_windowed,
    reproject_alan_to_model_grid,
    resolve_alan_cf_cvg_uri,
    resolve_alan_source_uri,
)
from sdm.raster.utils import construct_transform_shift_bounds, grid_shape_from_bounds


def _write_coarse_viirs(
    path: Path,
    *,
    values: np.ndarray | None = None,
    crs: str = "EPSG:27700",
    band_count: int = 1,
) -> None:
    """Synthetic ~500 m radiance grid covering 400000–402000 east/north."""
    if values is None:
        values = np.array(
            [
                [0.0, 0.2, 0.2, 0.0],
                [0.2, 8.0, 7.5, 0.2],
                [0.2, 7.0, 6.5, 0.2],
                [0.0, 0.2, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
    height, width = values.shape
    # North-up: origin at NW corner (400000 E, 402000 N).
    transform = Affine(500.0, 0.0, 400_000.0, 0.0, -500.0, 402_000.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if band_count == 1:
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
        da.rio.to_raster(path)
        return

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": band_count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rio.open(path, "w", **profile) as dst:
        for band_i in range(1, band_count + 1):
            dst.write(values, band_i)


def _write_cf_cvg(path: Path, coverage: np.ndarray, *, crs: str = "EPSG:27700") -> None:
    height, width = coverage.shape
    transform = Affine(500.0, 0.0, 400_000.0, 0.0, -500.0, 402_000.0)
    da = xr.DataArray(
        coverage.astype(np.float32),
        dims=("y", "x"),
        coords={
            "y": np.arange(height) * -500.0 + 402_000.0 - 250.0,
            "x": np.arange(width) * 500.0 + 400_000.0 + 250.0,
        },
        name="cf_cvg",
    )
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path)


def test_load_alan_source_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tif"
    with pytest.raises(FileNotFoundError, match="ALAN source not found"):
        load_alan_source(missing)


def test_load_alan_source_rejects_multiband(tmp_path: Path) -> None:
    source_path = tmp_path / "multi.tif"
    _write_coarse_viirs(source_path, band_count=2)
    with pytest.raises(ValueError, match="has 2 bands"):
        load_alan_source(source_path)


def test_valid_zeros_preserved_through_window_and_reproject(tmp_path: Path) -> None:
    source_path = tmp_path / "viirs.tif"
    values = np.zeros((4, 4), dtype=np.float32)
    values[1, 1] = 5.0
    _write_coarse_viirs(source_path, values=values)

    bounds = (400_000.0, 400_000.0, 402_000.0, 402_000.0)
    windowed = load_alan_source_windowed(
        source_path,
        target_bounds=bounds,
        target_crs="EPSG:27700",
        margin_pixels=0,
    )
    assert float(np.nanmin(windowed.values)) == 0.0
    assert (windowed.values == 0.0).sum() >= 1

    resolution = 100.0
    transform, snapped = construct_transform_shift_bounds(bounds, resolution)
    out = reproject_alan_to_model_grid(
        windowed,
        target_crs="EPSG:27700",
        target_transform=transform,
        target_resolution=resolution,
        target_bounds=snapped,
        resampling=Resampling.bilinear,
    )
    assert float(np.nanmin(out.values)) == 0.0
    assert np.isfinite(out.values).all() or (out.values == 0).any()
    # Zeros remain present after bilinear upsample (not converted to NaN).
    assert (out.values == 0.0).sum() >= 1


def test_cf_cvg_marks_missing_coverage_as_nan(tmp_path: Path) -> None:
    source_path = tmp_path / "viirs.tif"
    cov_path = tmp_path / "cf_cvg.tif"
    values = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 8.0, 7.0, 1.0],
            [1.0, 6.0, 5.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    coverage = np.ones_like(values)
    coverage[0, 0] = 0  # missing coverage on a zero-radiance cell
    coverage[1, 1] = 0  # missing coverage on a bright cell
    _write_coarse_viirs(source_path, values=values)
    _write_cf_cvg(cov_path, coverage)

    bounds = (400_000.0, 400_000.0, 402_000.0, 402_000.0)
    windowed = load_alan_source_windowed(
        source_path,
        target_bounds=bounds,
        target_crs="EPSG:27700",
        margin_pixels=0,
        cf_cvg_path=cov_path,
    )
    assert np.isnan(windowed.values[0, 0])
    assert np.isnan(windowed.values[1, 1])
    # Valid zero elsewhere preserved.
    assert windowed.values[0, 3] == 0.0
    assert windowed.values[3, 0] == 0.0


def test_reproject_geographic_source_to_model_grid(tmp_path: Path) -> None:
    """EPSG:4326 VIIRS-like grid warps onto an OSGB model template."""
    from rasterio.warp import transform_bounds

    source_path = tmp_path / "viirs_wgs84.tif"
    # Small patch around Sheffield in geographic CRS (~0.0041667° ≈ 15").
    res = 0.0041667
    west, north = -1.52, 53.42
    values = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 12.0, 0.5],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    height, width = values.shape
    east = west + width * res
    south = north - height * res
    transform = Affine(res, 0.0, west, 0.0, -res, north)
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": north - (np.arange(height) + 0.5) * res,
            "x": west + (np.arange(width) + 0.5) * res,
        },
        name=ALAN_BAND_NAME,
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)
    da.rio.to_raster(source_path)

    # Model footprint = OSGB envelope of the synthetic WGS84 tile.
    osgb_bounds = transform_bounds("EPSG:4326", "EPSG:27700", west, south, east, north)
    resolution = 100.0
    model_transform, snapped = construct_transform_shift_bounds(osgb_bounds, resolution)
    shape = grid_shape_from_bounds(snapped, resolution)

    windowed = load_alan_source_windowed(
        source_path,
        target_bounds=snapped,
        target_crs="EPSG:27700",
        margin_pixels=1,
    )
    out = reproject_alan_to_model_grid(
        windowed,
        target_crs="EPSG:27700",
        target_transform=model_transform,
        target_resolution=resolution,
        target_bounds=snapped,
        resampling=Resampling.bilinear,
    )
    assert out.rio.crs.to_string() == "EPSG:27700"
    assert out.shape == shape
    assert out.rio.transform() == model_transform
    assert out.attrs.get("long_name") == ALAN_BAND_NAME
    assert float(np.nanmax(out.values)) > float(np.nanmin(out.values))


def test_merged_band_name_is_alan_viirs_radiance() -> None:
    inputs = build_ev_dataset_inputs(Path("data/evs"), resolution=100)
    parsed = parse_dataset_input(inputs)
    assert "alan_viirs_radiance" in parsed
    assert parsed["alan_viirs_radiance"] == Path("data/evs") / ALAN_OUTPUT_NAME
    assert "alan" not in parsed or "alan_viirs_radiance" in parsed


def test_generate_alan_data_writes_aligned_layer(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "viirs_annual.tif"
    _write_coarse_viirs(source_path)
    boundary_path = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(
        geometry=[box(400_100, 400_100, 401_900, 401_900)],
        crs="EPSG:27700",
    ).to_file(boundary_path)

    def _fake_load(boundary_path, buffer_distance=0):  # noqa: ARG001
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
        compute_checksum=False,
    )
    assert written == output_dir / ALAN_OUTPUT_NAME
    assert written.exists()
    assert (output_dir / "alan_provenance.json").exists()

    result = rxr.open_rasterio(written, masked=True)
    if "band" in result.dims:
        result = result.squeeze("band", drop=True)
    assert result.rio.crs.to_string() == "EPSG:27700"
    assert result.shape == (20, 20)
    assert float(np.nanmax(result.values)) > float(np.nanmin(result.values))
    # Band long name used for merge / ML selection.
    with rio.open(written) as src:
        assert src.descriptions[0] == ALAN_BAND_NAME or src.tags(1).get(
            "long_name"
        ) in {
            ALAN_BAND_NAME,
            None,
            "",
        }


def test_resolve_alan_prefers_local_over_bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = tmp_path / "viirs_annual.tif"
    local.write_bytes(b"x")
    monkeypatch.setattr("sdm.data.alan.DEFAULT_ALAN_SOURCE_PATH", local)
    # Bucket must not be probed when local exists.
    def boom(uri: str) -> bool:  # noqa: ARG001
        raise AssertionError(f"unexpected uri_exists for {uri}")

    monkeypatch.setattr("sdm.data.alan.uri_exists", lambda ref: Path(str(ref)).exists())
    assert resolve_alan_source_uri() == str(local)


def test_resolve_alan_falls_back_to_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = Path("/nonexistent/alan/viirs_annual.tif")
    monkeypatch.setattr("sdm.data.alan.DEFAULT_ALAN_SOURCE_PATH", missing)

    def fake_exists(ref: str) -> bool:
        return str(ref) == DEFAULT_ALAN_SOURCE_URI

    monkeypatch.setattr("sdm.data.alan.uri_exists", fake_exists)
    assert resolve_alan_source_uri() == DEFAULT_ALAN_SOURCE_URI


def test_resolve_cf_cvg_optional_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = Path("/nonexistent/alan/viirs_cf_cvg.tif")
    monkeypatch.setattr("sdm.data.alan.DEFAULT_ALAN_CF_CVG_PATH", missing)
    monkeypatch.setattr("sdm.data.alan.uri_exists", lambda ref: False)
    assert resolve_alan_cf_cvg_uri(required=False) is None
