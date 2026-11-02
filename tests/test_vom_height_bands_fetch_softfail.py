"""Fused-path soft-fail / concurrency / cache guards for VOM height bands (#122)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from dask.delayed import delayed
from shapely.geometry import Polygon
from tilearray import array as tilearray_array
from tilearray.errors import NetworkError
from tilearray.fetch import FetchProgress

from sdm.commands.data_preparation.environmental.generate_vom_height_bands import (
    DEFAULT_MAX_FAILED_CHUNK_FRACTION,
    EA_FETCH_CONCURRENCY_MAX,
    EA_FETCH_CONCURRENCY_MIN_PARALLEL,
    VomHeightBandsChunkFailureError,
    _map_blocks_summaries_fused,
    generate_vom_height_bands,
    resolve_vom_fetch_concurrency,
    tilearray_soft_fail_tiles,
)
from sdm.raster.vom_height_bands import CELL_LAYER_STEMS


def test_resolve_vom_fetch_concurrency_never_collapses_parallel_to_one() -> None:
    """Guard: parallel downloads must not be serialized via max_concurrent=1."""
    assert resolve_vom_fetch_concurrency(1) == 1  # explicit serial OK
    assert resolve_vom_fetch_concurrency(8) == 8
    assert resolve_vom_fetch_concurrency(4) == EA_FETCH_CONCURRENCY_MIN_PARALLEL
    assert resolve_vom_fetch_concurrency(20) == EA_FETCH_CONCURRENCY_MAX
    assert resolve_vom_fetch_concurrency(8) > 1


@dataclass
class _FakeTileRequest:
    height: int
    width: int


def test_soft_fail_real_fused_path_injected_failures() -> None:
    """Plan-time soft-fail loader: 10% injected fails → complete + correct count."""
    n_tiles = 10
    tile_px = 100
    progress = FetchProgress(total=n_tiles)
    calls = {"n": 0}

    def flaky_load(
        request,
        cache_dir,
        decoder,
        dtype,
        fetch_policy=None,
        progress=None,
    ):
        calls["n"] += 1
        h = int(request.height or tile_px)
        w = int(request.width or tile_px)
        if calls["n"] % 10 == 0:
            if progress is not None:
                progress.abort("injected fetch failure")
            raise NetworkError("injected fetch failure")
        return np.full((h, w), 12.0, dtype=dtype)

    real_load = tilearray_array._load_tile_array
    tilearray_array._load_tile_array = flaky_load  # type: ignore[assignment]
    try:
        with tilearray_soft_fail_tiles() as soft_failed:
            load_fn = tilearray_array._load_tile_array
            row_blocks = []
            for _ in range(n_tiles):
                req = _FakeTileRequest(height=tile_px, width=tile_px)
                delayed_tile = delayed(load_fn)(
                    req, None, None, np.dtype("float32"), None, progress
                )
                row_blocks.append(
                    [
                        da.from_delayed(
                            delayed_tile,
                            shape=(tile_px, tile_px),
                            dtype=np.float32,
                        )
                    ]
                )
            mosaic = da.block(row_blocks)
            y = np.arange(float(n_tiles * tile_px), 0.0, -1.0) - 0.5
            x = np.arange(0.0, float(tile_px), 1.0) + 0.5
            lazy = xr.DataArray(
                mosaic, dims=("y", "x"), coords={"y": y, "x": x}, name="vom"
            )
            out, n_chunks, _stages = _map_blocks_summaries_fused(
                lazy,
                fine_resolution_m=1.0,
                summary_resolution_m=100,
            )
            n_failed = len(soft_failed)
    finally:
        tilearray_array._load_tile_array = real_load  # type: ignore[assignment]

    assert n_chunks == n_tiles
    assert n_failed == 1
    assert n_failed / n_chunks < DEFAULT_MAX_FAILED_CHUNK_FRACTION
    assert not progress.is_aborted()
    band0 = np.asarray(out.isel(band=0).values)
    assert np.any(np.isfinite(band0))
    assert out.sizes["band"] == len(CELL_LAYER_STEMS)


def test_soft_fail_abort_threshold_via_generate(tmp_path: Path) -> None:
    """generate_vom_height_bands raises when soft-failed fraction exceeds 15%."""
    boundary = gpd.GeoDataFrame(
        {
            "id": [1],
            "geometry": [
                Polygon(
                    [
                        (420000, 380000),
                        (421000, 380000),
                        (421000, 381000),
                        (420000, 381000),
                        (420000, 380000),
                    ]
                )
            ],
        },
        crs="EPSG:27700",
    )
    boundary_path = tmp_path / "boundary.geojson"
    boundary.to_file(boundary_path)

    # Two delayed tiles; fail both so fail_frac = 1.0 > 0.15.
    n_tiles = 2
    tile_px = 100
    calls = {"n": 0}

    def flaky_load(
        request,
        cache_dir,
        decoder,
        dtype,
        fetch_policy=None,
        progress=None,
    ):
        calls["n"] += 1
        h = int(request.height or tile_px)
        w = int(request.width or tile_px)
        if progress is not None:
            progress.abort("injected")
        raise NetworkError("injected fetch failure")

    def fake_fetch_wcs_dataarray(**_kwargs):
        # Build mosaic while soft-fail is already active (generate installs it
        # before calling fetch). Bind whatever loader is currently installed.
        load_fn = tilearray_array._load_tile_array
        progress = FetchProgress(total=n_tiles)
        row_blocks = []
        for _ in range(n_tiles):
            req = _FakeTileRequest(height=tile_px, width=tile_px)
            delayed_tile = delayed(load_fn)(
                req, None, None, np.dtype("float32"), None, progress
            )
            row_blocks.append(
                [
                    da.from_delayed(
                        delayed_tile, shape=(tile_px, tile_px), dtype=np.float32
                    )
                ]
            )
        mosaic = da.block(row_blocks)
        y = np.arange(float(n_tiles * tile_px), 0.0, -1.0) - 0.5
        x = np.arange(0.0, float(tile_px), 1.0) + 0.5
        return xr.DataArray(
            mosaic, dims=("y", "x"), coords={"y": y, "x": x}, name="vom"
        )

    real_load = tilearray_array._load_tile_array
    # Soft-fail wraps real_load; inject flaky as the "original" underneath by
    # patching before generate enters soft-fail (soft-fail captures current).
    tilearray_array._load_tile_array = flaky_load  # type: ignore[assignment]
    try:
        with patch(
            "sdm.commands.data_preparation.environmental.generate_vom_height_bands.fetch_wcs_dataarray",
            fake_fetch_wcs_dataarray,
        ):
            with pytest.raises(VomHeightBandsChunkFailureError):
                generate_vom_height_bands(
                    output_dir=tmp_path / "out",
                    boundary_path=boundary_path,
                    wcs_download_resolution_m=1.0,
                    summary_target_resolution_m=100,
                    max_concurrent_downloads=1,
                    max_failed_chunk_fraction=DEFAULT_MAX_FAILED_CHUNK_FRACTION,
                    verbose=False,
                )
    finally:
        tilearray_array._load_tile_array = real_load  # type: ignore[assignment]


def test_generate_default_disables_persistent_wcs_cache(tmp_path: Path) -> None:
    """Full-area default: no wcs-cache under output (disk-fill guard)."""
    boundary = gpd.GeoDataFrame(
        {
            "id": [1],
            "geometry": [
                Polygon(
                    [
                        (420000, 380000),
                        (421000, 380000),
                        (421000, 381000),
                        (420000, 381000),
                        (420000, 380000),
                    ]
                )
            ],
        },
        crs="EPSG:27700",
    )
    boundary_path = tmp_path / "boundary.geojson"
    boundary.to_file(boundary_path)
    output_dir = tmp_path / "out"

    yy = np.linspace(381000.5, 380800.5, 200)
    xx = np.linspace(420000.5, 420200.5, 200)
    vom = xr.DataArray(
        np.full((200, 200), 8.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": yy, "x": xx},
        name="vom",
    )
    seen: dict[str, object] = {}

    def fake_fetch(**kwargs):
        seen["cache_dir"] = kwargs.get("cache_dir")
        return vom

    with patch(
        "sdm.commands.data_preparation.environmental.generate_vom_height_bands.fetch_wcs_dataarray",
        fake_fetch,
    ):
        generate_vom_height_bands(
            output_dir=output_dir,
            boundary_path=boundary_path,
            wcs_download_resolution_m=1.0,
            summary_target_resolution_m=100,
            max_concurrent_downloads=1,
            verbose=False,
        )

    assert seen["cache_dir"] is None
    assert not (output_dir / "wcs-cache").exists()
    notes = json.loads(
        (output_dir / "vom_height_bands_notes.json").read_text(encoding="utf-8")
    )
    assert notes["wcs_cache"] == "off"
    assert notes["n_failed"] == 0
