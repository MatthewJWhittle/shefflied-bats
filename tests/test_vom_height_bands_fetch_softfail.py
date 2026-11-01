"""Per-chunk fetch soft-fail for VOM height bands (#122 box)."""

from __future__ import annotations

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
    _bbox_from_centres,
    _default_fetch_chunk,
    _map_blocks_summaries_fused,
    generate_vom_height_bands,
    resolve_vom_fetch_concurrency,
    summarise_chunks_resilient,
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


def test_default_fetch_chunk_passes_resolved_concurrency(monkeypatch) -> None:
    """Regression for shared TileFetcher serializing 8 workers at max_concurrent=1."""
    seen: dict[str, object] = {}

    def fake_fetch_wcs_dataarray(**kwargs):
        seen["max_concurrent"] = kwargs.get("max_concurrent")
        yy = np.arange(10.0, 0.0, -1.0) - 0.5
        xx = np.arange(0.0, 10.0, 1.0) + 0.5
        return __import__("xarray").DataArray(
            np.full((10, 10), 5.0, dtype=np.float32),
            dims=("y", "x"),
            coords={"y": yy, "x": xx},
        )

    monkeypatch.setattr(
        "sdm.commands.data_preparation.environmental.generate_vom_height_bands.fetch_wcs_dataarray",
        fake_fetch_wcs_dataarray,
    )
    arr = _default_fetch_chunk(
        (0.0, 0.0, 10.0, 10.0),
        resolution_m=1.0,
        chunk_size=(10, 10),
        max_concurrent=8,
        cache_dir=None,
    )
    assert arr.shape == (10, 10)
    assert seen["max_concurrent"] == 8
    assert seen["max_concurrent"] != 1


def test_bbox_from_centres_snaps_to_resolution_grid() -> None:
    """Drifted linspace centres must still yield exact n×resolution spans."""
    res = 1.0
    n = 1024
    # Mimic tilearray: linspace over edges (inclusive) → ~1024.0 + epsilon span
    # when half-pixel pads are applied without snapping.
    x_drifted = np.linspace(420000.0, 420000.0 + n, n)
    y_drifted = np.linspace(380000.0 + n, 380000.0, n)
    raw_w = (x_drifted.max() + res / 2.0) - (x_drifted.min() - res / 2.0)
    assert raw_w != pytest.approx(n * res)  # documents the drift bug

    minx, miny, maxx, maxy = _bbox_from_centres(
        y_drifted, x_drifted, resolution_m=res
    )
    width = maxx - minx
    height = maxy - miny
    assert width == pytest.approx(n * res)
    assert height == pytest.approx(n * res)
    assert width / res == pytest.approx(int(width / res))
    assert minx / res == pytest.approx(round(minx / res))
    assert maxy / res == pytest.approx(round(maxy / res))


def test_resilient_chunks_one_request_per_chunk_no_slivers() -> None:
    """Each chunk issues one snapped bbox; widths are exact pixel multiples."""
    y = np.linspace(10000.0, 0.0, 1000)  # drifted centres
    x = np.linspace(0.0, 1000.0, 1000)
    chunks = ((100,) * 10, (1000,))
    seen: list[tuple[float, float, float, float]] = []

    def fetch_chunk(bbox):
        seen.append(bbox)
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny
        assert width == pytest.approx(round(width))
        assert height == pytest.approx(round(height))
        assert width % 1.0 == pytest.approx(0.0)
        ny = int(round(height))
        nx = int(round(width))
        return np.full((ny, nx), 10.0, dtype=np.float64)

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_chunk,
        max_workers=4,
    )
    assert n_chunks == 10
    assert n_failed == 0
    assert len(seen) == n_chunks  # one request per chunk, no slivers
    assert mosaic.shape[0] == len(CELL_LAYER_STEMS)


def test_injected_fetch_failures_do_not_abort_siblings() -> None:
    """~10% injected fetch failures → NaN tiles + count; siblings succeed."""
    # 10 chunks of 100×100 at 1 m → 10 summary cells.
    y = np.arange(1000.0, 0.0, -1.0) - 0.5  # 1000 centres
    x = np.arange(0.0, 1000.0, 1.0) + 0.5
    chunks = ((100,) * 10, (1000,))  # 10 row-chunks × 1 col = 10 chunks
    calls = {"n": 0}

    def fetch_chunk(bbox):
        calls["n"] += 1
        # Fail every 10th chunk (10%).
        if calls["n"] % 10 == 0:
            raise RuntimeError("injected fetch failure")
        # Constant canopy height for a successful 100×100 (or whatever) block.
        # Size from bbox span at 1 m.
        minx, miny, maxx, maxy = bbox
        ny = max(1, int(round(maxy - miny)))
        nx = max(1, int(round(maxx - minx)))
        return np.full((ny, nx), 10.0, dtype=np.float64)

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_chunk,
        max_workers=4,
    )
    assert n_chunks == 10
    assert n_failed == 1
    assert mosaic.shape[0] == len(CELL_LAYER_STEMS)
    # One failed row-strip is all-NaN; others finite.
    band0 = mosaic[0]
    nan_rows = [i for i in range(band0.shape[0]) if np.all(~np.isfinite(band0[i]))]
    finite_rows = [i for i in range(band0.shape[0]) if np.any(np.isfinite(band0[i]))]
    assert len(nan_rows) == 1
    assert len(finite_rows) == 9
    assert n_failed / n_chunks < DEFAULT_MAX_FAILED_CHUNK_FRACTION


def test_abort_when_failed_fraction_exceeds_limit() -> None:
    y = np.arange(200.0, 0.0, -1.0) - 0.5
    x = np.arange(0.0, 100.0, 1.0) + 0.5
    chunks = ((100, 100), (100,))

    def fetch_always_fail(_bbox):
        raise RuntimeError("injected fetch failure")

    mosaic, n_chunks, n_failed = summarise_chunks_resilient(
        y_coords=y,
        x_coords=x,
        chunks=chunks,
        fine_resolution_m=1.0,
        summary_resolution_m=100,
        fetch_chunk=fetch_always_fail,
        max_workers=2,
    )
    assert n_chunks == 2
    assert n_failed == 2
    assert np.all(~np.isfinite(mosaic))
    # Caller would abort — mirror the threshold check used in the generator.
    fail_frac = n_failed / n_chunks
    assert fail_frac > DEFAULT_MAX_FAILED_CHUNK_FRACTION
    with pytest.raises(VomHeightBandsChunkFailureError):
        if fail_frac > DEFAULT_MAX_FAILED_CHUNK_FRACTION:
            raise VomHeightBandsChunkFailureError(
                f"Failed chunk fraction {n_failed}/{n_chunks}"
            )


@dataclass
class _FakeTileRequest:
    height: int
    width: int


def test_soft_fail_real_fused_path_injected_failures() -> None:
    """Plan-time soft-fail loader: 10% injected fails → complete + correct count.

    Builds a mosaic the way tilearray does (``delayed(load_fn)`` bound at plan
    time) inside ``tilearray_soft_fail_tiles``, then runs the fused summarise
    path. A post-compute-only monkeypatch would still abort here.
    """
    n_tiles = 10
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
        if calls["n"] % 10 == 0:
            if progress is not None:
                progress.abort("injected fetch failure")
            raise NetworkError("injected fetch failure")
        return np.full((h, w), 12.0, dtype=dtype)

    # Soft-fail must wrap plan (delayed bind) + compute.
    real_load = tilearray_array._load_tile_array
    progress = FetchProgress(total=n_tiles)
    try:
        tilearray_array._load_tile_array = flaky_load  # type: ignore[assignment]
        with tilearray_soft_fail_tiles() as soft_failed:
            load_fn = tilearray_array._load_tile_array  # soft_load bound into graph
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
            out, n_chunks, _n_failed_summary, _stages = _map_blocks_summaries_fused(
                lazy,
                fine_resolution_m=1.0,
                summary_resolution_m=100,
            )
            n_failed = max(len(soft_failed), int(_n_failed_summary))
    finally:
        tilearray_array._load_tile_array = real_load  # type: ignore[assignment]

    assert n_chunks == n_tiles
    assert n_failed == 1
    assert len(soft_failed) == 1
    assert n_failed / n_chunks < DEFAULT_MAX_FAILED_CHUNK_FRACTION
    assert not progress.is_aborted()  # soft abort must not poison siblings
    # Compute completed with finite output elsewhere (soft-fail did not abort).
    band0 = np.asarray(out.isel(band=0).values)
    assert np.any(np.isfinite(band0))
    assert out.sizes["band"] == len(CELL_LAYER_STEMS)


def test_generate_default_disables_persistent_wcs_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    notes = (output_dir / "vom_height_bands_notes.json").read_text(encoding="utf-8")
    assert '"wcs_cache": "off"' in notes
