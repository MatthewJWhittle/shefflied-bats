"""Pin EV point annotation: elapid path matches legacy sjoin values (#51 item 1)."""

import time
import tracemalloc

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 — enables Dataset.rio
import xarray as xr
from shapely.geometry import Point

from sdm.commands.modelling.train_sdm_models import (
    annotate_points_with_evs,
    convert_evs_to_geodataframe,
    extract_grid_points,
)


def _synthetic_evs(nx: int = 30, ny: int = 30, n_vars: int = 5, res: float = 100.0):
    x = np.arange(nx, dtype=float) * res + res / 2
    y = (np.arange(ny, dtype=float)[::-1]) * res + res / 2
    data_vars = {}
    rng = np.random.default_rng(0)
    for i in range(n_vars):
        data_vars[f"v{i}"] = (("y", "x"), rng.random((ny, nx)))
    ds = xr.Dataset(data_vars, coords={"x": x, "y": y})
    ds = ds.rio.write_crs("EPSG:27700")
    return ds


def _random_points(evs: xr.Dataset, n: int = 50, seed: int = 1) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = evs.rio.bounds()
    # Stay inside cell centres bbox
    xs = rng.uniform(xmin + 1, xmax - 1, size=n)
    ys = rng.uniform(ymin + 1, ymax - 1, size=n)
    return gpd.GeoDataFrame(
        {"pid": np.arange(n)},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=evs.rio.crs,
    )


def _legacy_annotate(points, evs, tolerance=None):
    """Pre-swap path: full GeoDataFrame + sjoin_nearest."""
    import geopandas as gpd

    ev_gdf = convert_evs_to_geodataframe(evs)
    if tolerance is None:
        tolerance = abs(evs.rio.resolution()[0])
    annotated = gpd.sjoin_nearest(points, ev_gdf, how="inner", max_distance=tolerance)
    annotated = annotated.drop(columns=["index_right"])
    return annotated


def test_annotate_matches_legacy_sjoin_values_and_grid():
    evs = _synthetic_evs(20, 20, n_vars=4)
    points = _random_points(evs, n=40, seed=2)
    legacy = _legacy_annotate(points, evs)
    modern = annotate_points_with_evs(points, evs)

    merged = legacy.merge(modern, on="pid", suffixes=("_l", "_m"))
    assert len(merged) == len(legacy) == len(modern)
    for col in ["v0", "v1", "v2", "v3"]:
        np.testing.assert_allclose(
            merged[f"{col}_l"].to_numpy(),
            merged[f"{col}_m"].to_numpy(),
            rtol=0,
            atol=1e-9,
        )
    np.testing.assert_allclose(merged["grid_x_l"], merged["grid_x_m"], atol=1e-9)
    np.testing.assert_allclose(merged["grid_y_l"], merged["grid_y_m"], atol=1e-9)


def test_extract_grid_points_thin_has_coords_not_full_vars():
    evs = _synthetic_evs(8, 8, n_vars=3)
    grid = extract_grid_points(evs)
    assert {"grid_x", "grid_y", "geometry"} <= set(grid.columns)
    # Thin path: do not materialise every EV column on every cell
    assert "v0" not in grid.columns
    assert len(grid) == 8 * 8


def test_prepare_training_data_monkeypatch_signature(monkeypatch):
    """Background-all-records style patch still works with new annotate signature."""
    import sdm.commands.modelling.train_sdm_models as tsm

    class _Stop(Exception):
        pass

    def fake_background(occurrence_data, **kwargs):
        raise _Stop

    monkeypatch.setattr(
        tsm, "annotate_points_with_evs", lambda gdf, evs, **kw: gdf
    )
    monkeypatch.setattr(tsm, "generate_background_points_from_data", fake_background)

    class _Rio:
        @staticmethod
        def resolution():
            return (100.0, -100.0)

    class _Evs:
        data_vars = {"v0": None}
        rio = _Rio()

    bats = gpd.GeoDataFrame(
        {
            "latin_name": ["a", "b"],
            "activity_type": ["Roost", "Roost"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:27700",
    )
    with pytest.raises(_Stop):
        tsm.prepare_training_data(
            occurrence_gdf=bats,
            boundary=None,
            evs_to_model=_Evs(),
            n_background_points=10,
            background_method=None,
            background_value=0.0,
            sigma=1.0,
            transform_method=None,
            seed=1,
        )


@pytest.mark.slow
def test_annotate_avoids_full_grid_geodataframe_memory():
    """Materialising the full EV grid as a GeoDataFrame uses far more peak RAM."""
    # ~40k cells × 30 vars — large enough for a clear GeoDataFrame allocation gap
    evs = _synthetic_evs(200, 200, n_vars=30)
    points = _random_points(evs, n=200, seed=3)

    tracemalloc.start()
    t0 = time.perf_counter()
    legacy = _legacy_annotate(points, evs)
    legacy_s = time.perf_counter() - t0
    _, legacy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    modern = annotate_points_with_evs(points, evs)
    modern_s = time.perf_counter() - t0
    _, modern_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(legacy) == len(modern)
    # Full-grid GeoDataFrame peak must exceed point sampling (+ tempfile write).
    # On study-sized stacks the gap grows with cell count; here we require a
    # directional win and record absolute peaks for the PR proof section.
    assert modern_peak < legacy_peak, (
        f"expected memory win: modern={modern_peak/1e6:.1f}MB "
        f"legacy={legacy_peak/1e6:.1f}MB; "
        f"modern_s={modern_s:.3f}s legacy_s={legacy_s:.3f}s"
    )
    # Publish numbers into the assertion message on failure; also guard ratio
    # so a regression that nearly erases the win is caught.
    ratio = modern_peak / legacy_peak
    assert ratio < 0.85, (
        f"memory win too small: modern/legacy={ratio:.2f} "
        f"({modern_peak/1e6:.1f}/{legacy_peak/1e6:.1f} MB)"
    )
