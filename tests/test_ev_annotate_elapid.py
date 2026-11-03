"""Behaviour pins for EV point annotation via elapid (#47 / #51 item 1).

Samples known band values at known cell centres. Does not re-run the old
full-grid ``sjoin_nearest`` path or assert memory/speed wins.
"""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 — enables Dataset.rio
import xarray as xr
from shapely.geometry import Point

from sdm.commands.modelling.train_sdm_models import (
    annotate_points_with_evs,
    extract_grid_points,
)


def _known_evs() -> xr.Dataset:
    """3×3 stack with deterministic band values and 100 m cell centres."""
    res = 100.0
    x = np.array([50.0, 150.0, 250.0])
    y = np.array([250.0, 150.0, 50.0])  # north-up
    v0 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    v1 = v0 * 10.0
    ds = xr.Dataset(
        {"v0": (("y", "x"), v0), "v1": (("y", "x"), v1)},
        coords={"x": x, "y": y},
    )
    return ds.rio.write_crs("EPSG:27700")


def test_annotate_samples_known_values_at_cell_centres():
    evs = _known_evs()
    # Exact centres of (row0,col0), (row1,col1), (row2,col2)
    points = gpd.GeoDataFrame(
        {"pid": [0, 1, 2]},
        geometry=[
            Point(50.0, 250.0),
            Point(150.0, 150.0),
            Point(250.0, 50.0),
        ],
        crs="EPSG:27700",
    )
    annotated = annotate_points_with_evs(points, evs).sort_values("pid")

    np.testing.assert_allclose(annotated["v0"].to_numpy(), [1.0, 5.0, 9.0], atol=1e-9)
    np.testing.assert_allclose(
        annotated["v1"].to_numpy(), [10.0, 50.0, 90.0], atol=1e-9
    )
    np.testing.assert_allclose(
        annotated["grid_x"].to_numpy(), [50.0, 150.0, 250.0], atol=1e-9
    )
    np.testing.assert_allclose(
        annotated["grid_y"].to_numpy(), [250.0, 150.0, 50.0], atol=1e-9
    )


def test_annotate_nearest_cell_for_off_centre_point():
    """Point inside a cell (not on centre) still samples that cell's value."""
    evs = _known_evs()
    # Inside centre cell (v0=5): centre (150,150), cell spans ~100–200
    points = gpd.GeoDataFrame(
        {"pid": [0]},
        geometry=[Point(160.0, 140.0)],
        crs="EPSG:27700",
    )
    annotated = annotate_points_with_evs(points, evs)
    assert float(annotated["v0"].iloc[0]) == pytest.approx(5.0)
    assert float(annotated["grid_x"].iloc[0]) == pytest.approx(150.0)
    assert float(annotated["grid_y"].iloc[0]) == pytest.approx(150.0)


def test_extract_grid_points_thin_has_coords_not_full_vars():
    evs = _known_evs()
    grid = extract_grid_points(evs)
    assert {"grid_x", "grid_y", "geometry"} <= set(grid.columns)
    assert "v0" not in grid.columns
    assert len(grid) == 9
    # Corner cell centres present
    centres = set(zip(grid["grid_x"].tolist(), grid["grid_y"].tolist()))
    assert (50.0, 250.0) in centres
    assert (250.0, 50.0) in centres


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
