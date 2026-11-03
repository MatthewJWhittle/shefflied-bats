"""Background points for filtered runs must come from every bat record.

Synthetic data only.
"""
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

import sdm.commands.modelling.train_sdm_models as tsm


def _synthetic_bats() -> gpd.GeoDataFrame:
    rows = []
    for i in range(6):
        rows.append(("Species a", "Roost", 1000.0 + i * 10))
    for i in range(9):
        rows.append(("Species b", "In flight", 5000.0 + i * 10))
    return gpd.GeoDataFrame(
        {
            "latin_name": [r[0] for r in rows],
            "activity_type": [r[1] for r in rows],
            "accuracy": [10] * len(rows),
        },
        geometry=[Point(r[2], r[2]) for r in rows],
        crs="EPSG:27700",
    )


class _Stop(Exception):
    pass


def test_prepare_training_data_uses_background_occurrence(monkeypatch):
    captured = {}

    def fake_background(occurrence_data, **kwargs):
        captured["n"] = len(occurrence_data)
        raise _Stop

    monkeypatch.setattr(tsm, "convert_evs_to_geodataframe", lambda evs: None)
    monkeypatch.setattr(tsm, "annotate_points_with_evs", lambda gdf, ev, tolerance: gdf)
    monkeypatch.setattr(tsm, "generate_background_points_from_data", fake_background)

    class _Rio:
        @staticmethod
        def resolution():
            return (100.0, -100.0)

    class _Evs:
        data_vars = {"x": None}
        rio = _Rio()

    bats = _synthetic_bats()
    subset = bats[bats.latin_name == "Species a"]
    with pytest.raises(_Stop):
        tsm.prepare_training_data(
            occurrence_gdf=subset, boundary=None, evs_to_model=_Evs(),
            n_background_points=10, background_method=None, background_value=0.0,
            sigma=1.0, transform_method=None, seed=1,
            background_occurrence_gdf=bats,
        )
    assert captured["n"] == len(bats)

    with pytest.raises(_Stop):
        tsm.prepare_training_data(
            occurrence_gdf=subset, boundary=None, evs_to_model=_Evs(),
            n_background_points=10, background_method=None, background_value=0.0,
            sigma=1.0, transform_method=None, seed=1,
        )
    assert captured["n"] == len(subset)


def test_species_filtered_run_passes_all_records_for_background(monkeypatch, tmp_path):
    bats = _synthetic_bats()
    captured = {}

    @contextmanager
    def fake_materialized(uri):
        yield str(uri)

    def fake_prepare(**kwargs):
        captured["presence"] = kwargs["occurrence_gdf"]
        captured["background"] = kwargs["background_occurrence_gdf"]
        raise _Stop

    monkeypatch.setattr(tsm, "local_materialized_path", fake_materialized)
    monkeypatch.setattr(tsm, "load_bat_data", lambda path: bats)
    monkeypatch.setattr(
        tsm, "load_boundary",
        lambda **kw: gpd.GeoDataFrame(geometry=[box(0, 0, 10000, 10000)], crs="EPSG:27700"),
    )
    monkeypatch.setattr(tsm, "simplify_boundary", lambda b, tolerance=100: b)
    monkeypatch.setattr(tsm, "load_environmental_variables", lambda uri: (None, None))
    monkeypatch.setattr(tsm, "resolve_mlflow_config", lambda cfg: cfg.mlflow)
    monkeypatch.setattr(tsm, "configure_mlflow", lambda cfg: False)
    monkeypatch.setattr(tsm, "prepare_training_data", fake_prepare)

    with pytest.raises(_Stop):
        tsm.train_sdm_models(
            bats_file=Path("bats.geojson"), ev_file="evs.tif", output_dir=tmp_path,
            species=["Species a"], activity_types=["Roost"],
        )
    assert set(captured["presence"].latin_name) == {"Species a"}
    assert len(captured["background"]) == len(bats)
    assert set(captured["background"].latin_name) == {"Species a", "Species b"}
