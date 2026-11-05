"""Notebook MaxEnt experiment helpers share the ``sdm train`` path.

Synthetic data only. Pins:

1. ``train_maxent_experiment`` forwards the same kwargs as CLI → ``train_sdm_models``.
2. ``prepare_maxent_experiment`` + ``train_from_cache`` (no overrides) matches
   ``train_sdm_models`` mean CV AUC and selected variables with the same seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 — Dataset.rio
import xarray as xr
import yaml
from shapely.geometry import Point, box

from sdm.commands.modelling import maxent_experiment as mex
from sdm.commands.modelling import train_sdm_models as tsm
from sdm.models.maxent.folds import FoldParams, FoldMethod
from sdm.models.maxent.maxent_model import DefaultMaxentConfig
from sdm.occurrence.background_settings import (
    BackgroundSurface,
    ResolvedBackgroundSettings,
)
from sdm.occurrence.sampling import BackgroundMethod, TransformMethod


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _synthetic_ev_dataset(nx: int = 20, ny: int = 20, res: float = 100.0) -> xr.Dataset:
    """Small north-up EV stack with deterministic bands."""
    x = (np.arange(nx) + 0.5) * res
    y = (np.arange(ny)[::-1] + 0.5) * res
    xx, yy = np.meshgrid(x, y)
    rng = np.random.default_rng(0)
    bands = {
        "ev_a": (yy / y.max()) + 0.1 * rng.random(yy.shape),
        "ev_b": (xx / x.max()) + 0.1 * rng.random(xx.shape),
        "ev_c": ((xx + yy) / (x.max() + y.max())) + 0.05 * rng.random(xx.shape),
    }
    ds = xr.Dataset(
        {name: (("y", "x"), values.astype("float32")) for name, values in bands.items()},
        coords={"x": x, "y": y},
    )
    ds = ds.rio.write_crs("EPSG:27700")
    ds.rio.write_transform(inplace=True)
    for name in bands:
        ds[name].attrs["long_name"] = name
    return ds


def _write_ev_geotiff(path: Path, ds: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in ds.data_vars:
        ds[name].attrs["long_name"] = name
        ds[name].rio.write_nodata(np.nan, inplace=True)
    ds.rio.to_raster(path)


def _synthetic_bats(n_presence: int = 40, n_other: int = 20) -> gpd.GeoDataFrame:
    """Presence cluster for Demo species Roost plus other bats for background."""
    rng = np.random.default_rng(1)
    rows = []
    # Spread presence across the 2 km × 2 km grid so blocks / thinning keep rows.
    for i in range(n_presence):
        rows.append(
            {
                "latin_name": "Demo species",
                "activity_type": "Roost",
                "accuracy": 10,
                "geometry": Point(
                    200.0 + (i % 10) * 150.0,
                    200.0 + (i // 10) * 150.0,
                ),
            }
        )
    for i in range(n_other):
        rows.append(
            {
                "latin_name": "Other species",
                "activity_type": "In flight",
                "accuracy": 10,
                "geometry": Point(
                    300.0 + rng.random() * 1400.0,
                    300.0 + rng.random() * 1400.0,
                ),
            }
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:27700")


def _project_fixture(tmp_path: Path) -> dict[str, Path]:
    """Minimal project tree for one Demo species Roost model."""
    root = tmp_path / "project"
    processed = root / "data" / "processed"
    evs_dir = root / "data" / "evs"
    config_dir = root / "data" / "sdm_config" / "Demo_species_Roost"
    models_dir = root / "data" / "sdm_models"
    for d in (processed, evs_dir, config_dir, models_dir):
        d.mkdir(parents=True)

    bats = _synthetic_bats()
    bats_path = processed / "bats.geojson"
    bats.to_file(bats_path, driver="GeoJSON")

    boundary = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2000, 2000)],
        crs="EPSG:27700",
    )
    boundary_path = processed / "boundary.geojson"
    boundary.to_file(boundary_path, driver="GeoJSON")

    ev_ds = _synthetic_ev_dataset()
    ev_path = evs_dir / "evs-to-model.tif"
    _write_ev_geotiff(ev_path, ev_ds)

    project_cfg = {
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "models": str(models_dir),
            "tuning_dir": "data/sdm_tuning",
            "config_dir": str(root / "data" / "sdm_config"),
            "predictions": "data/sdm_predictions",
            "model_config": str(root / "model_config.yml"),
            "variables_config": str(root / "variables_config.yml"),
            "occurence_data": str(bats_path),
            "background_points": str(processed / "background.geojson"),
            "boundary": str(boundary_path),
            "grid_points": str(evs_dir / "grid.parquet"),
            "evs": str(evs_dir),
            "ev_tiff": str(ev_path),
        },
        "spatial": {
            "top": 2000,
            "left": 0,
            "crs": "EPSG:27700",
            "resolution": 100,
            "study_area_buffer": 0,
        },
        "crs": "EPSG:27700",
        "mlflow": {"enabled": False, "settings_file": "mlflow.settings.yml"},
        "background": {
            "n_background_points": 80,
            "surface": "uniform",
            "background_method": "contrast",
            "background_value": 0.025,
            "sigma": 6.5,
            "transform_method": "presence",
            "presence_exclusion_m": 0,
            "background_thin_m": 0,
            "presence_thin_m": 0,
            "cap_ratio": None,
            "distance_weights": False,
            "background_distance_weights": False,
            "sample_weight_n_neighbors": 5,
            "class_balance": True,
        },
    }
    project_path = root / "config.yml"
    _write_yaml(project_path, project_cfg)

    model_cfg = {
        "model": {
            "seed": 42,
            "type": "maxent",
            "cv": {
                "method": "kmeans",
                "n_folds": 3,
                "block_size_m": 500,
                "block_n_folds": 3,
            },
            "maxent": {
                "feature_types": ["linear"],
                "beta_multiplier": 1.0,
                "beta_lqp": 1.0,
                "beta_hinge": 1.0,
                "beta_threshold": 1.0,
                "beta_categorical": 1.0,
                "n_hinge_features": 5,
                "n_threshold_features": 0,
                "clamp": True,
                "convergence_tolerance": 1.0e-5,
                "use_lambdas": "best",
                "n_lambdas": 20,
                "class_weights": "balanced",
                "tau": 0.5,
                "transform": "cloglog",
            },
            "sampling": {"min_presence": 10},
            "background": project_cfg["background"],
        }
    }
    model_path = root / "model_config.yml"
    _write_yaml(model_path, model_cfg)

    variables_path = root / "variables_config.yml"
    _write_yaml(variables_path, {"variables": ["ev_a", "ev_b", "ev_c"]})

    _write_yaml(
        config_dir / "model_config.yml",
        {
            "model": {
                "maxent": model_cfg["model"]["maxent"],
            }
        },
    )
    _write_yaml(
        config_dir / "variables_config.yml",
        {"variables": ["ev_a", "ev_b"]},
    )

    return {
        "root": root,
        "project": project_path,
        "model": model_path,
        "variables": variables_path,
        "config_dir": root / "data" / "sdm_config",
        "output": models_dir,
        "ev": ev_path,
        "bats": bats_path,
    }


def test_train_maxent_experiment_forwards_to_train_sdm_models(tmp_path: Path) -> None:
    """Notebook one-liner must be the CLI train path for one species × activity."""
    captured: dict[str, object] = {}

    def _stub(**kwargs: object) -> None:
        captured.update(kwargs)

    with patch.object(tsm, "train_sdm_models", _stub):
        # Import binding inside mex resolves train_sdm_models at call time.
        with patch(
            "sdm.commands.modelling.train_sdm_models.train_sdm_models",
            _stub,
        ):
            mex.train_maxent_experiment(
                latin_name="Plecotus auritus",
                activity_type="Roost",
                output_dir=tmp_path,
                verbose=True,
            )

    assert captured["species"] == ["Plecotus auritus"]
    assert captured["activity_types"] == ["Roost"]
    assert captured["output_dir"] == tmp_path
    assert captured["verbose"] is True


def _inline_train_models_parallel(
    training_data,
    max_threads_per_model: int = 1,
    n_jobs=None,
    *,
    seed: int,
    fold_params=None,
):
    """Avoid ProcessPoolExecutor fork hangs under pytest (GDAL/xarray loaded)."""
    from sdm.models.maxent.training import train_single_model

    return [
        train_single_model(
            data,
            max_threads_per_model,
            seed=seed,
            fold_params=fold_params,
        )
        for data in training_data
    ]


def _balanced_annotated_frames(
    ev_ds: xr.Dataset,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str]]:
    """Interleaved presence/background with EV values — valid kmeans folds."""
    rng = np.random.default_rng(7)
    xs = ev_ds["x"].values
    ys = ev_ds["y"].values
    presence_rows = []
    background_rows = []
    for i, (x, y) in enumerate(
        [(float(xs[i % len(xs)]), float(ys[i % len(ys)])) for i in range(60)]
    ):
        # Sample cell values at nearest centre
        xi = int(np.argmin(np.abs(xs - x)))
        yi = int(np.argmin(np.abs(ys - y)))
        row = {
            "latin_name": "Demo species",
            "activity_type": "Roost",
            "class": 1,
            "ev_a": float(ev_ds["ev_a"].values[yi, xi]),
            "ev_b": float(ev_ds["ev_b"].values[yi, xi]),
            "ev_c": float(ev_ds["ev_c"].values[yi, xi]),
            "grid_x": float(xs[xi]),
            "grid_y": float(ys[yi]),
            "geometry": Point(float(xs[xi]), float(ys[yi])),
        }
        # Presence prefers higher ev_a; background lower — separable signal.
        if i % 2 == 0:
            row["ev_a"] = float(0.7 + 0.3 * rng.random())
            presence_rows.append(row)
        else:
            bg = dict(row)
            bg["class"] = 0
            bg["latin_name"] = "background"
            bg["activity_type"] = "background"
            bg["ev_a"] = float(0.3 * rng.random())
            background_rows.append(bg)
    presence = gpd.GeoDataFrame(presence_rows, crs="EPSG:27700")
    background = gpd.GeoDataFrame(background_rows, crs="EPSG:27700")
    return presence, background, ["ev_a", "ev_b", "ev_c"]


def test_train_from_cache_matches_train_sdm_models_same_seed(tmp_path: Path) -> None:
    """Defaults unchanged → notebook cache path == ``train_sdm_models`` metrics."""
    paths = _project_fixture(tmp_path)
    ev_ds = _synthetic_ev_dataset()
    presence, background, ev_columns = _balanced_annotated_frames(ev_ds)

    def _fixed_prepare(**kwargs):
        return presence.copy(), background.copy(), list(ev_columns)

    with (
        patch(
            "sdm.models.adapters.train_models_parallel",
            _inline_train_models_parallel,
        ),
        patch.object(tsm, "prepare_training_data", _fixed_prepare),
        patch.object(mex, "prepare_training_data", _fixed_prepare),
        patch.object(tsm, "load_environmental_variables", lambda uri: (ev_ds, Path(uri))),
        patch.object(mex, "load_environmental_variables", lambda uri: (ev_ds, Path(uri))),
    ):
        cli_df = tsm.train_sdm_models(
            project_config_path=paths["project"],
            model_config_path=paths["model"],
            variables_config_path=paths["variables"],
            config_dir=paths["config_dir"],
            bats_file=paths["bats"],
            ev_file=paths["ev"],
            output_dir=paths["output"] / "cli",
            species=["Demo species"],
            activity_types=["Roost"],
            n_jobs=1,
            max_threads_per_model=1,
            verbose=False,
        )
        assert len(cli_df) == 1
        cli_mean = float(cli_df.iloc[0]["mean_cv_score"])
        cli_std = float(cli_df.iloc[0]["std_cv_score"])
        assert not np.isnan(cli_mean)

        cache = mex.prepare_maxent_experiment(
            latin_name="Demo species",
            activity_type="Roost",
            project_config_path=paths["project"],
            model_config_path=paths["model"],
            variables_config_path=paths["variables"],
            config_dir=paths["config_dir"],
            bats_file=paths["bats"],
            ev_file=paths["ev"],
            verbose=False,
        )
        assert cache.model_features == ["ev_a", "ev_b"]

        result = mex.train_from_cache(
            cache,
            output_dir=paths["output"] / "notebook",
            set_as_baseline=True,
            n_jobs=1,
            max_threads_per_model=1,
            verbose=False,
        )

        assert result.features == ["ev_a", "ev_b"]
        assert result.mean_cv_score == pytest.approx(cli_mean, rel=0, abs=1e-9)
        assert result.std_cv_score == pytest.approx(cli_std, rel=0, abs=1e-9)
        assert result.baseline_mean_cv == pytest.approx(cli_mean, rel=0, abs=1e-9)

        tweaked = mex.train_from_cache(
            cache,
            maxent_overrides={"beta_multiplier": 5.0},
            output_dir=paths["output"] / "notebook-tweak",
            n_jobs=1,
            max_threads_per_model=1,
            verbose=False,
        )
        assert tweaked.features == ["ev_a", "ev_b"]
        assert tweaked.baseline_mean_cv == pytest.approx(cli_mean, rel=0, abs=1e-9)
        assert tweaked.wall_seconds >= 0.0

    pkg = Path(result.package_dir) / "package.json"
    package = json.loads(pkg.read_text(encoding="utf-8"))
    assert package["feature_names"] == ["ev_a", "ev_b"]


def test_train_from_cache_variable_override_changes_features(tmp_path: Path) -> None:
    paths = _project_fixture(tmp_path)
    cache = mex.prepare_maxent_experiment(
        latin_name="Demo species",
        activity_type="Roost",
        project_config_path=paths["project"],
        model_config_path=paths["model"],
        variables_config_path=paths["variables"],
        config_dir=paths["config_dir"],
        bats_file=paths["bats"],
        ev_file=paths["ev"],
    )
    with patch(
        "sdm.models.adapters.train_models_parallel",
        _inline_train_models_parallel,
    ):
        result = mex.train_from_cache(
            cache,
            variables=["ev_a", "ev_c"],
            output_dir=paths["output"] / "vars",
            set_as_baseline=True,
            n_jobs=1,
            max_threads_per_model=1,
        )
    assert result.features == ["ev_a", "ev_c"]


def test_unknown_override_key_rejected() -> None:
    cache = MaxentExperimentCache_minimal()
    with pytest.raises(ValueError, match="Unknown maxent override"):
        mex.train_from_cache(cache, maxent_overrides={"not_a_knob": 1})


def MaxentExperimentCache_minimal() -> mex.MaxentExperimentCache:
    """Tiny cache that fails fast before training (override validation only)."""
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
    settings = ResolvedBackgroundSettings(
        n_background_points=10,
        surface=BackgroundSurface.UNIFORM,
        background_method=BackgroundMethod.CONTRAST,
        background_value=0.025,
        sigma=1.0,
        transform_method=TransformMethod.PRESENCE,
        surface_resolution_m=100,
        presence_exclusion_m=0.0,
        presence_exclusion_max_m=float("inf"),
        background_thin_m=0.0,
        presence_thin_m=0.0,
        cap_ratio=None,
        min_background=None,
        max_background=None,
        distance_weights=False,
        background_distance_weights=False,
        sample_weight_n_neighbors=5,
        class_balance=True,
        seed=0,
    )
    return mex.MaxentExperimentCache(
        latin_name="Demo species",
        activity_type="Roost",
        presence_with_evs=empty,
        background_with_evs=empty,
        background_occurrence_gdf=empty,
        filtered_occurrence_gdf=empty,
        boundary=empty,
        evs_to_model=xr.Dataset(),
        ev_columns=[],
        background_settings=settings,
        fold_params=FoldParams(
            method=FoldMethod.KMEANS,
            n_folds=3,
            random_state=0,
            block_size_m=500.0,
            block_n_tries=10,
            buffer_m=500.0,
            buffered_loo_warn_above=25,
        ),
        training_seed=0,
        maxent_config=DefaultMaxentConfig(feature_types=["linear"]),
        model_features=["ev_a"],
        model_limited=False,
        model_config_path=Path("model_config.yml"),
        project_config_path=Path("config.yml"),
        config_dir=None,
        trainer="maxent",
        _species_data=empty,
    )
