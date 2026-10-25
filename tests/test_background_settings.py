"""Selectable background knobs (#103): defaults stay live-identical; each knob works."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from sdm.commands.modelling.train_sdm_models import (
    apply_sample_weights,
    drop_duplicate_grid_points,
    filter_background_by_distance,
    grid_sample_points,
    prepare_species_training_data,
    save_run_metadata,
)
from sdm.occurrence.background_settings import (
    BackgroundSurface,
    DEFAULT_CAP_RATIO,
    DEFAULT_N_BACKGROUND_POINTS,
    DEFAULT_SAMPLE_WEIGHT_N_NEIGHBORS,
    resolve_background_settings,
)
from sdm.occurrence.sampling import BackgroundMethod, TransformMethod
from sdm.types import BackgroundConfig, MaxentConfigModel, ModelConfig


def _points(xy, **cols):
    geom = [Point(x, y) for x, y in xy]
    data = {"geometry": geom}
    data.update(cols)
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


def _synthetic_species_frames():
    """Presence cluster + denser background pool with EV columns (no NAs)."""
    latin = "Myotis brandtii"
    activity = "Roost"
    ev = "elev"
    presence = _points(
        [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (3000.0, 0.0), (3100.0, 0.0)],
        latin_name=latin,
        activity_type=activity,
        grid_x=[0.0, 0.0, 0.0, 3000.0, 3000.0],
        grid_y=[0.0, 0.0, 0.0, 0.0, 0.0],
        **{"class": 1, ev: [1.0] * 5},
    )
    # Background: some near presence (<500 m), some far; unique 2 km cells
    bg_xy = [
        (100.0, 0.0),  # near
        (200.0, 0.0),  # near
        (2000.0, 0.0),
        (4000.0, 0.0),
        (6000.0, 0.0),
        (8000.0, 0.0),
        (10000.0, 0.0),
        (12000.0, 0.0),
        (14000.0, 0.0),
        (16000.0, 0.0),
        (18000.0, 0.0),
        (20000.0, 0.0),
    ]
    background = _points(
        bg_xy,
        grid_x=[float(x) for x, _ in bg_xy],
        grid_y=[0.0] * len(bg_xy),
        density=list(range(len(bg_xy))),
        **{"class": 0, ev: [0.5] * len(bg_xy)},
    )
    return presence, background, latin, activity, ev


def _legacy_default_prepare(presence, background, latin, activity, ev, *, seed=42):
    """Independent recreation of pre-#103 live-train preparation steps."""
    species_presence = presence[
        (presence.latin_name == latin) & (presence.activity_type == activity)
    ].copy()
    species_presence = drop_duplicate_grid_points(species_presence)
    species_absence = filter_background_by_distance(
        background, species_presence, d_min=500, d_max=np.inf
    )
    species_absence = drop_duplicate_grid_points(species_absence)
    species_presence = grid_sample_points(
        species_presence, grid_size_x=2000, grid_size_y=2000, random_state=seed
    )
    species_absence = grid_sample_points(
        species_absence, grid_size_x=2000, grid_size_y=2000, random_state=seed
    )
    n_max = len(species_presence) * 10
    species_absence = species_absence.sample(
        n=min(len(species_absence), n_max), replace=False, random_state=seed
    )
    species_presence = apply_sample_weights(species_presence, n_neighbors=10)
    species_absence = apply_sample_weights(species_absence, n_neighbors=10)
    import pandas as pd

    out = pd.concat([species_presence, species_absence])
    out = out[["class", "sample_weight", "geometry", ev]].dropna(subset=[ev])
    return gpd.GeoDataFrame(out, geometry="geometry", crs=presence.crs)


class TestResolveBackgroundSettings:
    def test_defaults_match_live_train(self) -> None:
        settings = resolve_background_settings(seed=42)
        assert settings.n_background_points == DEFAULT_N_BACKGROUND_POINTS
        assert settings.surface == BackgroundSurface.ALL_BAT
        assert settings.background_method == BackgroundMethod.CONTRAST
        assert settings.background_value == pytest.approx(0.025)
        assert settings.sigma == pytest.approx(6.5)
        assert settings.transform_method == TransformMethod.PRESENCE
        assert settings.presence_exclusion_m == pytest.approx(500.0)
        assert settings.background_thin_m == pytest.approx(2000.0)
        assert settings.presence_thin_m == pytest.approx(2000.0)
        assert settings.cap_ratio == pytest.approx(DEFAULT_CAP_RATIO)
        assert settings.min_background is None
        assert settings.max_background is None
        assert settings.distance_weights is True
        assert settings.background_distance_weights is True
        assert settings.sample_weight_n_neighbors == DEFAULT_SAMPLE_WEIGHT_N_NEIGHBORS
        assert settings.class_balance is True
        assert settings.seed == 42

    def test_model_background_overrides(self) -> None:
        model = ModelConfig.model_validate(
            {
                "seed": 7,
                "maxent": {
                    "feature_types": ["linear"],
                    "beta_multiplier": 1.0,
                    "beta_lqp": 1.0,
                    "beta_hinge": 1.0,
                    "beta_threshold": 1.0,
                    "beta_categorical": 1.0,
                    "n_hinge_features": 1,
                    "n_threshold_features": 1,
                    "clamp": True,
                    "convergence_tolerance": 1e-5,
                    "use_lambdas": "best",
                    "n_lambdas": 10,
                    "class_weights": "balanced",
                    "tau": 0.5,
                    "transform": "cloglog",
                },
                "background": {
                    "n_background_points": 10000,
                    "cap_ratio": None,
                    "presence_exclusion_m": 0,
                    "background_thin_m": 0,
                    "background_distance_weights": False,
                },
            }
        )
        settings = resolve_background_settings(model_config=model)
        assert settings.n_background_points == 10000
        assert settings.cap_ratio is None
        assert settings.presence_exclusion_m == 0.0
        assert settings.background_thin_m == 0.0
        assert settings.background_distance_weights is False
        assert settings.presence_thin_m == pytest.approx(2000.0)
        assert settings.class_balance is True
        assert settings.seed == 7

    def test_sampling_grid_size_fills_thin_when_background_omits(self) -> None:
        model = ModelConfig.model_validate(
            {
                "seed": 1,
                "maxent": {
                    "feature_types": ["linear"],
                    "beta_multiplier": 1.0,
                    "beta_lqp": 1.0,
                    "beta_hinge": 1.0,
                    "beta_threshold": 1.0,
                    "beta_categorical": 1.0,
                    "n_hinge_features": 1,
                    "n_threshold_features": 1,
                    "clamp": True,
                    "convergence_tolerance": 1e-5,
                    "use_lambdas": "best",
                    "n_lambdas": 10,
                    "class_weights": "balanced",
                    "tau": 0.5,
                    "transform": "cloglog",
                },
                "sampling": {"grid_size_m": 3000},
            }
        )
        settings = resolve_background_settings(model_config=model)
        assert settings.presence_thin_m == pytest.approx(3000.0)
        assert settings.background_thin_m == pytest.approx(3000.0)

    def test_cli_overrides_win(self) -> None:
        settings = resolve_background_settings(
            seed=42,
            n_background_points_override=50000,
            presence_exclusion_m_override=0.0,
            sample_weight_n_neighbors_override=3,
        )
        assert settings.n_background_points == 50000
        assert settings.presence_exclusion_m == 0.0
        assert settings.sample_weight_n_neighbors == 3

    def test_metadata_records_every_knob(self) -> None:
        settings = resolve_background_settings(seed=42)
        meta = settings.to_metadata()
        for key in (
            "n_background_points",
            "surface",
            "background_method",
            "background_value",
            "sigma",
            "transform_method",
            "surface_resolution_m",
            "presence_exclusion_m",
            "presence_exclusion_max_m",
            "background_thin_m",
            "presence_thin_m",
            "cap_ratio",
            "min_background",
            "max_background",
            "distance_weights",
            "background_distance_weights",
            "sample_weight_n_neighbors",
            "class_balance",
            "seed",
        ):
            assert key in meta


class TestPrepareSpeciesTrainingDataDefaults:
    def test_default_config_byte_identical_to_legacy_main(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        expected = _legacy_default_prepare(
            presence, background, latin, activity, ev, seed=42
        )
        result = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            grid_size_m=2000,
            d_min=500,
            d_max=np.inf,
            sample_weight_n_neighbors=10,
            random_state=42,
            presence_thin_m=2000,
            background_thin_m=2000,
            distance_weights=True,
            background_distance_weights=True,
            cap_ratio=10,
        )
        assert len(result) == len(expected)
        exp_bg = expected[expected["class"] == 0].sort_index()
        got_bg = result[result["class"] == 0].sort_index()
        assert list(got_bg.geometry.x) == list(exp_bg.geometry.x)
        assert list(got_bg.geometry.y) == list(exp_bg.geometry.y)
        np.testing.assert_allclose(
            got_bg["sample_weight"].to_numpy(),
            exp_bg["sample_weight"].to_numpy(),
            rtol=0,
            atol=0,
        )
        exp_pr = expected[expected["class"] == 1].sort_index()
        got_pr = result[result["class"] == 1].sort_index()
        assert list(got_pr.geometry.x) == list(exp_pr.geometry.x)
        np.testing.assert_allclose(
            got_pr["sample_weight"].to_numpy(),
            exp_pr["sample_weight"].to_numpy(),
            rtol=0,
            atol=0,
        )


class TestBackgroundKnobs:
    def test_presence_exclusion_zero_keeps_near_points(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        with_exclusion = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=500,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            background_distance_weights=False,
            distance_weights=False,
            random_state=0,
        )
        without = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            background_distance_weights=False,
            distance_weights=False,
            random_state=0,
        )
        assert int((without["class"] == 0).sum()) > int(
            (with_exclusion["class"] == 0).sum()
        )

    def test_background_thin_zero_keeps_all_cells(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        thinned = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=2000,
            presence_thin_m=2000,
            cap_ratio=None,
            background_distance_weights=False,
            distance_weights=False,
            random_state=1,
        )
        unthinned = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            background_distance_weights=False,
            distance_weights=False,
            random_state=1,
        )
        assert int((unthinned["class"] == 0).sum()) >= int(
            (thinned["class"] == 0).sum()
        )

    def test_cap_ratio_null_keeps_full_pool(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        capped = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=1,
            background_distance_weights=False,
            distance_weights=False,
            random_state=2,
        )
        uncapped = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            background_distance_weights=False,
            distance_weights=False,
            random_state=2,
        )
        n_pres = int((uncapped["class"] == 1).sum())
        assert int((capped["class"] == 0).sum()) == n_pres
        assert int((uncapped["class"] == 0).sum()) == len(background)

    def test_background_distance_weights_off_are_uniform(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        result = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            distance_weights=True,
            background_distance_weights=False,
            sample_weight_n_neighbors=10,
            random_state=3,
        )
        bg = result[result["class"] == 0]
        pr = result[result["class"] == 1]
        assert (bg["sample_weight"] == 1.0).all()
        # Presence still distance-weighted (or filled to 1 when too few points).
        assert pr["sample_weight"].notna().all()

    def test_as_drawn_trial_keeps_exactly_n_background(self) -> None:
        presence, background, latin, activity, ev = _synthetic_species_frames()
        n = len(background)
        result = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            d_min=0,
            background_thin_m=0,
            presence_thin_m=2000,
            cap_ratio=None,
            distance_weights=True,
            background_distance_weights=False,
            random_state=4,
        )
        assert int((result["class"] == 0).sum()) == n
        assert (result.loc[result["class"] == 0, "sample_weight"] == 1.0).all()

    def test_class_balance_flag_maps_to_maxent_weights(self) -> None:
        on = resolve_background_settings(
            model_config=None,
            seed=1,
        )
        assert on.maxent_class_weights() == "balanced"
        off_cfg = BackgroundConfig(class_balance=False)

        model = ModelConfig(
            seed=1,
            maxent=MaxentConfigModel(
                feature_types=["linear"],
                beta_multiplier=1.0,
                beta_lqp=1.0,
                beta_hinge=1.0,
                beta_threshold=1.0,
                beta_categorical=1.0,
                n_hinge_features=1,
                n_threshold_features=1,
                clamp=True,
                convergence_tolerance=1e-5,
                use_lambdas="best",
                n_lambdas=10,
                class_weights="balanced",
                tau=0.5,
                transform="cloglog",
            ),
            background=off_cfg,
        )
        settings = resolve_background_settings(model_config=model)
        assert settings.class_balance is False
        assert settings.maxent_class_weights() == 1.0


class TestRunMetadataBackground:
    def test_run_metadata_records_background(self, tmp_path: Path) -> None:
        settings = resolve_background_settings(seed=42)
        path = save_run_metadata(
            tmp_path,
            seed=42,
            model_config_path="model_config.yml",
            background_settings=settings,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["background"]["n_background_points"] == 8000
        assert payload["background"]["surface"] == "all_bat"
        assert payload["background"]["cap_ratio"] == 10
        assert payload["background"]["sample_weight_n_neighbors"] == 10
