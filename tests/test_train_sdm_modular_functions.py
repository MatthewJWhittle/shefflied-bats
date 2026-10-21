"""
Tests for modular data preparation functions in train_sdm_models.py.
"""

import logging

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point

# Import functions directly to avoid importing TrainingSetup which may have dependencies
from sdm.commands.modelling.train_sdm_models import (
    _summarize_cv_scores,
    apply_sample_weights,
    drop_duplicate_grid_points,
    generate_training_data,
    get_species_presence_data,
    prepare_species_training_data,
)
from sdm.models.maxent.maxent_model import ActivityType, DefaultMaxentConfig


class TestSummarizeCvScores:
    """Test the _summarize_cv_scores function."""
    
    def test_valid_scores(self):
        """Test with valid scores."""
        scores = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
        mean, std, n_valid, n_total = _summarize_cv_scores(scores)
        
        assert n_valid == 5
        assert n_total == 5
        assert abs(mean - 0.824) < 0.01
        assert std > 0
    
    def test_scores_with_nan(self):
        """Test with some NaN values."""
        scores = np.array([0.8, np.nan, 0.9, 0.75, np.nan])
        mean, std, n_valid, n_total = _summarize_cv_scores(scores)
        
        assert n_valid == 3
        assert n_total == 5
        assert not np.isnan(mean)
        assert not np.isnan(std)
        assert mean == pytest.approx(0.8167, abs=0.01)
    
    def test_all_nan(self):
        """Test with all NaN values."""
        scores = np.array([np.nan, np.nan, np.nan])
        mean, std, n_valid, n_total = _summarize_cv_scores(scores)
        
        assert n_valid == 0
        assert n_total == 3
        assert np.isnan(mean)
        assert np.isnan(std)
    
    def test_empty_array(self):
        """Test with empty array."""
        scores = np.array([])
        mean, std, n_valid, n_total = _summarize_cv_scores(scores)
        
        assert n_valid == 0
        assert n_total == 0
        assert np.isnan(mean)
        assert np.isnan(std)
    
    def test_none_input(self):
        """Test with None input."""
        mean, std, n_valid, n_total = _summarize_cv_scores(None)
        
        assert n_valid == 0
        assert n_total == 0
        assert np.isnan(mean)
        assert np.isnan(std)


class TestGetSpeciesPresenceData:
    """Test the get_species_presence_data function."""
    
    @pytest.fixture
    def sample_presence_data(self):
        """Create sample presence data."""
        data = {
            'latin_name': ['Myotis daubentonii', 'Myotis daubentonii', 'Nyctalus noctula', 'Myotis daubentonii'],
            'activity_type': ['In flight', 'Roost', 'In flight', 'In flight'],
            'geometry': [
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
                Point(3, 3)
            ]
        }
        return gpd.GeoDataFrame(data, crs="EPSG:27700")
    
    def test_filter_by_species_and_activity(self, sample_presence_data):
        """Test filtering by species and activity type."""
        result = get_species_presence_data(
            sample_presence_data,
            latin_name="Myotis daubentonii",
            activity_type="In flight"
        )
        
        assert len(result) == 2
        assert all(result.latin_name == "Myotis daubentonii")
        assert all(result.activity_type == "In flight")
    
    def test_no_matches(self, sample_presence_data):
        """Test when no records match."""
        result = get_species_presence_data(
            sample_presence_data,
            latin_name="Pipistrellus pipistrellus",
            activity_type="In flight"
        )
        
        assert len(result) == 0
    
    def test_missing_columns(self):
        """Test that function asserts on missing columns."""
        gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs="EPSG:27700")
        
        with pytest.raises(AssertionError, match="latin_name column not found"):
            get_species_presence_data(gdf, "Test", "In flight")
        
        gdf['latin_name'] = ['Test']
        with pytest.raises(AssertionError, match="activity_type column not found"):
            get_species_presence_data(gdf, "Test", "In flight")


class TestDropDuplicateGridPoints:
    """Test the drop_duplicate_grid_points function."""
    
    @pytest.fixture
    def sample_grid_data(self):
        """Create sample grid data with duplicates."""
        data = {
            'grid_x': [100.0, 100.0, 200.0, 200.0, 300.0],
            'grid_y': [500.0, 500.0, 600.0, 600.0, 700.0],
            'value': [1, 2, 3, 4, 5],
            'geometry': [
                Point(100, 500),
                Point(100, 500),  # Duplicate
                Point(200, 600),
                Point(200, 600),  # Duplicate
                Point(300, 700)
            ]
        }
        return gpd.GeoDataFrame(data, crs="EPSG:27700")
    
    def test_removes_duplicates(self, sample_grid_data):
        """Test that duplicates are removed."""
        result = drop_duplicate_grid_points(sample_grid_data)
        
        assert len(result) == 3
        assert len(result) < len(sample_grid_data)
    
    def test_default_index_cols(self, sample_grid_data):
        """Test with default index columns."""
        result = drop_duplicate_grid_points(sample_grid_data)
        
        # Should keep first occurrence of each grid_x, grid_y pair
        assert len(result) == 3
        assert result.iloc[0]['value'] == 1  # First occurrence kept
        assert result.iloc[1]['value'] == 3  # First occurrence kept
    
    def test_custom_index_cols(self):
        """Test with custom index columns."""
        data = {
            'x': [100, 100, 200],
            'y': [500, 500, 600],
            'value': [1, 2, 3],
            'geometry': [Point(100, 500), Point(100, 500), Point(200, 600)]
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:27700")
        
        result = drop_duplicate_grid_points(gdf, index_cols=['x', 'y'])
        
        assert len(result) == 2
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        data = {
            'grid_x': [100.0, 200.0, 300.0],
            'grid_y': [500.0, 600.0, 700.0],
            'geometry': [Point(100, 500), Point(200, 600), Point(300, 700)]
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:27700")
        
        result = drop_duplicate_grid_points(gdf)
        
        assert len(result) == 3


def _nathusius_style_frames(
    n_presence_cells: int = 3,
    records_per_cell: int = 2,
    n_background: int = 30,
    ev_name: str = "ev1",
):
    """Presence clustered into few cells + distant background (issue #77 shape)."""
    latin = "Pipistrellus nathusii"
    activity = "Roost"
    presence_rows = []
    for cell in range(n_presence_cells):
        base_x = float(cell * 3000)
        for r in range(records_per_cell):
            presence_rows.append(
                {
                    "latin_name": latin,
                    "activity_type": activity,
                    "class": 1,
                    "grid_x": base_x,
                    "grid_y": 0.0,
                    ev_name: 1.0,
                    "geometry": Point(base_x + r, r),
                }
            )
    presence = gpd.GeoDataFrame(presence_rows, crs="EPSG:27700")

    background = gpd.GeoDataFrame(
        {
            "class": [0] * n_background,
            "grid_x": [20000.0 + i * 2500 for i in range(n_background)],
            "grid_y": [20000.0] * n_background,
            ev_name: [0.5] * n_background,
            "density": list(range(n_background)),
            "geometry": [
                Point(20000 + i * 2500, 20000) for i in range(n_background)
            ],
        },
        crs="EPSG:27700",
    )
    return presence, background, latin, activity, ev_name


class TestSampleWeightsAndPresenceCounts:
    """Issue #77: post-thinning presence must not be logged as 0 via NaN weights."""

    def test_apply_sample_weights_fills_nan_when_few_points(self):
        """elapid returns all-NaN weights when n_points < n_neighbors; fill to 1.0."""
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1000, 0), Point(2000, 0)]},
            crs="EPSG:27700",
        )
        weighted = apply_sample_weights(gdf, n_neighbors=10)
        assert len(weighted) == 3
        assert weighted["sample_weight"].notna().all()
        assert (weighted["sample_weight"] == 1.0).all()

    def test_prepare_keeps_post_thinning_presence_when_weights_would_be_nan(self):
        """
        Before the fix, 3 thinned roost cells + n_neighbors=10 made sample_weight
        NaN, dropna() removed every presence, and the skip log reported 0.
        """
        presence, background, latin, activity, ev = _nathusius_style_frames(
            n_presence_cells=3, records_per_cell=2
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
            n_max_background=20,
            sample_weight_n_neighbors=10,
            random_state=42,
        )
        n_presence = int((result["class"] == 1).sum())
        assert n_presence == 3
        assert result["sample_weight"].notna().all()

    def test_skip_log_reports_real_post_thinning_count_not_zero(self, caplog):
        """Training skip message must use the real cell count (e.g. 3), not 0."""
        presence, background, latin, activity, ev = _nathusius_style_frames(
            n_presence_cells=3, records_per_cell=4
        )
        min_presence = 15
        result = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type=activity,
            ev_columns=[ev],
            grid_size_m=2000,
            sample_weight_n_neighbors=10,
            random_state=0,
        )
        n_presence = int((result["class"] == 1).sum())
        assert n_presence == 3
        assert n_presence < min_presence

        with caplog.at_level(logging.WARNING):
            logging.getLogger(
                "sdm.commands.modelling.train_sdm_models"
            ).warning(
                "Skipping %s - %s: Only %s presence records (minimum %s required)",
                latin,
                activity,
                n_presence,
                min_presence,
            )
        messages = [r.message for r in caplog.records]
        assert any("Only 3 presence records" in m for m in messages)
        assert not any("Only 0 presence records" in m for m in messages)

    def test_below_minimum_still_excluded_from_training_data(self):
        """Do not add a model when post-thinning count is below min_presence."""
        presence, background, latin, activity, ev = _nathusius_style_frames(
            n_presence_cells=3, records_per_cell=2
        )
        # generate_training_data expects density on background via setup path;
        # pass frames shaped like annotated bats + background.
        bats = presence.copy()
        training = generate_training_data(
            bats_ant=bats,
            background_points_gdf=background,
            background_density_series=None,
            grid_points=None,
            latin_names=[latin],
            activity_types=[activity],
            ev_columns=[ev],
            min_presence=15,
            subset=None,
            subset_background=True,
            order_by_density_for_subset=False,
            sample_weight_n_neighbors=10,
            background_min_bg=10,
            background_max_bg=20,
            background_factor=10,
            model_config=DefaultMaxentConfig(),
            feature_selection={ActivityType.ROOST: [ev]},
            grid_size_m=2000,
            seed=42,
        )
        assert training == []

    def test_eleven_in_flight_cells_preserved(self):
        """In-flight shape from #77: 11 cells after thin, still below 15."""
        presence, background, latin, _, ev = _nathusius_style_frames(
            n_presence_cells=11, records_per_cell=1, n_background=40
        )
        presence["activity_type"] = "In flight"
        result = prepare_species_training_data(
            presence_data=presence,
            background_data=background,
            latin_name=latin,
            activity_type="In flight",
            ev_columns=[ev],
            grid_size_m=2000,
            sample_weight_n_neighbors=10,
            random_state=1,
        )
        assert int((result["class"] == 1).sum()) == 11

