"""
Tests for modular data preparation functions in train_sdm_models.py.
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point

# Import functions directly to avoid importing TrainingSetup which may have dependencies
from sdm.commands.modelling.train_sdm_models import (
    _summarize_cv_scores,
    get_species_presence_data,
    drop_duplicate_grid_points,
)


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

