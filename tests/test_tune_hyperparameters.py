"""
Minimal tests for hyperparameter tuning utility functions.
"""

import pytest
from unittest.mock import Mock

from sdm.commands.modelling.tune_hyperparameters import (
    get_default_trial_params,
    pick_features,
)


class TestGetDefaultTrialParams:
    """Test the get_default_trial_params function."""
    
    def test_basic_functionality(self):
        """Test basic parameter generation."""
        features = ["temp", "precip", "elevation"]
        params = get_default_trial_params(features)
        
        # Check feature selection params
        assert params["feature_temp"] is True
        assert params["feature_precip"] is True
        assert params["feature_elevation"] is True
        
        # Check feature type params
        assert params["feature_type_linear"] is True
        assert params["feature_type_quadratic"] is True
        assert params["feature_type_hinge"] is True
        assert params["feature_type_threshold"] is False
        
        # Check beta parameters
        assert params["beta_multiplier"] == 1.5
        assert params["beta_lqp"] == 1.0
        assert params["beta_hinge"] == 1.0
        assert params["beta_threshold"] == 1.0
        assert params["beta_categorical"] == 1.0
        
        # Check feature counts
        assert params["n_hinge_features"] == 10
        assert params["n_threshold_features"] == 0
    
    def test_empty_features(self):
        """Test with empty feature list."""
        params = get_default_trial_params([])
        
        # Should still have feature type and beta params
        assert "feature_type_linear" in params
        assert "beta_multiplier" in params
        # But no feature selection params
        assert not any(k.startswith("feature_") and k != "feature_type_linear" 
                      and k != "feature_type_quadratic" 
                      and k != "feature_type_hinge" 
                      and k != "feature_type_threshold"
                      for k in params.keys())
    
    def test_single_feature(self):
        """Test with single feature."""
        params = get_default_trial_params(["temp"])
        
        assert params["feature_temp"] is True
        assert len([k for k in params.keys() if k.startswith("feature_") and k != "feature_type_linear"]) == 4  # temp + 3 feature types


class TestPickFeatures:
    """Test the pick_features function."""
    
    def test_all_features_selected(self):
        """Test when all features are selected."""
        features = ["temp", "precip", "elevation"]
        
        # Create mock trial that selects all features
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: True)
        
        result = pick_features(mock_trial, features)
        
        assert len(result) == 3
        assert set(result) == set(features)
        assert result == sorted(features)  # Should be sorted
    
    def test_no_features_selected(self):
        """Test when no features are selected."""
        features = ["temp", "precip", "elevation"]
        
        # Create mock trial that selects no features
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: False)
        
        result = pick_features(mock_trial, features)
        
        assert len(result) == 0
        assert result == []
    
    def test_partial_selection(self):
        """Test when some features are selected."""
        features = ["temp", "precip", "elevation"]
        
        # Create mock trial that selects first feature only
        selection_map = {
            "feature_temp": True,
            "feature_precip": False,
            "feature_elevation": False,
        }
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: selection_map[name])
        
        result = pick_features(mock_trial, features)
        
        assert len(result) == 1
        assert result == ["temp"]
    
    def test_parameter_names(self):
        """Test that correct parameter names are used."""
        features = ["temp", "precip"]
        
        mock_trial = Mock()
        call_args = []
        
        def capture_call(name, choices):
            call_args.append(name)
            return True
        
        mock_trial.suggest_categorical = Mock(side_effect=capture_call)
        
        pick_features(mock_trial, features)
        
        # Check that correct parameter names were used
        assert "feature_temp" in call_args
        assert "feature_precip" in call_args
    
    def test_empty_features_list(self):
        """Test with empty feature list."""
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(return_value=True)
        
        result = pick_features(mock_trial, [])
        
        assert len(result) == 0
        assert result == []
        # Should not call suggest_categorical for empty list
        mock_trial.suggest_categorical.assert_not_called()

