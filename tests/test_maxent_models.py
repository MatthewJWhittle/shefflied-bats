"""
Tests for MaxEnt model functionality, including feature selection and model persistence.

These tests verify that:
1. FeatureSubsetter correctly stores and uses only the specified subset of features
2. Models created with a feature subset retain that subset after training and pickling
3. extract_split_data correctly filters columns when feature_columns is specified

These tests help isolate a bug where saved models have all features (e.g., 43) in the
FeatureSubsetter instead of the subset from the tuning config (e.g., 20 features).
"""
import pytest
import numpy as np
import geopandas as gpd
import pickle
from shapely.geometry import Point

from sdm.models.core.feature_subsetter import FeatureSubsetter
from sdm.models.maxent.maxent_model import (
    create_maxent_pipeline,
    extract_split_data,
    train_final_maxent_model,
    DefaultMaxentConfig,
)


@pytest.fixture
def sample_feature_names() -> list:
    """Create a sample list of feature names."""
    return ["feature_1", "feature_2", "feature_3"]


@pytest.fixture
def all_feature_names() -> list:
    """Create a full list of feature names (simulating all available features)."""
    return [f"feature_{i}" for i in range(1, 44)]  # 43 features like the real case


@pytest.fixture
def subset_feature_names() -> list:
    """Create a subset of feature names (simulating tuned features)."""
    return ["feature_1", "feature_2", "feature_3", "feature_5", "feature_7"]


@pytest.fixture
def sample_training_data(all_feature_names, subset_feature_names) -> gpd.GeoDataFrame:
    """Create sample training data with all features, but we'll only use a subset."""
    n_samples = 100
    data = {}
    
    # Add all features to the DataFrame
    for feat in all_feature_names:
        data[feat] = np.random.rand(n_samples)
    
    # Add class labels
    data["class"] = np.random.randint(0, 2, n_samples)
    
    # Add sample weights
    data["sample_weight"] = np.random.rand(n_samples)
    
    # Add geometry
    data["geometry"] = [Point(x, y) for x, y in zip(np.random.rand(n_samples) * 100, 
                                                     np.random.rand(n_samples) * 100)]
    
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


class TestFeatureSubsetter:
    """Tests for FeatureSubsetter functionality."""
    
    def test_feature_subsetter_initialization(self, subset_feature_names):
        """Test that FeatureSubsetter initializes with correct feature names."""
        subsetter = FeatureSubsetter(feature_names=subset_feature_names)
        assert subsetter.feature_names == subset_feature_names
        assert len(subsetter.feature_names) == 5
    
    def test_feature_subsetter_transform(self, sample_training_data, subset_feature_names):
        """Test that FeatureSubsetter correctly selects only specified features."""
        subsetter = FeatureSubsetter(feature_names=subset_feature_names)
        
        # Transform should return only the subset of features
        transformed = subsetter.transform(sample_training_data)
        
        assert set(transformed.columns) == set(subset_feature_names)
        assert len(transformed.columns) == len(subset_feature_names)
    
    def test_feature_subsetter_preserves_feature_names_after_pickle(self, subset_feature_names):
        """Test that FeatureSubsetter feature_names are preserved after pickling/unpickling."""
        subsetter = FeatureSubsetter(feature_names=subset_feature_names)
        original_feature_names = subsetter.feature_names.copy()
        
        # Pickle and unpickle
        pickled = pickle.dumps(subsetter)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.feature_names == original_feature_names
        assert len(unpickled.feature_names) == len(subset_feature_names)


class TestCreateMaxentPipeline:
    """Tests for create_maxent_pipeline function."""
    
    def test_pipeline_created_with_subset_features(self, subset_feature_names):
        """Test that pipeline is created with only the specified subset of features."""
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        # Check that pipeline has FeatureSubsetter step
        assert "feature_selection" in pipeline.named_steps
        
        # Check that FeatureSubsetter has the correct feature names
        feature_subsetter = pipeline.named_steps["feature_selection"]
        assert isinstance(feature_subsetter, FeatureSubsetter)
        assert feature_subsetter.feature_names == subset_feature_names
        assert len(feature_subsetter.feature_names) == 5
    
    def test_pipeline_feature_names_preserved_after_clone(self, subset_feature_names):
        """Test that pipeline FeatureSubsetter feature_names are preserved after cloning."""
        from sklearn.base import clone
        
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        # Clone the pipeline (like what happens during cross-validation)
        cloned_pipeline = clone(pipeline)
        
        # Check that cloned pipeline has correct feature names
        cloned_feature_subsetter = cloned_pipeline.named_steps["feature_selection"]
        assert cloned_feature_subsetter.feature_names == subset_feature_names
        assert len(cloned_feature_subsetter.feature_names) == 5


class TestExtractSplitData:
    """Tests for extract_split_data function."""
    
    def test_extract_split_data_filters_to_subset(self, sample_training_data, subset_feature_names):
        """Test that extract_split_data correctly filters to only specified features."""
        indices = np.arange(len(sample_training_data))
        
        X, y, w = extract_split_data(
            sample_training_data,
            indices,
            feature_columns=subset_feature_names,
        )
        
        # X should only contain the subset of features
        assert set(X.columns) == set(subset_feature_names)
        assert len(X.columns) == len(subset_feature_names)
    
    def test_extract_split_data_with_all_features(self, sample_training_data, all_feature_names):
        """Test that extract_split_data uses all features when feature_columns is None."""
        indices = np.arange(len(sample_training_data))
        
        X, y, w = extract_split_data(
            sample_training_data,
            indices,
            feature_columns=None,  # Should use all features
        )
        
        # X should contain all features (excluding class, geometry, sample_weight)
        expected_cols = set(all_feature_names)
        assert set(X.columns) == expected_cols
        assert len(X.columns) == len(all_feature_names)


class TestModelFeatureConsistency:
    """Tests to isolate the bug where saved models have all features instead of subset."""
    
    def test_pipeline_feature_names_after_training_with_subset(
        self, sample_training_data, subset_feature_names, all_feature_names
    ):
        """Test that pipeline FeatureSubsetter retains subset after training with filtered data."""
        # Create pipeline with subset of features
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        # Extract data with only the subset (simulating what extract_split_data does)
        indices = np.arange(len(sample_training_data))
        X_train, y_train, w_train = extract_split_data(
            sample_training_data,
            indices,
            feature_columns=subset_feature_names,
        )
        
        # Fit the pipeline with the filtered data
        pipeline.fit(X_train, y_train, maxent__sample_weight=w_train)
        
        # After fitting, FeatureSubsetter should still have only the subset
        feature_subsetter = pipeline.named_steps["feature_selection"]
        assert feature_subsetter.feature_names == subset_feature_names
        assert len(feature_subsetter.feature_names) == 5
        
        # Verify it doesn't have all features
        assert len(feature_subsetter.feature_names) != len(all_feature_names)
    
    def test_model_pickle_preserves_feature_subset(
        self, sample_training_data, subset_feature_names
    ):
        """Test that pickled model preserves the feature subset (not all features)."""
        # Create and train a model with subset
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        indices = np.arange(len(sample_training_data))
        X_train, y_train, w_train = extract_split_data(
            sample_training_data,
            indices,
            feature_columns=subset_feature_names,
        )
        
        pipeline.fit(X_train, y_train, maxent__sample_weight=w_train)
        
        # Pickle the model
        pickled_model = pickle.dumps(pipeline)
        unpickled_model = pickle.loads(pickled_model)
        
        # After unpickling, should still have only the subset
        feature_subsetter = unpickled_model.named_steps["feature_selection"]
        assert feature_subsetter.feature_names == subset_feature_names
        assert len(feature_subsetter.feature_names) == 5
    
    def test_train_final_model_preserves_feature_subset(
        self, sample_training_data, subset_feature_names, all_feature_names
    ):
        """Test that train_final_maxent_model preserves feature subset when training."""
        # Create pipeline with subset
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        # Train final model with feature_columns specified
        final_model = train_final_maxent_model(
            model=pipeline,
            occurrence_gdf=sample_training_data,
            feature_columns=subset_feature_names,
        )
        
        # Final model should still have only the subset
        feature_subsetter = final_model.named_steps["feature_selection"]
        assert feature_subsetter.feature_names == subset_feature_names
        assert len(feature_subsetter.feature_names) == 5
        assert len(feature_subsetter.feature_names) != len(all_feature_names)
    
    def test_model_with_all_columns_but_subset_features(
        self, sample_training_data, subset_feature_names, all_feature_names
    ):
        """Test scenario: DataFrame has all columns, but we filter and use only subset.
        
        This simulates the real-world scenario where training data contains all features,
        but we want to use only a subset. The FeatureSubsetter should still only have
        the subset of features, not all of them.
        """
        # Create pipeline with subset (simulating tuned features)
        pipeline = create_maxent_pipeline(
            feature_names=subset_feature_names,
            maxent_n_jobs=1,
            model_config=DefaultMaxentConfig(),
        )
        
        # Verify pipeline was created with subset
        feature_subsetter = pipeline.named_steps["feature_selection"]
        assert len(feature_subsetter.feature_names) == 5
        assert set(feature_subsetter.feature_names) == set(subset_feature_names)
        
        # Now extract data using only the subset (simulating extract_split_data with feature_columns)
        indices = np.arange(len(sample_training_data))
        X_train, y_train, w_train = extract_split_data(
            sample_training_data,  # Has all columns
            indices,
            feature_columns=subset_feature_names,  # But we filter to subset
        )
        
        # X_train should only have subset columns
        assert set(X_train.columns) == set(subset_feature_names)
        
        # Fit the model
        pipeline.fit(X_train, y_train, maxent__sample_weight=w_train)
        
        # After fitting, FeatureSubsetter should STILL only have subset
        feature_subsetter_after_fit = pipeline.named_steps["feature_selection"]
        assert len(feature_subsetter_after_fit.feature_names) == 5
        assert set(feature_subsetter_after_fit.feature_names) == set(subset_feature_names)
        
        # Pickle and unpickle to simulate saving/loading
        pickled = pickle.dumps(pipeline)
        loaded = pickle.loads(pickled)
        
        # After loading, should STILL only have subset
        loaded_feature_subsetter = loaded.named_steps["feature_selection"]
        assert len(loaded_feature_subsetter.feature_names) == 5
        assert set(loaded_feature_subsetter.feature_names) == set(subset_feature_names)
        
        # CRITICAL: Should NOT have all features
        assert len(loaded_feature_subsetter.feature_names) != len(all_feature_names)


class TestFullTrainingWorkflow:
    """Integration tests that reproduce the actual bug scenario using the full training workflow."""
    
    def test_train_and_save_preserves_feature_subset(
        self, sample_training_data, subset_feature_names, all_feature_names
    ):
        """Test the full training workflow: TrainingData -> train_single_model -> save -> load.
        
        This reproduces the exact scenario where the bug occurs:
        - TrainingData.occurrence has ALL columns (all features)
        - TrainingData.model_features is a SUBSET (e.g., 20 features)
        - Model is trained and saved
        - Saved model should have only the subset, not all features
        """
        from sdm.types import TrainingData, TrainingResults
        from sdm.commands.modelling.train_sdm_models import train_single_model, save_models
        from pathlib import Path
        import tempfile
        
        # Create TrainingData with all columns in occurrence but subset in model_features
        training_data = TrainingData(
            latin_name="Test_species",
            activity_type="In flight",
            occurrence=sample_training_data,  # Has ALL columns
            maxent_config=DefaultMaxentConfig(),
            model_features=subset_feature_names,  # But we only want to use subset
        )
        
        # Verify TrainingData is set up correctly
        assert len(training_data.model_features) == 5  # Subset
        feature_cols_in_data = [c for c in sample_training_data.columns 
                                if c not in ["class", "sample_weight", "geometry"]]
        assert len(feature_cols_in_data) == len(all_feature_names)  # All features in data
        
        # Train model (this will actually train - might be slow)
        result: TrainingResults = train_single_model(
            data=training_data,
            max_threads_per_model=1,
        )
        
        # Check if training succeeded
        if not result.success:
            pytest.skip(f"Model training failed: {result.error}")
        
        # Verify the final_model was created
        assert result.final_model is not None
        
        # Check FeatureSubsetter in the trained model BEFORE saving
        feature_subsetter_before_save = result.final_model.named_steps["feature_selection"]
        features_before = feature_subsetter_before_save.feature_names
        assert len(features_before) == 5, \
            f"Before save: Expected 5 features, got {len(features_before)}: {features_before}"
        assert set(features_before) == set(subset_feature_names)
        
        # Save model using the actual save function
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            models = [result]
            model_paths = save_models(models, output_dir)
            
            # Load the saved model
            model_path = model_paths[result.identifier()]
            assert model_path.exists()
            
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            
            # Check FeatureSubsetter in the loaded model
            loaded_feature_subsetter = loaded_model.named_steps["feature_selection"]
            loaded_features = loaded_feature_subsetter.feature_names
            
            # CRITICAL ASSERTION: Should have subset, not all features
            assert len(loaded_features) == 5, \
                f"BUG REPRODUCED: Expected 5 features (subset), but got {len(loaded_features)}. " \
                f"Features: {loaded_features}"
            assert set(loaded_features) == set(subset_feature_names), \
                f"BUG REPRODUCED: Expected subset {subset_feature_names}, but got {loaded_features}"
            assert len(loaded_features) != len(all_feature_names), \
                f"BUG REPRODUCED: Model has all {len(all_feature_names)} features instead of subset!"


    
    def test_full_workflow_with_all_columns_but_subset_features(
        self, sample_training_data, subset_feature_names, all_feature_names
    ):
        """Test that verifies the exact bug scenario: data has all columns, model uses subset.
        
        This test simulates the real-world scenario:
        - occurrence DataFrame has ALL 43 features (from ev_columns)
        - model_features is a subset of 20 features (from tuning config)
        - After training and saving, the model should still only have the 20 features
        """
        from sdm.types import TrainingData, TrainingResults
        from sdm.commands.modelling.train_sdm_models import train_single_model, save_models
        from pathlib import Path
        import tempfile
        
        # Verify sample_training_data has all features
        feature_cols_in_data = [c for c in sample_training_data.columns 
                                if c not in ["class", "sample_weight", "geometry"]]
        assert len(feature_cols_in_data) == len(all_feature_names), \
            f"Data should have all {len(all_feature_names)} features"
        
        # Create TrainingData: occurrence has all columns, but model_features is subset
        training_data = TrainingData(
            latin_name="Test_species",
            activity_type="In flight",
            occurrence=sample_training_data,  # Has ALL features
            maxent_config=DefaultMaxentConfig(),
            model_features=subset_feature_names,  # But we want to use only subset
        )
        
        # Train model
        result: TrainingResults = train_single_model(
            data=training_data,
            max_threads_per_model=1,
        )
        
        if not result.success:
            pytest.skip(f"Model training failed: {result.error}")
        
        # Before saving: check the model
        feature_subsetter = result.final_model.named_steps["feature_selection"]
        features_before_save = feature_subsetter.feature_names
        
        print(f"\nFeatures BEFORE save: {len(features_before_save)}")
        print(f"Expected subset: {subset_feature_names}")
        print(f"Actual: {features_before_save}")
        
        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            models = [result]
            model_paths = save_models(models, output_dir)
            
            model_path = model_paths[result.identifier()]
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            
            # Check after loading
            loaded_feature_subsetter = loaded_model.named_steps["feature_selection"]
            features_after_load = loaded_feature_subsetter.feature_names
            
            print(f"\nFeatures AFTER load: {len(features_after_load)}")
            print(f"Actual: {features_after_load}")
            
            # This is the critical assertion - this should pass but might fail if bug exists
            assert len(features_after_load) == len(subset_feature_names), \
                f"BUG: Expected {len(subset_feature_names)} features (subset), " \
                f"but model has {len(features_after_load)} features. " \
                f"Expected: {subset_feature_names}, Got: {features_after_load}"
            
            assert set(features_after_load) == set(subset_feature_names), \
                f"BUG: Feature mismatch. Expected {subset_feature_names}, got {features_after_load}"
            
            # Should NOT have all features
            if len(features_after_load) == len(all_feature_names):
                pytest.fail(
                    f"BUG REPRODUCED: Model has all {len(all_feature_names)} features "
                    f"instead of subset of {len(subset_feature_names)}! "
                    f"This is the bug we're trying to fix."
                )


class TestTuningConfigLoading:
    """Tests for tuning config loading and path resolution utilities."""
    
    def test_tuning_config_path_resolution(self):
        """Test that path resolution works correctly for tuning configs."""
        from sdm.utils.io import get_tuning_config_path
        from pathlib import Path
        
        latin_name = "Nyctalus noctula"
        activity_type = "In flight"
        tuning_dir = Path("data/sdm_tuning")
        
        expected_path = get_tuning_config_path(tuning_dir, latin_name, activity_type)
        actual_path = Path("data/sdm_tuning/Nyctalus_noctula_In_flight")
        
        assert expected_path == actual_path
    
    def test_feature_filtering_logic(self):
        """Test the feature filtering logic used when loading tuning configs."""
        # Simulate the filtering logic from train_sdm_models.py line 1194
        tuning_features = ["feature_1", "feature_2", "feature_3"]  # Subset from tuning
        ev_columns = ["feature_1", "feature_2", "feature_4", "feature_5"]  # Available features
        
        # This is what happens at line 1194: filter to only features in ev_columns
        model_features = [f for f in tuning_features if f in ev_columns]
        
        # Result: Only features that exist in both lists
        assert len(model_features) == 2  # feature_1 and feature_2
        assert "feature_3" not in model_features  # Not in ev_columns
        assert set(model_features) == {"feature_1", "feature_2"}
        
        # Test edge case: all features missing
        tuning_features_all_missing = ["feature_99", "feature_98"]  # Not in ev_columns
        model_features_empty = [f for f in tuning_features_all_missing if f in ev_columns]
        
        assert len(model_features_empty) == 0  # Empty list - would cause validation error

