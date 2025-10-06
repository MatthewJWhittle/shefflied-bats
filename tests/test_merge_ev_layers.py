"""
Tests for merge EV layers functionality.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, Mock

from sdm.commands.data_preparation.processing.merge_ev_layers import (
    parse_dataset_input,
    merge_ev_layers
)


class TestParseDatasetInput:
    """Test the parse_dataset_input function."""
    
    def test_valid_inputs(self):
        """Test parsing valid dataset inputs."""
        inputs = ["dtm=data/dtm.tif", "dsm=data/dsm.tif", "land_cover=data/lc.tif"]
        result = parse_dataset_input(inputs)
        
        expected = {
            "dtm": Path("data/dtm.tif"),
            "dsm": Path("data/dsm.tif"),
            "land_cover": Path("data/lc.tif")
        }
        assert result == expected
    
    def test_name_cleaning(self):
        """Test that dataset names are cleaned using tidy_variable_name."""
        inputs = ["DTM-1m=data/dtm.tif", "Land Cover=data/lc.tif"]
        result = parse_dataset_input(inputs)
        
        expected = {
            "dtm_1m": Path("data/dtm.tif"),
            "land_cover": Path("data/lc.tif")
        }
        assert result == expected
    
    def test_invalid_inputs(self):
        """Test handling of invalid input formats."""
        inputs = ["valid=data/file.tif", "invalid_format"]
        result = parse_dataset_input(inputs)
        
        # Should only include valid entries
        expected = {"valid": Path("data/file.tif")}
        assert result == expected


class TestMergeEvLayersWorkflow:
    """Test the complete merge_ev_layers workflow."""
    
    def test_merge_ev_layers_no_valid_datasets(self):
        """Test error handling when no valid datasets are provided."""
        with pytest.raises(ValueError, match="No valid datasets provided"):
            merge_ev_layers(
                dataset_inputs=[],
                output_path=Path("output.tif"),
                boundary_path=Path("boundary.geojson")
            )
    
    def test_merge_ev_layers_basic_workflow(self, tmp_path):
        """Test basic workflow with mocked dependencies."""
        # Create test files
        boundary_file = tmp_path / "boundary.geojson"
        boundary_file.touch()
        
        raster_file = tmp_path / "test.tif"
        raster_file.touch()
        
        output_path = tmp_path / "output.tif"
        
        with patch('sdm.commands.data_preparation.processing.merge_ev_layers.load_boundary_and_transform') as mock_load_boundary, \
             patch('sdm.commands.data_preparation.processing.merge_ev_layers.load_and_preprocess_dataset') as mock_load, \
             patch('sdm.commands.data_preparation.processing.merge_ev_layers.reproject_dataset') as mock_reproject, \
             patch('sdm.commands.data_preparation.processing.merge_ev_layers.merge_datasets') as mock_merge, \
             patch('sdm.commands.data_preparation.processing.merge_ev_layers.clip_and_save_dataset') as mock_clip_save:
            
            # Mock boundary loading
            mock_boundary_gdf = Mock()
            mock_boundary_gdf.crs = "EPSG:27700"
            mock_transform = Mock()
            mock_spatial_config = {"resolution": 100}
            mock_load_boundary.return_value = (mock_boundary_gdf, mock_transform, None, mock_spatial_config)
            
            # Mock other functions
            mock_load.return_value = xr.Dataset({'test': xr.DataArray(np.zeros((100, 100)))})
            mock_reproject.return_value = xr.Dataset({'test': xr.DataArray(np.zeros((100, 100)))})
            mock_merge.return_value = xr.Dataset({'test': xr.DataArray(np.zeros((100, 100)))})
            mock_clip_save.return_value = None
            
            # Run the function
            merge_ev_layers(
                dataset_inputs=[f"test_data={raster_file}"],
                output_path=output_path,
                boundary_path=boundary_file,
                verbose=False
            )
            
            # Verify the workflow was called correctly
            mock_load_boundary.assert_called_once()
            mock_load.assert_called_once()
            mock_reproject.assert_called_once()
            mock_merge.assert_called_once()
            mock_clip_save.assert_called_once()
    
    def test_no_datasets_processed(self, tmp_path):
        """Test error handling when no datasets are successfully processed."""
        boundary_file = tmp_path / "boundary.geojson"
        boundary_file.touch()
        
        missing_file1 = tmp_path / "missing1.tif"
        missing_file2 = tmp_path / "missing2.tif"
        output_path = tmp_path / "output.tif"
        
        with patch('sdm.commands.data_preparation.processing.merge_ev_layers.load_boundary_and_transform') as mock_load_boundary, \
             patch('sdm.commands.data_preparation.processing.merge_ev_layers.load_and_preprocess_dataset') as mock_load:
            
            # Mock boundary loading
            mock_boundary_gdf = Mock()
            mock_boundary_gdf.crs = "EPSG:27700"
            mock_transform = Mock()
            mock_spatial_config = {"resolution": 100}
            mock_load_boundary.return_value = (mock_boundary_gdf, mock_transform, None, mock_spatial_config)
            
            # Mock load function to always raise FileNotFoundError
            mock_load.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(ValueError, match="No datasets were successfully processed"):
                merge_ev_layers(
                    dataset_inputs=[f"missing1={missing_file1}", f"missing2={missing_file2}"],
                    output_path=output_path,
                    boundary_path=boundary_file,
                    verbose=False
                )