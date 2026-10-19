"""
Tests for merge EV layers functionality.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, Mock

from sdm.commands.data_preparation.processing.merge_ev_layers import (
    build_ev_dataset_inputs,
    parse_dataset_input,
    merge_ev_layers
)


class TestBuildEvDatasetInputs:
    """Test merge input paths for the sdm data pipeline."""

    def test_build_ev_dataset_inputs_use_files_not_directories(self):
        evs_dir = Path("data/evs")
        inputs = build_ev_dataset_inputs(evs_dir, resolution=100)
        parsed = parse_dataset_input(inputs)

        assert len(parsed) == 9
        assert "ceh_landcover" in parsed
        assert "bgs_coast" in parsed
        assert "terrain" in parsed
        assert "climate_stats" in parsed
        assert "climate_bioclim" in parsed
        assert parsed["vom_vegetation_height"] == (
            evs_dir / "vom" / "vom_summary_metrics_100m.tif"
        )
        assert parsed["terrain"] == evs_dir / "terrain" / "dtm_dsm_100m.tif"

    def test_build_ev_dataset_inputs_skip_ceh_and_coastal(self):
        evs_dir = Path("data/evs")
        inputs = build_ev_dataset_inputs(
            evs_dir, resolution=100, skip_ceh=True, skip_coastal=True
        )
        parsed = parse_dataset_input(inputs)
        assert len(parsed) == 7
        assert "ceh_landcover" not in parsed
        assert "bgs_coast" not in parsed
        assert "landcover" not in parsed
        assert "coastal" not in parsed
        for name, path in parsed.items():
            assert path.suffix == ".tif", f"{name} should point to a GeoTIFF, got {path}"

        assert parsed["vom_vegetation_height"] == (
            evs_dir / "vom" / "vom_summary_metrics_100m.tif"
        )
        assert parsed["climate_bioclim"] == evs_dir / "climate" / "bio.tif"


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

class TestMergeGridMatchesRelease:
    """Shared grid construction must match the release EV snap (7 km buffer)."""

    def test_construct_transform_round_aligns_partial_min_edges(self):
        """Mins just shy of a cell line must not floor-expand SW by one cell.

        Synthetic study-area box: remainders on min edges are > res/2, so
        floor-align adds a west column and shifts a row south; release snaps
        each edge to the nearest resolution multiple.
        """
        from sdm.raster.utils import construct_transform_shift_bounds

        resolution = 100.0
        # Remainders: xmin≈96, ymin≈56, xmax≈10, ymax≈10 (res=100).
        bounds = (196.02, 156.44, 410.10, 510.10)
        transform, snapped = construct_transform_shift_bounds(bounds, resolution)
        expected = (200.0, 200.0, 400.0, 500.0)
        assert snapped == expected, (
            f"expected round-aligned {expected}, got {snapped} "
            f"(transform={transform})"
        )
        width = int(round((snapped[2] - snapped[0]) / resolution))
        height = int(round((snapped[3] - snapped[1]) / resolution))
        assert (width, height) == (2, 3)

        # Already-aligned edges must stay put (no ceil-style extra cell).
        aligned = (200.0, 200.0, 400.0, 500.0)
        _, snapped_aligned = construct_transform_shift_bounds(aligned, resolution)
        assert snapped_aligned == aligned

    def test_buffered_study_area_shape_matches_round_snap(self):
        """Study area + 7 km buffer must snap to round-aligned shape/transform."""
        import geopandas as gpd
        from shapely.geometry import box
        from sdm.raster.utils import construct_transform_shift_bounds

        # Synthetic study box (not a real site): unaligned after 7 km buffer.
        study = gpd.GeoDataFrame(
            geometry=[box(1103.98, 2203.56, 2896.02, 3796.44)],
            crs="EPSG:27700",
        )
        buffered = study.copy()
        buffered["geometry"] = buffered.geometry.buffer(7000)
        raw = tuple(buffered.total_bounds)
        resolution = 100.0
        _, snapped = construct_transform_shift_bounds(raw, resolution)
        expected = tuple(round(v / resolution) * resolution for v in raw)
        assert snapped == expected
        # Floor-align of mins would differ on this box (SW expand).
        floored_xmin = raw[0] - (raw[0] % resolution)
        floored_ymin = raw[1] - (raw[1] % resolution)
        assert snapped[0] != floored_xmin or snapped[1] != floored_ymin

    def test_rebuild_passes_study_area_buffer_to_merge(self, tmp_path, monkeypatch):
        """Merge inside sdm data must use the 7 km study-area buffer, not 0."""
        from sdm.commands.data_preparation import rebuild_ev_layers as mod

        boundary = tmp_path / "boundary.geojson"
        boundary.write_text(
            '{"type":"FeatureCollection","features":[]}',
            encoding="utf-8",
        )
        out = tmp_path / "evs-rebuild"
        captured: dict = {}

        def _merge(*_args, **kwargs):
            captured.update(kwargs)

        def _ok(*_args, **_kwargs):
            return out

        monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
        for name in (
            "generate_terrain_data",
            "generate_terrain_stats",
            "generate_climate_data",
            "generate_ceh_lc_data",
            "generate_vom_data",
            "generate_coastal_distance",
            "process_os_data",
        ):
            monkeypatch.setattr(mod, name, _ok)
        monkeypatch.setattr(mod, "merge_ev_layers", _merge)
        monkeypatch.setattr(mod, "build_ev_dataset_inputs", lambda *a, **k: ["x=y.tif"])
        stub_settings = {
            "layer_clipping": {"enabled": True, "layers": {}, "applied": {}}
        }
        monkeypatch.setattr(
            mod,
            "build_model_ev_stack",
            lambda *_a, **_k: (out / "evs-to-model.tif", stub_settings),
        )
        monkeypatch.setattr(
            mod, "build_settings_from_config", lambda *_a, **_k: stub_settings
        )
        monkeypatch.setattr(mod, "load_config", lambda: {})
        monkeypatch.setattr(mod, "collect_rebuild_provenance", lambda **_k: {})

        mod.rebuild_environmental_layers(
            out,
            boundary_path=boundary,
            resolution=100,
            buffer_distance_m=7000,
            only=["merge"],
        )
        assert captured.get("boundary_buffer_m") == 7000
