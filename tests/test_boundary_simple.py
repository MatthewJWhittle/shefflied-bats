"""
Simplified tests for boundary functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import Polygon

from sdm.data.spatial import create_boundary


class TestCreateBoundary:
    """Test the simplified create_boundary function."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when counties file doesn't exist."""
        non_existent_file = Path("non_existent.geojson")
        
        with pytest.raises(FileNotFoundError, match="Counties file not found"):
            create_boundary(
                counties_file=non_existent_file,
                county_names=["Test County"]
            )
    
    def test_yorkshire_default(self):
        """Test that Yorkshire boundary is created by default."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame({
            'CTYUA23NM': ["Sheffield", "Barnsley", "Leeds", "York"],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
                Polygon([(6, 6), (7, 6), (7, 7), (6, 7)])
            ]
        }, crs="EPSG:4326")
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('geopandas.read_file', return_value=mock_gdf):
                result = create_boundary(counties_file=mock_file)
        
        assert len(result) == 1  # Dissolved into single boundary
        assert 'geometry' in result.columns
        assert result.crs == "EPSG:27700"
    
    def test_custom_counties(self):
        """Test boundary creation with custom counties."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame({
            'CTYUA23NM': ["County1", "County2", "County3"],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
            ]
        }, crs="EPSG:4326")
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('geopandas.read_file', return_value=mock_gdf):
                result = create_boundary(
                    counties_file=mock_file,
                    county_names=["County1", "County2"]
                )
        
        assert len(result) == 1  # Dissolved into single boundary
        assert result.crs == "EPSG:27700"
    
    def test_no_matching_counties(self):
        """Test that ValueError is raised when no counties match."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame({
            'CTYUA23NM': ['County1', 'County2'],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
            ]
        })
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('geopandas.read_file', return_value=mock_gdf):
                with pytest.raises(ValueError, match="No counties found matching"):
                    create_boundary(
                        counties_file=mock_file,
                        county_names=["NonExistentCounty"]
                    )
