"""
Simplified tests for boundary functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import Polygon, box

from sdm.data.spatial import create_boundary


class TestCreateBoundary:
    """Test the simplified create_boundary function."""

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when counties file doesn't exist."""
        non_existent_file = Path("non_existent.geojson")

        with patch("sdm.data.spatial.core.resolve_counties_file") as resolve_mock:
            resolve_mock.side_effect = FileNotFoundError(
                "Counties file not found: non_existent.geojson"
            )
            with pytest.raises(FileNotFoundError, match="Counties file not found"):
                create_boundary(
                    counties_file=non_existent_file,
                    county_names=["Test County"],
                    live_download=False,
                )

    def test_yorkshire_default(self):
        """Test that Yorkshire boundary is created by default."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame(
            {
                "CTYUA23NM": ["Sheffield", "Barnsley", "Leeds", "York"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                    Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
                    Polygon([(6, 6), (7, 6), (7, 7), (6, 7)]),
                ],
            },
            crs="EPSG:4326",
        )

        with patch(
            "sdm.data.spatial.core.resolve_counties_file", return_value=mock_file
        ):
            with patch("geopandas.read_file", return_value=mock_gdf):
                result = create_boundary(counties_file=mock_file, live_download=False)

        assert len(result) == 1  # Dissolved into single boundary
        assert "geometry" in result.columns
        assert result.crs == "EPSG:27700"

    def test_custom_counties(self):
        """Test boundary creation with custom counties."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame(
            {
                "CTYUA23NM": ["County1", "County2", "County3"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                    Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
                ],
            },
            crs="EPSG:4326",
        )

        with patch.object(Path, "exists", return_value=True):
            with patch("geopandas.read_file", return_value=mock_gdf):
                result = create_boundary(
                    counties_file=mock_file,
                    county_names=["County1", "County2"],
                )

        assert len(result) == 1  # Dissolved into single boundary
        assert result.crs == "EPSG:27700"

    def test_no_matching_counties(self):
        """Test that ValueError is raised when no counties match."""
        mock_file = Path("test.geojson")
        mock_gdf = gpd.GeoDataFrame(
            {
                "CTYUA23NM": ["County1", "County2"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            }
        )

        with patch(
            "sdm.data.spatial.core.resolve_counties_file", return_value=mock_file
        ):
            with patch("geopandas.read_file", return_value=mock_gdf):
                with pytest.raises(ValueError, match="No counties found matching"):
                    create_boundary(
                        counties_file=mock_file,
                        county_names=["NonExistentCounty"],
                        live_download=False,
                    )

    def test_buffer_clipped_to_land_keeps_inland_drops_sea(self):
        """Buffered extent keeps inland neighbours and drops seaward lobes (#78)."""
        mock_file = Path("test.geojson")
        # EPSG:27700 metres: study block, inland neighbour to the east, empty = sea west.
        mock_gdf = gpd.GeoDataFrame(
            {
                "CTYUA24NM": ["Study", "InlandNeighbour"],
                "geometry": [
                    box(0, 0, 10_000, 10_000),
                    box(10_000, 0, 20_000, 10_000),
                ],
            },
            crs="EPSG:27700",
        )

        with patch(
            "sdm.data.spatial.core.resolve_counties_file", return_value=mock_file
        ):
            with patch("geopandas.read_file", return_value=mock_gdf):
                result = create_boundary(
                    counties_file=mock_file,
                    county_names=["Study"],
                    target_crs="EPSG:27700",
                    simplify_tolerance=None,
                    live_download=False,
                    buffer_distance=5_000,
                )

        geom = result.geometry.union_all()
        minx, miny, maxx, maxy = geom.bounds
        # Seaward (west / south / north of land) must not keep the full buffer lobe.
        assert minx >= -1.0
        assert miny >= -1.0
        assert maxy <= 10_000 + 1.0
        # Inland buffer into the neighbouring authority is retained.
        assert maxx >= 14_000
        assert geom.intersects(box(12_000, 4_000, 13_000, 6_000))
        assert not geom.intersects(box(-4_000, 4_000, -1_000, 6_000))
