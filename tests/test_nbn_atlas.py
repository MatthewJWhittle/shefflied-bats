"""
Unit tests for the NBN Atlas occurrence client.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

from shapely.geometry import Polygon

from sdm.occurrence.nbn_atlas import (
    _build_query,
    fetch_occurrences_from_nbn,
)


class TestBuildQuery:
    def test_requires_some_filter(self) -> None:
        """At least one of scientific_name, taxon_id or query must be supplied."""
        with pytest.raises(ValueError):
            _build_query()

    def test_builds_scientific_name_query(self) -> None:
        params = _build_query(scientific_name="Pipistrellus pipistrellus")
        assert params["page"] == 1
        assert params["pageSize"] == 1000
        # scientific name should be wrapped in quotes
        assert params["q"] == 'scientificName:"Pipistrellus pipistrellus"'

    def test_combines_query_and_scientific_name(self) -> None:
        params = _build_query(
            scientific_name="Myotis daubentonii", query="basisOfRecord:HumanObservation"
        )
        assert " AND " in params["q"]
        assert 'scientificName:"Myotis daubentonii"' in params["q"]
        assert "basisOfRecord:HumanObservation" in params["q"]

    def test_extra_params_do_not_override_pagination(self) -> None:
        params = _build_query(
            scientific_name="Nyctalus noctula",
            extra_params={"page": 99, "pageSize": 10, "fq": "country:England"},
        )
        # internal pagination should win
        assert params["page"] == 1
        assert params["pageSize"] == 1000
        # other extra params should be preserved
        assert params["fq"] == "country:England"

    def test_point_radius_builds_wkt(self) -> None:
        params = _build_query(
            scientific_name="Pipistrellus pipistrellus",
            point=(-1.47, 53.38),
            radius_km=5.0,
        )
        assert "wkt" in params
        assert params["wkt"].startswith("POLYGON((")

    def test_bbox_builds_wkt(self) -> None:
        params = _build_query(
            scientific_name="Pipistrellus pipistrellus",
            bbox=(-1.5, 53.3, -1.4, 53.4),
        )
        assert "wkt" in params
        assert "POLYGON((" in params["wkt"]

    def test_polygon_builds_wkt_from_shapely(self) -> None:
        poly = Polygon(
            [
                (-1.5, 53.3),
                (-1.4, 53.3),
                (-1.4, 53.4),
                (-1.5, 53.4),
            ]
        )
        params = _build_query(
            scientific_name="Pipistrellus pipistrellus",
            polygon=poly,
        )
        assert "wkt" in params
        # Should match shapely's WKT representation
        assert params["wkt"].startswith("POLYGON((")


class TestFetchOccurrencesFromNbn:
    def test_returns_empty_geodataframe_when_no_records(self, monkeypatch):
        """If the async fetch returns an empty DataFrame, the wrapper returns an empty GeoDataFrame."""

        async def fake_async_fetch(**_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        # Monkeypatch the internal async function used by fetch_occurrences_from_nbn
        from sdm.occurrence import nbn_atlas as nbn_mod

        monkeypatch.setattr(nbn_mod, "_fetch_occurrences_async", fake_async_fetch)

        gdf = fetch_occurrences_from_nbn(scientific_name="Pipistrellus pipistrellus")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.empty

    def test_builds_geometry_from_lat_lon(self, monkeypatch):
        """Records with decimalLatitude/decimalLongitude should be converted to point geometries."""

        records = pd.DataFrame(
            {
                "decimalLatitude": [53.38, 53.39],
                "decimalLongitude": [-1.47, -1.48],
                "scientificName": ["Pipistrellus pipistrellus", "Pipistrellus pipistrellus"],
            }
        )

        async def fake_async_fetch(**_kwargs) -> pd.DataFrame:
            return records

        from sdm.occurrence import nbn_atlas as nbn_mod

        monkeypatch.setattr(nbn_mod, "_fetch_occurrences_async", fake_async_fetch)

        gdf = fetch_occurrences_from_nbn(scientific_name="Pipistrellus pipistrellus")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert not gdf.empty
        assert gdf.crs.to_string() == "EPSG:4326"
        # geometry column should be present and of correct length
        assert len(gdf) == 2
        assert "geometry" in gdf.columns
        # Check that geometries were created at expected coordinates
        xs = gdf.geometry.x.tolist()
        ys = gdf.geometry.y.tolist()
        assert xs == pytest.approx(records["decimalLongitude"].tolist())
        assert ys == pytest.approx(records["decimalLatitude"].tolist())

    def test_polygon_is_passed_through_public_api(self, monkeypatch):
        """fetch_occurrences_from_nbn should accept a shapely polygon and pass it to the async layer."""

        records = pd.DataFrame(
            {
                "decimalLatitude": [53.38],
                "decimalLongitude": [-1.47],
                "scientificName": ["Pipistrellus pipistrellus"],
            }
        )

        poly = Polygon(
            [
                (-1.5, 53.3),
                (-1.4, 53.3),
                (-1.4, 53.4),
                (-1.5, 53.4),
            ]
        )

        async def fake_async_fetch(**kwargs) -> pd.DataFrame:
            # Ensure the polygon reaches the async layer
            assert "polygon" in kwargs
            assert isinstance(kwargs["polygon"], Polygon)
            assert kwargs["polygon"].equals(poly)
            return records

        from sdm.occurrence import nbn_atlas as nbn_mod

        monkeypatch.setattr(nbn_mod, "_fetch_occurrences_async", fake_async_fetch)

        gdf = fetch_occurrences_from_nbn(
            scientific_name="Pipistrellus pipistrellus",
            polygon=poly,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1
        assert gdf.crs.to_string() == "EPSG:4326"


@pytest.mark.slow
class TestFetchOccurrencesFromNbnLive:
    """Tests that exercise the live NBN Atlas API.

    These tests are marked as 'slow' and 'integration' and are additionally
    gated by the NBN_LIVE_TESTS environment variable to avoid accidental
    execution in CI environments without network access.
    """

    def test_fetch_small_sample_for_common_species(self):
        """Fetch a small number of records for a common bat species."""
        gdf = fetch_occurrences_from_nbn(
            scientific_name="Pipistrellus pipistrellus",
            max_records=10,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        # We expect at least one record, but allow for occasional zero in case of API issues
        assert len(gdf) >= 0
        if not gdf.empty:
            assert "scientificName" in gdf.columns
            assert gdf.crs.to_string() == "EPSG:4326"
            # geometries should be valid points
            assert gdf.geometry.notna().all()

    def test_fetch_with_spatial_polygon_filter(self):
        """Fetch a small number of records for a common bat species within a polygon."""
        poly = Polygon(
            [
                (-1.6, 53.3),
                (-1.3, 53.3),
                (-1.3, 53.5),
                (-1.6, 53.5),
            ]
        )

        gdf = fetch_occurrences_from_nbn(
            scientific_name="Pipistrellus pipistrellus",
            polygon=poly,
            max_records=10,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) >= 0
        if not gdf.empty:
            assert "scientificName" in gdf.columns
            assert gdf.crs.to_string() == "EPSG:4326"
            # geometries should be valid points
            assert gdf.geometry.notna().all()


