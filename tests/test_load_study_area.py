from generate_evs.ingestion.load_study_area import load_study_area
import pytest
import geopandas as gpd

def test_load_study_area():
    study_area = load_study_area()

    assert isinstance(study_area, gpd.GeoDataFrame)
    assert len(study_area) > 0

