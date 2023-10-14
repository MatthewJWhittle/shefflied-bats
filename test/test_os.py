import pytest
from sdm.data.os import load_os_shps
import geopandas as gpd

def test_load_os_data():
    datasets = ["Ornament", "Glasshouse"]
    os_data = load_os_shps(datasets=datasets, dir = "data/raw/big-files/os-vector-map")
    assert isinstance(os_data, dict)
    assert list(os_data.keys()) == datasets
    gdf = os_data[datasets[0]]
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0