"""
Climate data loading functionality.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

import xarray as xr
import rioxarray as rxr
import geopandas as gpd

logger = logging.getLogger(__name__)

class ClimateData:
    """
    A class to handle downloading and processing climate data.
    """
    def __init__(self, cache_folder: Union[str, Path] = "data/raw/big-files/climate_cache"):
        self.base_url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/GBR_wc2.1_30s_{var}.tif"
        self.datasets = {
            "tmin": "Minimum temperature",
            "tmax": "Maximum temperature",
            "tavg": "Average temperature",
            "prec": "Precipitation",
            "bio": "Bioclimatic variables",
            "wind": "Wind speed",
            "srad": "Solar radiation",
        }
        self.cache_folder = Path(cache_folder)
        self._downloaded_datasets = set()
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    def _url(self, variable: str) -> str:
        return self.base_url.format(var=variable)

    def _local_path(self, variable: str) -> Path:
        return self.cache_folder / f"{variable}.tif"

    def download_dataset(self, variable: str) -> Path:
        cache_path = self._local_path(variable)
        if not cache_path.exists():
            data = rxr.open_rasterio(self._url(variable))
            if isinstance(data, xr.DataArray):
                data.rio.to_raster(cache_path)
            else:
                raise ValueError(f"Expected DataArray from {self._url(variable)}, got {type(data)}")
            self._downloaded_datasets.add(variable)
        return cache_path

    def get_dataset(self, variable: str, aoi: Optional[gpd.GeoDataFrame] = None) -> xr.DataArray:
        """Get climate dataset, optionally clipped to area of interest."""
        cache_path = self.download_dataset(variable)
        data = rxr.open_rasterio(cache_path)
        
        if not isinstance(data, xr.DataArray):
            raise ValueError(f"Expected DataArray from {cache_path}, got {type(data)}")
        
        if aoi is not None:
            data = data.rio.clip(aoi.geometry, aoi.crs)
            
        return data 