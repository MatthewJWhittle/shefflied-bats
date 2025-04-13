from pathlib import Path
from typing import Union, Dict

import geopandas as gpd
import rioxarray as rxr
import numpy as np
from rasterio.enums import Resampling
import xarray as xr

from data_prep.utils.load import load_boundary, load_spatial_config, construct_transform_shift_bounds
from data_prep.generate_evs.ingestion.geo_utils import reproject_data




class ClimateData:
    """
    A class to handle downloading and processing climate data.

    Attributes:
        base_url (str): The base URL for downloading climate data.
        datasets (dict): A dictionary mapping climate variables to their descriptions.
        cache_folder (Path): The folder where downloaded datasets are cached.
        _downloaded_datasets (set): A set of downloaded datasets.

    Methods:
        __init__(cache_folder="data/raw/big-files/climate_cache"):
            Initializes the ClimateData object with a cache folder.
        
        _url(variable) -> str:
            Constructs the URL for a given climate variable.
        
        _local_path(variable) -> Path:
            Constructs the local cache path for a given climate variable.
        
        download_dataset(variable):
            Downloads a climate variable dataset and caches it.
        
        get_dataset(variable, aoi: gpd.GeoDataFrame):
            Retrieves a dataset. If an area of interest (AOI) is provided, the dataset is clipped to the AOI.
        
        _set_band_names(data):
            Sets the band names of the dataset based on the long name attribute.
    """
    def __init__(self, cache_folder="data/raw/big-files/climate_cache"):
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

    def _url(self, variable) -> str:
        """Get the URL for a given climate variable."""
        return self.base_url.format(var=variable)

    def _local_path(self, variable) -> Path:
        """Get the local cache path for a given climate variable."""
        return self.cache_folder / f"{variable}.tif"

    def download_dataset(self, variable):
        """Download a climate variable dataset and cache it."""
        cache_path = self._local_path(variable)
        if not cache_path.exists():
            rxr.open_rasterio(self._url(variable)).rio.to_raster(cache_path)
            self._downloaded_datasets.add(variable)
        return cache_path

    def get_dataset(self, variable, aoi: gpd.GeoDataFrame):
        """Retrieve a dataset. If bbox is provided, the dataset is clipped to the bbox."""

        assert (
            variable in self.datasets.keys()
        ), f"Variable must be one of {self.datasets.keys()}"

        if variable not in self._downloaded_datasets:
            self.download_dataset(variable)

        data = rxr.open_rasterio(self._local_path(variable))

        if aoi is not None:
            # transform aoi to 27700
            aoi = aoi.to_crs(27700)
            # add a small buffer to the aoi
            # this ensures that when projecting the aoi to the data crs, the aoi is still within the data
            aoi["geometry"] = aoi.geometry.buffer(1000)

            # transform the bbox to the same crs as the data
            aoi.to_crs(data.rio.crs, inplace=True)

            # Get the bounding box of the AOI
            aoi_bbox = tuple(aoi.total_bounds)
            # Clip the data
            data = data.rio.clip_box(*aoi_bbox)

        # Write the nodata value
        data.rio.write_nodata(data.attrs["_FillValue"], inplace=True)

        return data

    def _set_band_names(self, data):
        dataset = data.to_dataset(dim="band")
        # Get the long name attribute and use it to rename the bands
        var_names = data.attrs["long_name"]
        # Tidy up the name
        var_names = [tidy_long_name(var_name) for var_name in var_names]
        dataset = dataset.rename(dict(zip(dataset.data_vars, var_names)))
        return dataset


def tidy_long_name(long_name):
    """
    Cleans and formats a given long name string by removing a specific prefix,
    replacing spaces with underscores, and converting the string to lowercase.

    Args:
        long_name (str): The original long name string to be tidied.

    Returns:
        str: The tidied and formatted string.
    """
    return long_name.replace("wc2.1_30s_", "").replace(" ", "_").lower()




def get_climate_data(boundary):
    """
    Fetches various climate data for a given geographical boundary.

    Parameters:
    boundary (object): The area of interest (AOI) for which the climate data is to be fetched.

    Returns:
    tuple: A tuple containing the following climate datasets:
        - bioclim: Bioclimatic variables dataset.
        - temp_average: Average temperature dataset.
        - precipitation: Precipitation dataset.
        - wind: Wind dataset.
    """
    climate_data = ClimateData()
    bioclim = climate_data.get_dataset(variable="bio", aoi=boundary)
    temp_average = climate_data.get_dataset(variable="tavg", aoi=boundary)
    precipitation = climate_data.get_dataset(variable="prec", aoi=boundary)
    wind = climate_data.get_dataset(variable="wind", aoi=boundary)
    return bioclim, temp_average, precipitation, wind




def reproject_all_datasets(bioclim, temp_average, precipitation, wind, crs, transform, resolution):
    """
    Reprojects multiple climate datasets to a specified coordinate reference system (CRS) and transform.

    Parameters:
    bioclim (array-like): The bioclimatic dataset to be reprojected.
    temp_average (array-like): The temperature average dataset to be reprojected.
    precipitation (array-like): The precipitation dataset to be reprojected.
    wind (array-like): The wind dataset to be reprojected.
    crs (str or dict): The target coordinate reference system.
    transform (Affine): The affine transformation to apply.

    Returns:
    tuple: A tuple containing the reprojected bioclim, temp_average, precipitation, and wind datasets.
    """
    bioclim = reproject_data(bioclim, crs, transform, resolution)
    temp_average = reproject_data(temp_average, crs, transform, resolution)
    precipitation = reproject_data(precipitation, crs, transform, resolution)
    wind = reproject_data(wind, crs, transform, resolution)
    return bioclim, temp_average, precipitation, wind



def assign_variable_names(temp_average, precipitation, bioclim, wind):
    """
    Assigns long_name attributes to the given climate data variables.

    Parameters:
    temp_average (xarray.DataArray): The temperature average data array.
    precipitation (xarray.DataArray): The precipitation data array.
    bioclim (xarray.DataArray): The bioclimatic data array.
    wind (xarray.DataArray): The wind data array.

    Each input data array will have its 'long_name' attribute set to a tuple of strings,
    where each string is formatted as 'variable_name_i' with 'i' being the index of the
    corresponding element in the data array.
    """
    temp_average.attrs["long_name"] = tuple(
        [f"temp_{i}" for i in range(1, temp_average.shape[0] + 1)]
    )
    precipitation.attrs["long_name"] = tuple(
        [f"prec_{i}" for i in range(1, precipitation.shape[0] + 1)]
    )
    bioclim.attrs["long_name"] = tuple(
        [f"bio_{i}" for i in range(1, bioclim.shape[0] + 1)]
    )
    wind.attrs["long_name"] = tuple([f"wind_{i}" for i in range(1, wind.shape[0] + 1)])



def write_data(temp_average, precipitation, bioclim, wind, output_dir) -> Dict[str, Path]:
    """
    Writes climate data to raster files in the specified output directory.

    Parameters:
    temp_average (xarray.DataArray): The average temperature data to be written to a raster file.
    precipitation (xarray.DataArray): The precipitation data to be written to a raster file.
    bioclim (xarray.DataArray): The bioclimatic data to be written to a raster file.
    wind (xarray.DataArray): The wind data to be written to a raster file.
    output_dir (str or Path): The directory where the raster files will be saved. If the directory does not exist, it will be created.

    Returns:
    None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Temperature
    temp_path = output_dir / "temp_average.tif"
    temp_average.rio.to_raster(temp_path)

    # Precipitation variables
    precipitation_path = output_dir / "precipitation.tif"
    precipitation.rio.to_raster(precipitation_path)

    # Bioclimatic variables
    bioclim_path = output_dir / "bioclim.tif"
    bioclim.rio.to_raster(bioclim_path)
    
    # Wind variables
    wind_path = output_dir / "wind.tif"
    wind.rio.to_raster(wind_path)

    return {
        "temp_average": temp_path,
        "precipitation": precipitation_path,
        "bioclim": bioclim_path,
        "wind": wind_path,
    }


def calculate_climate_stats(temp_average, precipitation, wind, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    climate_stats = xr.zeros_like(temp_average[0])
    climate_stats = climate_stats.to_dataset(name="zeros")

    climate_stats["temp_ann_var"] = temp_average.std(axis=0)
    climate_stats["temp_ann_avg"] = temp_average.mean(axis=0)
    climate_stats["temp_mat_avg"] = temp_average[3:6].mean(axis=0)

    climate_stats["prec_ann_var"] = precipitation.std(axis=0)
    climate_stats["prec_ann_avg"] = precipitation.mean(axis=0)

    climate_stats["wind_ann_var"] = wind.std(axis=0)
    climate_stats["wind_ann_avg"] = wind.mean(axis=0)

    climate_stats = climate_stats.drop_vars("zeros")
    path = output_dir / "climate_stats.tif"
    climate_stats.rio.to_raster(path)
    return path


def main(
        output_dir: Union[str, Path] = "data/evs",
        boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
) -> Dict[str, Path]:
    """
    Main function to process climate data based on a given boundary.
    Parameters:
    boundary_path (Union[str, Path]): Path to the boundary GeoJSON file. Default is "data/processed/boundary.geojson".
    output_dir (Union[str, Path]): Directory where the output data will be saved. Default is "data/evs".
    This function performs the following steps:
    1. Loads the boundary from the specified path.
    2. Loads spatial configuration settings.
    3. Transforms the boundary to the specified coordinate reference system (CRS).
    4. Constructs a model transform based on the boundary's total bounds and spatial resolution.
    5. Retrieves climate data (bioclimatic variables, temperature average, precipitation, and wind) for the boundary.
    6. Reprojects all datasets to the specified CRS and transform.
    7. Assigns variable names to the datasets.
    8. Writes the processed data to the specified output directory.
    9. Calculates and saves climate statistics based on the processed data.
    """
    boundary = load_boundary(boundary_path)
    spatial_config = load_spatial_config()
    boundary = boundary.to_crs(spatial_config["crs"])
    model_transform, _ = construct_transform_shift_bounds(tuple(boundary.total_bounds), spatial_config["resolution"])

    bioclim, temp_average, precipitation, wind = get_climate_data(boundary)

    bioclim, temp_average, precipitation, wind = reproject_all_datasets(
        bioclim, temp_average, precipitation, wind, crs=27700, transform=model_transform, resolution=spatial_config["resolution"]
    )
    assign_variable_names(temp_average, precipitation, bioclim, wind)
    
    variable_paths = write_data(temp_average, precipitation, bioclim, wind, output_dir)


    stats_path = calculate_climate_stats(temp_average, precipitation, wind, output_dir)

    variable_paths["climate_stats"] = stats_path

    return variable_paths


if __name__ == "__main__":
    main()
