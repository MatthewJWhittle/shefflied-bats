from sre_constants import SUCCESS
import stat
import requests
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from tqdm.asyncio import tqdm

from sdm import data


# A functino to download a layer from an esri feature service and use a query to filter the data for certain features
def download_feature_layer(url, query="1=1"):
    # Construct the query parameters
    query_params = {"where": query, "outFields": "*", "f": "json"}
    # Make the request
    r = requests.get(url, params=query_params)
    # Check the status code
    if r.status_code == 200:
        # Get the response json
        response = r.json()

        gdf = gpd.read_file(response)

        return gdf


import aiohttp
import asyncio
import os
import hashlib
import geopandas as gpd
from shapely.geometry import box


class ImageTileDownloader:
    def __init__(self, base_url, cache_folder="tile_cache"):
        self.base_url = base_url
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)
        self.chunk_size = 2000

    async def fetch_tile(self, session, url, tile_hash):
        cache_path = os.path.join(self.cache_folder, f"{tile_hash}.tif")

        if os.path.exists(cache_path):
            return cache_path

        retries = 0
        success = False
        status = None
        while retries < 3 and not success and status != 200:
            async with session.get(url) as response:
                tile_data = await response.read()
                with open(cache_path, "wb") as f:
                    f.write(tile_data)
                try:
                    rxr.open_rasterio(cache_path)
                    success = True
                    status = response.status
                except:
                    retries += 1
                    os.remove(cache_path)
                    print(f"Retrying {url} - attempt {retries}")

            return cache_path

    async def fetch_tiles(self, tile_urls):
        tasks = []
        async with aiohttp.ClientSession() as session:
            # Create all tasks
            for url, tile_hash in tile_urls:
                task = self.fetch_tile(session, url, tile_hash)
                tasks.append(task)

            # Create an asynchronous progress bar
            pbar = tqdm(total=len(tasks), desc="Downloading tiles", dynamic_ncols=True)

            # Wait for tasks to complete and update the progress bar
            results = []
            for f in asyncio.as_completed(tasks):
                result = await f
                pbar.update(1)
                results.append(result)

            pbar.close()
            return results

    def get_tile_urls(self, polygon, target_resolution):
        # Convert polygon to bounding box
        bounding_box = polygon.bounds

        # Tile the bounding box into smaller boxes
        minx, miny, maxx, maxy = bounding_box
        width = maxx - minx
        height = maxy - miny
        tile_step = (
            self.chunk_size * target_resolution
        )  # This works out the size of the bounding box tile to achieve the specified resolution

        tile_urls = []
        for x in range(int(minx), int(maxx), tile_step):
            for y in range(int(miny), int(maxy), tile_step):
                tile_bbox = box(x, y, x + tile_step, y + tile_step)

                # Calculate parameters based on target resolution
                params = {
                    "bbox": f"{tile_bbox.bounds[0]},{tile_bbox.bounds[1]},{tile_bbox.bounds[2]},{tile_bbox.bounds[3]}",  # This is the AOI
                    "size": f"{self.chunk_size},{self.chunk_size}",  # This is how many pixels are in the tile
                    "format": "tiff",
                    "f": "image",
                    "imageSR": 27700,
                    "noData": -9999,
                }
                tile_param = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"{self.base_url}/exportImage?{tile_param}"
                tile_hash = hashlib.md5(url.encode()).hexdigest()
                tile_urls.append((url, tile_hash))

        return tile_urls

    def download_image(self, polygon, target_resolution):
        tile_urls = self.get_tile_urls(polygon, target_resolution)
        downloaded_files = asyncio.run(self.fetch_tiles(tile_urls))

        # Create a rioxarray multi-file dataset
        image = xr.open_mfdataset(
            downloaded_files,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
            engine="rasterio",
        )

        # Tidy up the image to just return an array
        image = image.squeeze()
        # Drop the band coordinate
        image = image.drop("band")

        # Get the array
        image_array = image.band_data

        # Name the image using the image server name from the url
        image_array = image_array.rename(self.base_url.split("/")[-2])

        # Add the image as an property of the class
        self.image = image_array

        # Return a copy
        return self.image.copy()

    def clear_cache(self):
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))


from pathlib import Path
import rioxarray as rxr


class ClimateData:
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
    return long_name.replace("wc2.1_30s_", "").replace(" ", "_").lower()


def ceh_lc_types():
    return {
        "1": "Broadleaved woodland",
        "2": "Coniferous woodland",
        "3": "Arable",
        "4": "Improved grassland",
        "5": "Neutral grassland",
        "6": "Calcareous grassland",
        "7": "Acid grassland",
        "8": "Fen, Marsh and Swamp",
        "9": "Heather and shrub",
        "10": "Heather grassland",
        "11": "Bog",
        "12": "Inland rock",
        "13": "Saltwater",
        "14": "Freshwater",
        "15": "Supralittoral rock",
        "16": "Supralittoral sediment",
        "17": "Littoral rock",
        "18": "Littoral sediment",
        "19": "Saltmarsh",
        "20": "Urban",
        "21": "Suburban",
    }
