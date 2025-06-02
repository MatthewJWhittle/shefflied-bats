"""
Image data loading functionality.
"""

import logging
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import xarray as xr
import rioxarray as rxr
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
from shapely.geometry import box

logger = logging.getLogger(__name__)

class ImageTileDownloader:
    """A class to handle downloading and processing image tiles."""
    
    def __init__(self, base_url: str, cache_folder: str = "tile_cache"):
        self.base_url = base_url
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)
        self.chunk_size = 2000

    async def fetch_tile(self, session, url: str, tile_hash: str) -> Optional[str]:
        """Fetch a single tile and cache it."""
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
                    logger.warning(f"Retrying {url} - attempt {retries}")

        return cache_path if success else None

    async def fetch_tiles(self, tile_urls: List[Tuple[str, str]]) -> List[str]:
        """Fetch multiple tiles asynchronously."""
        tasks = []
        async with aiohttp.ClientSession() as session:
            for url, tile_hash in tile_urls:
                task = self.fetch_tile(session, url, tile_hash)
                tasks.append(task)

            pbar = tqdm(total=len(tasks), desc="Downloading tiles", dynamic_ncols=True)
            results = []
            for f in asyncio.as_completed(tasks):
                result = await f
                pbar.update(1)
                if result is not None:
                    results.append(result)
            pbar.close()
            return results

    def get_tile_urls(self, polygon, target_resolution: float) -> List[Tuple[str, str]]:
        """Generate tile URLs for a given polygon and resolution."""
        bounding_box = polygon.bounds
        minx, miny, maxx, maxy = bounding_box
        width = maxx - minx
        height = maxy - miny
        tile_step = int(self.chunk_size * target_resolution)

        tile_urls = []
        for x in range(int(minx), int(maxx), tile_step):
            for y in range(int(miny), int(maxy), tile_step):
                tile_bbox = box(x, y, x + tile_step, y + tile_step)
                params = {
                    "bbox": f"{tile_bbox.bounds[0]},{tile_bbox.bounds[1]},{tile_bbox.bounds[2]},{tile_bbox.bounds[3]}",
                    "size": f"{self.chunk_size},{self.chunk_size}",
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

    def download_image(self, polygon, target_resolution: float) -> xr.DataArray:
        """Download and process an image for a given polygon."""
        tile_urls = self.get_tile_urls(polygon, target_resolution)
        downloaded_files = asyncio.run(self.fetch_tiles(tile_urls))
        downloaded_files = [f for f in downloaded_files if Path(f).exists()]

        image = xr.open_mfdataset(
            downloaded_files,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
            engine="rasterio",
        )

        image = image.squeeze()
        image = image.drop("band")
        image_array = image.band_data
        image_array = image_array.rename(self.base_url.split("/")[-2])
        self.image = image_array

        return self.image.copy()

    def clear_cache(self):
        """Clear the tile cache."""
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file)) 