import asyncio
import math
import tempfile
import shutil
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from hashlib import sha256
from typing import Union

import aiohttp
import numpy as np
from rasterio.io import MemoryFile
import xarray as xr
# import required to enable functionality even if not used directly
import rioxarray as rxr 
import requests
from tqdm.asyncio import tqdm_asyncio
from src.ingestion.geo_utils import BoxTiler




@dataclass
class TileRequest:
    """Container for WCS tile request parameters.

    Attributes:
        bbox: Tuple of (minx, miny, maxx, maxy) coordinates.
        width: Number of pixels in x direction.
        height: Number of pixels in y direction.
    """
    bbox: Tuple[float, float, float, float]
    width: int
    height: int


class WCSDownloader:
    """A client for downloading geospatial data from WCS 2.0.1 services.

    This class handles downloading raster data from Web Coverage Service (WCS) 2.0.1
    endpoints. It supports automatic tiling of large requests and concurrent downloads
    for better performance.

    Attributes:
        endpoint: Base URL of the WCS service.
        coverage_id: Identifier for the specific coverage to download.
        fill_value: Value to use for missing or invalid data.
        axis_labels: List of axis names from the service (typically ['x', 'y']).
        native_crs: Coordinate reference system of the coverage.
    """

    def __init__(
        self,
        endpoint: str,
        coverage_id: str,
        fill_value: Optional[float] = None,
        use_temp_storage: bool = False,
        request_tile_pixels: Tuple[int, int] = (512, 512),
    ):
        """Initializes the WCS downloader.

        Args:
            endpoint: Base URL of the WCS service.
            coverage_id: Identifier for the coverage to download.
            fill_value: Value to use for missing or invalid data. Defaults to np.nan.
            use_temp_storage: If True, saves downloaded tiles to disk to reduce memory usage.
        """
        self.endpoint = endpoint
        self.coverage_id = coverage_id
        self.fill_value = fill_value if fill_value is not None else np.nan
        self.use_temp_storage = use_temp_storage
        self.axis_labels, self.native_crs = self._fetch_coverage_description()
        self.max_tile_pixels = (4096, 4096) 
        self.request_tile_pixels = request_tile_pixels
        self.origin = (0, 0)
        
        if self.use_temp_storage:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="wcs_download_"))
        else:
            self.temp_dir = None

    def __del__(self):
        """Cleanup temporary files when the object is destroyed."""
        if hasattr(self, 'temp_dir') and self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def clear_temp_storage(self):
        """Clears temporary storage directory if used."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def __repr__(self):
        return f"WCSDownloader(endpoint={self.endpoint!r}, coverage_id={self.coverage_id!r})"

    def _fetch_coverage_description(self) -> Tuple[List[str], str]:
        """
        Retrieves the coverage description (axis labels and native CRS) via DescribeCoverage.
        """
        params = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "DescribeCoverage",
            "coverageId": self.coverage_id,
        }
        resp = requests.get(self.endpoint, params=params, timeout=10)
        resp.raise_for_status()

        ns = {
            "wcs": "http://www.opengis.net/wcs/2.0",
            "gml": "http://www.opengis.net/gml/3.2",
        }
        root = ET.fromstring(resp.content)
        envelope = root.find(".//gml:Envelope", ns)
        if envelope is None:
            raise ValueError("No envelope found in DescribeCoverage response.")
        axis_str = envelope.attrib.get("axisLabels", "x y")
        srs_name = envelope.attrib.get("srsName", "")
        return axis_str.split(), srs_name



    async def _fetch_tile(
        self,
        session: aiohttp.ClientSession,
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> bytes:
        """
        Asynchronously fetch one tile via a GetCoverage request.
        """
        params: Dict[str, Any] = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "coverageId": self.coverage_id,
            "format": "image/tiff",
            "width": str(width),
            "height": str(height),
        }
        minx, miny, maxx, maxy = bbox

        # The resolution is controled by a scale factor parameter.
        # I'm calculating this here but it could be done outside of this function if 
        # we know the source resolution and the desired resolution
        bbox_width = maxx - minx
        bbox_height = maxy - miny

        x_scale = width / bbox_width
        y_scale = height / bbox_height
        scalefactor = max(x_scale, y_scale)
        # It is a float value, if source res is 1m and desired res is 10m, scalefactor = 0.1
        params["scalefactor"] = str(scalefactor)


        # axis_labels[0] is typically "x" or "long", axis_labels[1] is "y" or "lat"
        params["subset"] = [
            f"{self.axis_labels[0]}({minx},{maxx})",
            f"{self.axis_labels[1]}({miny},{maxy})",
        ]
        params["subsettingcrs"] = self.native_crs

        async with session.get(self.endpoint, params=params) as resp:
            resp.raise_for_status()
            data = await resp.read()
            # Make sure we didn't get an XML error doc or invalid TIFF
            d = data.lstrip()
            if d.startswith(b"<?xml"):
                snippet = d[:200].decode(errors="replace")
                raise ValueError(f"Received XML instead of TIFF: {snippet}")
            if not (d.startswith(b"II*\x00") or d.startswith(b"MM\x00*")):
                raise ValueError("Data does not appear to be a valid TIFF.")
            return data

    async def get_coverage(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float,
        max_concurrent: int = 10,
    ) -> xr.Dataset:
        """Downloads coverage data for a specified bounding box.

        Downloads raster data from the WCS service, automatically splitting the request
        into tiles if needed. Handles concurrent downloads and merges results.

        Args:
            bbox: Tuple of (minx, miny, maxx, maxy) coordinates defining the area.
            resolution: Pixel size in coordinate system units.
            tile_size: Optional tuple of (width, height) for tiling. If None, downloads
                in a single request.
            max_concurrent: Maximum number of simultaneous download requests.

        Returns:
            xarray.Dataset containing the merged raster data with appropriate metadata.

        Raises:
            ValueError: If bbox is invalid, resolution is negative, or tile_size is invalid.
            requests.RequestException: If the WCS service request fails.
        """
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("bbox must be a tuple (minx, miny, maxx, maxy).")
        if resolution <= 0:
            raise ValueError("resolution must be positive")


        # Clear temp files if using temp storage
        if self.use_temp_storage and self.temp_dir:
            for f in self.temp_dir.glob("*.tif"):
                f.unlink()
        # TODO: Rethink and fix this logic
        # Where the user only wants a small area, this is causing usto download very large tiles. 
        # the tile size should only be utilised when the user requests an area that is larger than the tile size
        # otherwise we should just download the area requested.
        tile_width_crs = self.request_tile_pixels[0] * resolution
        tile_height_crs = self.request_tile_pixels[1] * resolution

        tile_factory = BoxTiler(
            tile_size=(tile_width_crs, tile_height_crs),
            origin=self.origin,
            )
        
        # Get standardized tiles
        standard_tiles = tile_factory.tile_bbox(bbox)
        

        arrays_or_paths = await self._download_tiles(standard_tiles, max_concurrent)

        if self.use_temp_storage:
            merged = self._merge_arrays_disk(arrays_or_paths)
        else:
            merged = self._merge_arrays(arrays_or_paths)

        merged.attrs.update({
            "source_url": self.endpoint,
            "coverage_id": self.coverage_id
        })
        
        return merged

    async def _download_tiles(
        self,
        tiles: List[Tuple[float, float, float, float]],
        max_concurrent: int,
    ) -> Union[List[xr.DataArray], List[Path]]:
        """Download tiles and keep them in memory."""
        downloaded_tiles = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent)

            for tile_bbox in tiles: 
                w_px, h_px = self.request_tile_pixels
                async def tile_task(bbox=tile_bbox, w=w_px, h=h_px, to_disk=self.use_temp_storage):
                    async with semaphore:
                        data = await self._fetch_tile(session, bbox, w, h)
                        if to_disk:
                            tile_id = str(bbox) + str((w_px, h_px))
                            bbox_hash = sha256(tile_id.encode()).hexdigest()
                            path = self.temp_dir / f"{bbox_hash}.tif"
                            path = self._save_tile(data, path)
                            return path
                        else:
                            return self._bytes_to_xarray(data)

                tasks.append(tile_task())

            for fut in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="Downloading tiles"
            ):
                tile = await fut
                downloaded_tiles.append(tile)

        return downloaded_tiles

    def _bytes_to_xarray(self, data: bytes) -> xr.DataArray:
        """Converts raw GeoTIFF bytes to an xarray.DataArray.

        Handles conversion of downloaded tile data to a properly georeferenced
        xarray.DataArray with coordinates and metadata.

        Args:
            data: Raw bytes of a GeoTIFF file.

        Returns:
            xarray.DataArray with proper geospatial metadata and coordinates.

        Raises:
            ValueError: If the data cannot be read as a valid GeoTIFF.
        """
        with MemoryFile(data) as mem:
            with mem.open() as ds:
                arr = rxr.open_rasterio(ds)

        return arr

    def _save_tile(self, data: bytes, path: Path) -> Path:
        """Save tile data to disk."""
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def _merge_arrays(self, tiles: List[xr.DataArray]) -> xr.Dataset:
        """Merge tiles held in memory."""
        all_tiles = [tile.rename(self.coverage_id) for tile in tiles]
        return xr.merge(all_tiles)

    def _merge_arrays_disk(self, paths: List[Path]) -> xr.Dataset:
        """Merge tiles saved to disk."""
        arrays  = xr.open_mfdataset(paths, engine="rasterio", chunks={"x": self.request_tile_pixels[0], "y": self.request_tile_pixels[1]})
        arrays = arrays.rename({"band_data" : self.coverage_id})
        return arrays