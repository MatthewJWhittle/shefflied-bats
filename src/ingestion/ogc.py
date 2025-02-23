import asyncio
import math
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple


import aiohttp
import numpy as np
from rasterio.io import MemoryFile
import xarray as xr
# import required to enable functionality even if not used directly
import rioxarray as rxr 
import requests
from tqdm.asyncio import tqdm_asyncio


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
    ):
        """Initializes the WCS downloader.

        Args:
            endpoint: Base URL of the WCS service.
            coverage_id: Identifier for the coverage to download.
            fill_value: Value to use for missing or invalid data. Defaults to np.nan.
        """
        self.endpoint = endpoint
        self.coverage_id = coverage_id
        self.fill_value = fill_value if fill_value is not None else np.nan
        self.axis_labels, self.native_crs = self._fetch_coverage_description()

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

    def split_bbox(
        self, bbox: Tuple[float, float, float, float], tile_size: Tuple[float, float]
    ) -> List[Tuple[Tuple[float, float, float, float], int, int]]:
        """
        Split an overall bounding box into smaller tiles of given size.

        Args:
            bbox: (minx, miny, maxx, maxy).
            tile_size: (tile_width, tile_height).
        Returns:
            A list of tuples: (tile_bbox, col_index, row_index).
        """
        minx, miny, maxx, maxy = bbox
        tile_w, tile_h = tile_size
        cols = math.ceil((maxx - minx) / tile_w)
        rows = math.ceil((maxy - miny) / tile_h)

        tiles = []
        for j in range(rows):  # j=0 is bottom row, j=rows-1 is top
            for i in range(cols):
                tx_min = minx + i * tile_w
                tx_max = min(tx_min + tile_w, maxx)
                ty_min = miny + j * tile_h
                ty_max = min(ty_min + tile_h, maxy)
                tiles.append(((tx_min, ty_min, tx_max, ty_max), i, j))
        return tiles

    def standardize_bbox(
        self, bbox: Tuple[float, float, float, float], tile_size: Tuple[float, float]
    ) -> List[Tuple[float, float, float, float]]:
        """Splits a bounding box into standardized tiles.

        Creates a set of tiles that align with a global grid to ensure consistent
        tiling across different requests.

        Args:
            bbox: Tuple of (minx, miny, maxx, maxy) coordinates.
            tile_size: Tuple of (width, height) for the tiles.

        Returns:
            List of tuples, each containing (minx, miny, maxx, maxy) for a tile.
        """
        minx, miny, maxx, maxy = bbox
        tile_w, tile_h = tile_size

        # Round to nearest tile boundary
        start_x = math.floor(minx / tile_w) * tile_w
        start_y = math.floor(miny / tile_h) * tile_h
        end_x = math.ceil(maxx / tile_w) * tile_w
        end_y = math.ceil(maxy / tile_h) * tile_h

        standard_tiles = []
        for y in np.arange(start_y, end_y, tile_h):
            for x in np.arange(start_x, end_x, tile_w):
                tile_bbox = (x, y, x + tile_w, y + tile_h)
                standard_tiles.append(tile_bbox)

        return standard_tiles

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
        tile_size: Optional[Tuple[float, float]] = None,
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
        minx, miny, maxx, maxy = bbox

        if tile_size is None:
            tile_size = (maxx - minx, maxy - miny)
        else:
            if tile_size[0] <= 0 or tile_size[1] <= 0:
                raise ValueError("tile_size must be positive")

        # Get standardized tiles
        standard_tiles = self.standardize_bbox(bbox, tile_size)

        # Download missing tiles
        downloaded_tiles = await self._download_tiles(
            standard_tiles, resolution, max_concurrent
        )

        # Merge all tiles
        merged_tiles = self._prepare_result(downloaded_tiles)
        # add attrs
        merged_tiles.attrs["source_url"] = self.endpoint
        merged_tiles.attrs["coverage_id"] = self.coverage_id

        return merged_tiles

    async def _download_tiles(
        self,
        tiles: List[Tuple[float, float, float, float]],
        resolution: float,
        max_concurrent: int,
    ) -> List[xr.DataArray]:
        """Downloads multiple tiles concurrently.

        Args:
            tiles: List of bbox tuples defining the tiles to download.
            resolution: Pixel size in coordinate system units.
            max_concurrent: Maximum number of simultaneous downloads.

        Returns:
            List of xarray.DataArray objects, one per downloaded tile.

        Raises:
            aiohttp.ClientError: If any tile download fails.
        """
        downloaded_tiles = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent)

            for tile_bbox in tiles:
                w_px = int(np.ceil((tile_bbox[2] - tile_bbox[0]) / resolution))
                h_px = int(np.ceil((tile_bbox[3] - tile_bbox[1]) / resolution))

                async def tile_task(bbox=tile_bbox, w=w_px, h=h_px):
                    async with semaphore:
                        data = await self._fetch_tile(session, bbox, w, h)
                        return (bbox, data)

                tasks.append(tile_task())

            for fut in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="Downloading tiles"
            ):
                _, data = await fut
                tile_array = self._bytes_to_xarray(data)
                downloaded_tiles.append(tile_array)

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
                arr = ds.read(1)
                transform = ds.transform
                width, height = ds.width, ds.height
                nodata = ds.nodata

        # Extract pixel size
        res_x = transform.a  # Pixel width
        res_y = -transform.e  # Pixel height (negative since Y decreases)

        # Generate X and Y coordinates (centers of the cells)
        x_coords = np.arange(width) * res_x + transform.xoff + (res_x / 2)
        y_coords = np.arange(height) * res_y + transform.yoff + (res_y / 2)

        # Flip Y to match array indexing (descending order)
        y_coords = y_coords[::-1]

        nodata = nodata if nodata is not None else self.fill_value

        da = xr.DataArray(
            arr,
            dims=("y", "x"),
            coords={
                "y": y_coords,
                "x": x_coords,
            },
        )
        da.attrs["transform"] = transform
        da.attrs["crs"] = self.native_crs
        da.rio.write_nodata(nodata, inplace=True)
        return da

    def _prepare_result(self, tiles: List[xr.DataArray]) -> xr.Dataset:
        """Merges downloaded tiles into a single dataset.

        Args:
            tiles: List of tile DataArrays to merge.

        Returns:
            xarray.Dataset containing the merged data with appropriate metadata.
        """
        all_tiles = [tile.rename(self.coverage_id) for tile in tiles]
        merged = xr.merge([tile for tile in all_tiles])
        merged.attrs.update(
            {"source_url": self.endpoint, "coverage_id": self.coverage_id}
        )
        return merged
