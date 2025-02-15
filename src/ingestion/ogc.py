import asyncio
import math
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import xarray as xr
import requests
from shapely.geometry import box
from tqdm.asyncio import tqdm_asyncio  # Progress bar for async tasks


class WCSDownloader:
    """
    Minimal client to retrieve a coverage from a WCS 2.0.1 service.

    This class obtains coverage metadata (axis labels and CRS) via DescribeCoverage,
    then splits an input bounding box into tiles, downloads them asynchronously via
    GetCoverage, and stitches them into a single-band xarray.DataArray.
    """

    def __init__(self, endpoint: str, coverage_id: str, fill_value: Optional[float] = None):
        """
        Initialize the client.

        Args:
            endpoint: The base URL for the WCS service (without query parameters).
            coverage_id: The identifier for the desired coverage.
            fill_value: Optional value to use if a tile fails to download.
        """
        self.endpoint = endpoint
        self.coverage_id = coverage_id
        self.fill_value = fill_value
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
        resp = requests.get(self.endpoint, params=params)
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

    def split_bbox(self, bbox: Tuple[float, float, float, float],
                   tile_size: Tuple[float, float]) -> List[Tuple[Tuple[float, float, float, float], int, int]]:
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

    async def _fetch_tile(self,
                          session: aiohttp.ClientSession,
                          bbox: Tuple[float, float, float, float],
                          width: int,
                          height: int) -> bytes:
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
            if resp.status != 200:
                raise Exception(f"Tile request failed with status {resp.status}")
            data = await resp.read()
            # Make sure we didn't get an XML error doc or invalid TIFF
            d = data.lstrip()
            if d.startswith(b'<?xml'):
                snippet = d[:200].decode(errors="replace")
                raise ValueError(f"Received XML instead of TIFF: {snippet}")
            if not (d.startswith(b'II*\x00') or d.startswith(b'MM\x00*')):
                raise ValueError("Data does not appear to be a valid TIFF.")
            return data

    async def get_coverage(self,
                           bbox: Tuple[float, float, float, float],
                           resolution: float,
                           tile_size: Optional[Tuple[float, float]] = None,
                           max_concurrent: int = 10) -> xr.DataArray:
        """
        Asynchronously retrieve coverage data for a bounding box.

        The box is split into tiles (if tile_size is given), each tile is fetched
        concurrently (limited by max_concurrent), and results are stitched together
        into a single xarray.DataArray (single-band).
        """
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("bbox must be a tuple (minx, miny, maxx, maxy).")
        minx, miny, maxx, maxy = bbox
        total_width = int(np.ceil((maxx - minx) / resolution))
        total_height = int(np.ceil((maxy - miny) / resolution))

        if tile_size is None:
            tile_size = (maxx - minx, maxy - miny)

        # Generate tile bounding boxes
        tiles_info = self.split_bbox(bbox, tile_size)

        semaphore = asyncio.Semaphore(max_concurrent)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for tile_bbox, i, j in tiles_info:
                w_px = int(np.ceil((tile_bbox[2] - tile_bbox[0]) / resolution))
                h_px = int(np.ceil((tile_bbox[3] - tile_bbox[1]) / resolution))

                async def tile_task(bbox=tile_bbox, w=w_px, h=h_px, i=i, j=j):
                    async with semaphore:
                        data = await self._fetch_tile(session, bbox, w, h)
                        return (i, j, data)

                tasks.append(tile_task())

            # Collect results with a progress bar
            results = []
            for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Downloading tiles"):
                results.append(await fut)

        # We need to figure out the data dtype from at least one tile
        tile_dtype = None
        tile_index_map: Dict[Tuple[int, int], Tuple[bytes, Tuple[int, int]]] = {}

        for i, j, data in results:
            with MemoryFile(data) as mem:
                with mem.open() as ds:
                    arr = ds.read(1)   # single band only
                    if tile_dtype is None:
                        tile_dtype = arr.dtype
                    shape_2d = arr.shape  # (rows, cols)
            tile_index_map[(j, i)] = (data, shape_2d)

        # For each row j, find the maximum tile height, and for each column i, the maximum tile width
        row_heights: Dict[int, int] = {}
        col_widths: Dict[int, int] = {}

        for (j, i), (data, (tile_h, tile_w)) in tile_index_map.items():
            row_heights[j] = max(row_heights.get(j, 0), tile_h)
            col_widths[i] = max(col_widths.get(i, 0), tile_w)

        total_px_height = sum(row_heights[j] for j in sorted(row_heights))
        total_px_width = sum(col_widths[i] for i in sorted(col_widths))

        # -------------------------------------------------
        # IMPORTANT: fix the 'shutter' effect by placing j=0 at top in array coords,
        # then flipping at the end. So we sort row_heights in descending order
        # when building offsets.
        # -------------------------------------------------

        row_offsets: Dict[int, int] = {}
        cum = 0
        # Sort in descending order, so the highest j gets offset=0 at top
        # and the lowest j ends up at the bottom of the array.
        for j in sorted(row_heights.keys(), reverse=True):
            row_offsets[j] = cum
            cum += row_heights[j]

        col_offsets: Dict[int, int] = {}
        cum = 0
        for i in sorted(col_widths.keys()):
            col_offsets[i] = cum
            cum += col_widths[i]

        # Create the final stitched array
        # Convert fill_value to tile_dtype
        if self.fill_value is not None:
            fill_val = np.array(self.fill_value, dtype=tile_dtype)
        else:
            fill_val = np.array(0, dtype=tile_dtype)

        stitched = np.full((total_px_height, total_px_width), fill_val, dtype=tile_dtype)

        # Write each tile into the final array
        for (j, i), (data, (tile_h, tile_w)) in tile_index_map.items():
            with MemoryFile(data) as mem:
                with mem.open() as ds:
                    tile_array = ds.read(1)

            # y offset: the "top" of the mosaic is j=max => offset=0
            # j=0 (lowest tile) => largest offset
            y_off = row_offsets[j]
            x_off = col_offsets[i]

            # place the tile array directly
            stitched[y_off : y_off + tile_h, x_off : x_off + tile_w] = tile_array

        # Build xarray with georeferencing
        transform = from_bounds(minx, miny, maxx, maxy, total_px_width, total_px_height)
        da = xr.DataArray(stitched, dims=("y", "x"))
        da.attrs["transform"] = transform
        da.attrs["crs"] = self.native_crs

        # Finally, flip so row 0 in the array becomes the bottom row in coords.
        # This aligns with the usual "north up" raster convention,
        # so (row=0, col=0) is top-left prior to flip, but after flipping, row=0 is the bottom.
        # from_bounds() expects row 0 to be top, so flipping is correct to align geometry.
        da = da.isel(y=slice(None, None, -1))

        # Then assign the coverage ID as the name
        da = da.rename(self.coverage_id)
        # add an attribute for the url
        da.attrs["source_url"] = self.endpoint
        da.attrs["coverage_id"] = self.coverage_id

        return da
