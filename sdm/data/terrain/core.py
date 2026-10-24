"""
Core terrain data processing functionality.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from io import BytesIO
import asyncio
import tempfile
import xml.etree.ElementTree as ET
import requests
import aiohttp

import numpy as np
import xarray as xr
import rioxarray as rxr

from sdm.raster.utils import BoxTiler

logger = logging.getLogger(__name__)

class WCSDownloader:
    """A class to handle WCS data downloads."""
    
    def __init__(
        self,
        endpoint: str,
        coverage_id: str,
        request_tile_pixels: Tuple[int, int] = (1024, 1024),
        use_temp_storage: bool = True
    ):
        self.endpoint = endpoint
        self.coverage_id = coverage_id
        self.tile_width, self.tile_height = request_tile_pixels
        self.use_temp_storage = use_temp_storage
        self.temp_dir: Optional[Path] = None
        self.axis_labels, self.native_crs = self._fetch_coverage_description()

    def tile_bboxes_for_coverage(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float,
    ) -> List[Tuple[float, float, float, float]]:
        """Return GetCoverage boxes on the fixed national tile grid.

        Tile edges fall on multiples of ``tile_pixels × resolution`` metres
        from origin (0, 0) in the coverage CRS (EPSG:27700 for EA products).
        Request boxes for a given cell are therefore independent of the
        study-area corner.
        """
        tile_width_crs = self.tile_width * resolution
        tile_height_crs = self.tile_height * resolution
        tiler = BoxTiler(
            tile_size=(tile_width_crs, tile_height_crs),
            origin=(0.0, 0.0),
        )
        return tiler.tile_bbox(bbox)

    def _fetch_coverage_description(self) -> Tuple[List[str], str]:
        """Retrieves the coverage description (axis labels and native CRS) via DescribeCoverage."""
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
        
    @staticmethod
    def _failed_tile_message(
        tiles: List[Tuple[float, float, float, float]],
        results: List[Union[Optional[str], Optional[bytes]]],
    ) -> str:
        """Describe tiles that returned no data (for the incomplete-mosaic error)."""
        failed = [tile for tile, result in zip(tiles, results) if result is None]
        preview = ", ".join(repr(tile) for tile in failed[:5])
        if len(failed) > 5:
            preview = f"{preview}, ..."
        return (
            f"WCS download incomplete: {len(failed)} of {len(tiles)} tiles "
            f"missing (refusing mosaic with coverage gaps). Failed tiles: "
            f"{preview}"
        )

    def _assert_mosaic_covers_tiles(
        self,
        merged: xr.Dataset,
        tiles: List[Tuple[float, float, float, float]],
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """Fail if any requested tile left an empty hole inside the AOI.

        ``open_mfdataset`` keeps full bounds when edge tiles remain, so a
        dropped north-east tile shows up as a nodata block rather than a
        shrunk extent (#79). Require at least one finite pixel in each
        tile∩request overlap (centre sampling would false-fail coastal
        tiles whose centre sits on water nodata).
        """
        da = merged[self.coverage_id]
        if "band" in da.dims:
            da = da.isel(band=0)
        req_minx, req_miny, req_maxx, req_maxy = bbox
        empty: List[Tuple[float, float, float, float]] = []
        for tile in tiles:
            minx, miny, maxx, maxy = tile
            inter_minx = max(minx, req_minx)
            inter_miny = max(miny, req_miny)
            inter_maxx = min(maxx, req_maxx)
            inter_maxy = min(maxy, req_maxy)
            if inter_minx >= inter_maxx or inter_miny >= inter_maxy:
                continue
            in_overlap = (
                (da.x >= inter_minx)
                & (da.x <= inter_maxx)
                & (da.y >= inter_miny)
                & (da.y <= inter_maxy)
            )
            if not np.isfinite(da.where(in_overlap).values).any():
                empty.append(tile)
        if empty:
            preview = ", ".join(repr(tile) for tile in empty[:5])
            if len(empty) > 5:
                preview = f"{preview}, ..."
            raise ValueError(
                f"WCS mosaic has nodata holes in {len(empty)} tile(s) inside "
                f"the request bbox (refusing coverage gaps): {preview}"
            )

    async def get_coverage(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float,
        max_concurrent: int = 5
    ) -> xr.Dataset:
        """Get coverage data from WCS service using proper WCS 2.0.1 parameters."""
        if self.use_temp_storage:
            self.temp_dir = Path(tempfile.mkdtemp())
            
        try:
            tiles = self.tile_bboxes_for_coverage(bbox, resolution)

            # Download tiles
            async with aiohttp.ClientSession() as session:
                tasks = []
                for tile_bbox in tiles:
                    task = self._download_tile(session, tile_bbox, resolution)
                    tasks.append(task)
                
                # Keep results aligned with ``tiles``. Silently dropping
                # ``None`` left nodata blocks (notably the north-east) while
                # mosaic bounds still covered the study area (#79).
                results: List[Union[Optional[str], Optional[bytes]]] = []
                for i in range(0, len(tasks), max_concurrent):
                    batch = tasks[i:i + max_concurrent]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)

            if any(result is None for result in results):
                raise ValueError(self._failed_tile_message(tiles, results))

            if self.use_temp_storage:
                # Use xarray.open_mfdataset for disk-based merging.
                # Load before finally deletes tile_*.tif — callers need a
                # concrete array, not a lazy view of removed paths.
                merged = xr.open_mfdataset(
                    results,
                    engine="rasterio",
                    chunks={"x": self.tile_width, "y": self.tile_height},
                )
                merged = merged.rename({"band_data": self.coverage_id}).load()
            else:
                arrays = [
                    self._bytes_to_xarray(data).rename(self.coverage_id)
                    for data in results
                    if data is not None
                ]
                merged = xr.merge(arrays)

            self._assert_mosaic_covers_tiles(merged, tiles, bbox)
            return merged
                
        finally:
            if self.use_temp_storage and self.temp_dir:
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
    
    async def _download_tile(self, session: aiohttp.ClientSession, tile_bbox: Tuple[float, float, float, float], resolution: float) -> Union[Optional[str], Optional[bytes]]:
        """Download a single tile using proper WCS 2.0.1 parameters."""
        try:
            minx, miny, maxx, maxy = tile_bbox
            bbox_width = maxx - minx
            bbox_height = maxy - miny
            
            # Calculate appropriate pixel dimensions based on resolution
            width_px = int(bbox_width / resolution)
            height_px = int(bbox_height / resolution)
            
            # Calculate scalefactor for WCS 2.0.1
            x_scale = width_px / bbox_width
            y_scale = height_px / bbox_height
            scalefactor = max(x_scale, y_scale)
            
            params = {
                "service": "WCS",
                "version": "2.0.1",
                "request": "GetCoverage",
                "coverageId": self.coverage_id,
                "format": "image/tiff;application=geotiff",
                "width": str(width_px),
                "height": str(height_px),
                "scalefactor": str(scalefactor),
                "subset": [
                    f"{self.axis_labels[0]}({minx},{maxx})",
                    f"{self.axis_labels[1]}({miny},{maxy})",
                ],
                "subsettingcrs": self.native_crs
            }
            
            # Handle subset parameter specially
            url_params = []
            for k, v in params.items():
                if k == "subset":
                    for subset in v:
                        url_params.append(f"subset={subset}")
                else:
                    url_params.append(f"{k}={v}")
            
            url = f"{self.endpoint}?" + "&".join(url_params)
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    
                    # Check if we got an XML error response
                    if data.startswith(b'<?xml'):
                        error_text = data[:200].decode(errors="replace")
                        logger.warning("WCS returned XML error: %s", error_text)
                        return None
                    
                    # Check if it's a valid TIFF
                    if not (data.startswith(b"II*\x00") or data.startswith(b"MM\x00*")):
                        logger.warning("Downloaded data is not a valid TIFF")
                        return None
                    
                    if self.use_temp_storage and self.temp_dir:
                        # Stable names (not hash) so collisions cannot overwrite
                        # a neighbouring tile and leave a silent coverage gap.
                        temp_file = (
                            self.temp_dir
                            / f"tile_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}.tif"
                        )
                        with open(temp_file, "wb") as f:
                            f.write(data)
                        return str(temp_file)
                    else:
                        # Return raw bytes for in-memory storage
                        return data
                else:
                    logger.warning("Failed to download tile: %s (status: %s)", url, response.status)
        except Exception as e:
            logger.error("Error downloading tile %s: %s", tile_bbox, e)
        return None

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
        with BytesIO(data) as bytes_io:
            arr = rxr.open_rasterio(bytes_io, masked=True)
        return arr


def create_terrain_wcs_downloaders(
    tile_pixels: Tuple[int, int] = (1024, 1024),
    use_temp_storage: bool = True
) -> Dict[str, WCSDownloader]:
    """Create WCS downloaders for DTM and DSM data."""
    return {
        "dtm": WCSDownloader(
            endpoint="https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
            coverage_id="13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
            request_tile_pixels=tile_pixels,
            use_temp_storage=use_temp_storage
        ),
        "dsm": WCSDownloader(
            endpoint="https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
            coverage_id="9ba4d5ac-d596-445a-9056-dae3ddec0178__Lidar_Composite_Elevation_LZ_DSM_1m",
            request_tile_pixels=tile_pixels,
            use_temp_storage=use_temp_storage
        )
    }

