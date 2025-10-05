"""
Core terrain data processing functionality.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from io import BytesIO
import asyncio
import aiohttp
import tempfile
import xml.etree.ElementTree as ET
import requests

import numpy as np
import rasterio
from scipy.ndimage import generic_filter
import xarray as xr
import rioxarray as rxr

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
            # Calculate tile bounds using BoxTiler approach from old implementation
            minx, miny, maxx, maxy = bbox
            tile_width_crs = self.tile_width * resolution
            tile_height_crs = self.tile_height * resolution
            
            # Generate tiles using BoxTiler logic
            tiles = []
            x = minx
            while x < maxx:
                y = miny
                while y < maxy:
                    tile_maxx = min(x + tile_width_crs, maxx)
                    tile_maxy = min(y + tile_height_crs, maxy)
                    tiles.append((x, y, tile_maxx, tile_maxy))
                    y += tile_height_crs
                x += tile_width_crs
            
            # Download tiles
            async with aiohttp.ClientSession() as session:
                tasks = []
                for tile_bbox in tiles:
                    task = self._download_tile(session, tile_bbox, resolution)
                    tasks.append(task)
                
                # Process tiles in batches
                results = []
                for i in range(0, len(tasks), max_concurrent):
                    batch = tasks[i:i + max_concurrent]
                    batch_results = await asyncio.gather(*batch)
                    results.extend([r for r in batch_results if r is not None])
            
            # Merge tiles using the same approach as old implementation
            if results:
                if self.use_temp_storage:
                    # Use xarray.open_mfdataset for disk-based merging
                    merged = xr.open_mfdataset(results, engine="rasterio", chunks={"x": self.tile_width, "y": self.tile_height})
                    merged = merged.rename({"band_data": self.coverage_id})
                else:
                    # Merge arrays in memory - results are bytes that need to be converted to xarray
                    arrays = []
                    for data in results:
                        if data is not None:  # Skip None results
                            arr = self._bytes_to_xarray(data)
                            arrays.append(arr.rename(self.coverage_id))
                    merged = xr.merge(arrays)
                return merged
            else:
                raise ValueError("No valid tiles were downloaded")
                
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
                        temp_file = self.temp_dir / f"tile_{hash(str(tile_bbox))}.tif"
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

def calculate_slope(
    dem_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Path:
    """Calculate slope from DEM.
    
    Args:
        dem_raster: Path to DEM raster
        output_path: Path to save slope raster
        window_size: Size of moving window for calculation
        
    Returns:
        Path to slope raster
    """
    try:
        # Read DEM
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate slope
        def slope_func(window):
            if nodata in window:
                return nodata
            center = window[len(window)//2]
            if center == nodata:
                return nodata
            dx = (window[2] - window[0]) / (2 * transform[0])
            dy = (window[6] - window[4]) / (2 * transform[0])
            return np.arctan(np.sqrt(dx*dx + dy*dy)) * 180 / np.pi
        
        slope = generic_filter(
            dem,
            slope_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Save slope raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=slope.shape[0],
            width=slope.shape[1],
            count=1,
            dtype=slope.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(slope, 1)
        
        logger.info("Saved slope raster to: %s", output_path)
        return output_path
        
    except Exception as e:
        logger.error("Error calculating slope: %s", e, exc_info=True)
        raise

def calculate_aspect(
    dem_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Path:
    """Calculate aspect from DEM.
    
    Args:
        dem_raster: Path to DEM raster
        output_path: Path to save aspect raster
        window_size: Size of moving window for calculation
        
    Returns:
        Path to aspect raster
    """
    try:
        # Read DEM
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate aspect
        def aspect_func(window):
            if nodata in window:
                return nodata
            center = window[len(window)//2]
            if center == nodata:
                return nodata
            dx = (window[2] - window[0]) / (2 * transform[0])
            dy = (window[6] - window[4]) / (2 * transform[0])
            aspect = np.arctan2(dy, dx) * 180 / np.pi
            return (aspect + 360) % 360
        
        aspect = generic_filter(
            dem,
            aspect_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Save aspect raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=aspect.shape[0],
            width=aspect.shape[1],
            count=1,
            dtype=aspect.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(aspect, 1)
        
        logger.info("Saved aspect raster to: %s", output_path)
        return output_path
        
    except Exception as e:
        logger.error("Error calculating aspect: %s", e, exc_info=True)
        raise

def calculate_terrain_ruggedness(
    dem_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Path:
    """Calculate terrain ruggedness index (TRI) from DEM.
    
    Args:
        dem_raster: Path to DEM raster
        output_path: Path to save TRI raster
        window_size: Size of moving window for calculation
        
    Returns:
        Path to TRI raster
    """
    try:
        # Read DEM
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate TRI
        def tri_func(window):
            if nodata in window:
                return nodata
            center = window[len(window)//2]
            if center == nodata:
                return nodata
            diffs = np.abs(window - center)
            return np.sqrt(np.sum(diffs * diffs))
        
        tri = generic_filter(
            dem,
            tri_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Save TRI raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=tri.shape[0],
            width=tri.shape[1],
            count=1,
            dtype=tri.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(tri, 1)
        
        logger.info("Saved terrain ruggedness raster to: %s", output_path)
        return output_path
        
    except Exception as e:
        logger.error("Error calculating terrain ruggedness: %s", e, exc_info=True)
        raise

def calculate_terrain_position(
    dem_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Path:
    """Calculate terrain position index (TPI) from DEM.
    
    Args:
        dem_raster: Path to DEM raster
        output_path: Path to save TPI raster
        window_size: Size of moving window for calculation
        
    Returns:
        Path to TPI raster
    """
    try:
        # Read DEM
        with rasterio.open(dem_raster) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate TPI
        def tpi_func(window):
            if nodata in window:
                return nodata
            center = window[len(window)//2]
            if center == nodata:
                return nodata
            return center - np.mean(window[window != nodata])
        
        tpi = generic_filter(
            dem,
            tpi_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Save TPI raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=tpi.shape[0],
            width=tpi.shape[1],
            count=1,
            dtype=tpi.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(tpi, 1)
        
        logger.info("Saved terrain position raster to: %s", output_path)
        return output_path
        
    except Exception as e:
        logger.error("Error calculating terrain position: %s", e, exc_info=True)
        raise 