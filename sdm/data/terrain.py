"""
Terrain data processing functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import asyncio
import aiohttp
import tempfile
import os

import xarray as xr
import rioxarray as rxr
import numpy as np
from rasterio.warp import transform_geom
from scipy.ndimage import generic_filter

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
        
    async def get_coverage(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float,
        max_concurrent: int = 5
    ) -> xr.Dataset:
        """Get coverage data from WCS service."""
        if self.use_temp_storage:
            self.temp_dir = Path(tempfile.mkdtemp())
            
        try:
            # Calculate tile bounds
            minx, miny, maxx, maxy = bbox
            width = maxx - minx
            height = maxy - miny
            
            # Calculate number of tiles
            nx = int(np.ceil(width / (self.tile_width * resolution)))
            ny = int(np.ceil(height / (self.tile_height * resolution)))
            
            # Generate tile URLs
            tile_urls = []
            for i in range(nx):
                for j in range(ny):
                    tile_minx = minx + i * self.tile_width * resolution
                    tile_miny = miny + j * self.tile_height * resolution
                    tile_maxx = min(tile_minx + self.tile_width * resolution, maxx)
                    tile_maxy = min(tile_miny + self.tile_height * resolution, maxy)
                    
                    params = {
                        "service": "WCS",
                        "version": "2.0.1",
                        "request": "GetCoverage",
                        "coverageId": self.coverage_id,
                        "bbox": f"{tile_minx},{tile_miny},{tile_maxx},{tile_maxy}",
                        "width": str(self.tile_width),
                        "height": str(self.tile_height),
                        "format": "image/tiff"
                    }
                    
                    url = f"{self.endpoint}?" + "&".join(f"{k}={v}" for k, v in params.items())
                    tile_urls.append(url)
            
            # Download tiles
            async with aiohttp.ClientSession() as session:
                tasks = []
                for url in tile_urls:
                    task = self._download_tile(session, url)
                    tasks.append(task)
                
                # Process tiles in batches
                results = []
                for i in range(0, len(tasks), max_concurrent):
                    batch = tasks[i:i + max_concurrent]
                    batch_results = await asyncio.gather(*batch)
                    results.extend([r for r in batch_results if r is not None])
            
            # Merge tiles
            if results:
                datasets = []
                for r in results:
                    ds = xr.open_dataset(r, engine="rasterio")
                    datasets.append(ds)
                merged = xr.concat(datasets, dim=["x", "y"])
                # Close individual datasets
                for ds in datasets:
                    ds.close()
                return merged
            else:
                raise ValueError("No valid tiles were downloaded")
                
        finally:
            if self.use_temp_storage and self.temp_dir:
                for file in self.temp_dir.iterdir():
                    file.unlink()
                self.temp_dir.rmdir()
    
    async def _download_tile(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Download a single tile."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    if self.use_temp_storage and self.temp_dir:
                        temp_file = self.temp_dir / f"tile_{hash(url)}.tif"
                        with open(temp_file, "wb") as f:
                            f.write(await response.read())
                        return str(temp_file)
                    else:
                        # Handle in-memory storage if needed
                        pass
                else:
                    logger.warning(f"Failed to download tile: {url} (status: {response.status})")
        except Exception as e:
            logger.error(f"Error downloading tile {url}: {e}")
        return None

def create_terrain_wcs_downloaders(
    tile_pixels: Tuple[int, int] = (1024, 1024),
    use_temp_storage: bool = True
) -> Dict[str, WCSDownloader]:
    """Create WCS downloaders for DTM and DSM data."""
    return {
        "dtm": WCSDownloader(
            endpoint="https://environment.data.gov.uk/spatialdata/terrain-50-dtm/wcs",
            coverage_id="DTM_50",
            request_tile_pixels=tile_pixels,
            use_temp_storage=use_temp_storage
        ),
        "dsm": WCSDownloader(
            endpoint="https://environment.data.gov.uk/spatialdata/terrain-50-dsm/wcs",
            coverage_id="DSM_50",
            request_tile_pixels=tile_pixels,
            use_temp_storage=use_temp_storage
        )
    }

def process_dem_to_terrain_attributes(
    dem_path: Union[str, Path],
    dem_band_index: int = 0,
    slope_window_size: int = 3,
    tpi_window_size: int = 3,
    output_slope_units: str = "radians"
) -> xr.Dataset:
    """Process DEM to calculate terrain attributes."""
    with rxr.open_rasterio(dem_path) as src:
        dem = src.sel(band=dem_band_index + 1)
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
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            if output_slope_units == "degrees":
                return slope * 180 / np.pi
            elif output_slope_units == "percent":
                return np.tan(slope) * 100
            return slope
        
        slope = generic_filter(
            dem,
            slope_func,
            size=slope_window_size,
            mode='constant',
            cval=nodata
        )
        
        # Calculate aspect
        def aspect_func(window):
            if nodata in window:
                return nodata
            center = window[len(window)//2]
            if center == nodata:
                return nodata
            dx = (window[2] - window[0]) / (2 * transform[0])
            dy = (window[6] - window[4]) / (2 * transform[0])
            aspect = np.arctan2(dy, dx)
            return (aspect + 2 * np.pi) % (2 * np.pi)
        
        aspect = generic_filter(
            dem,
            aspect_func,
            size=slope_window_size,
            mode='constant',
            cval=nodata
        )
        
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
            size=tpi_window_size,
            mode='constant',
            cval=nodata
        )
        
        # Create dataset
        ds = xr.Dataset(
            {
                "dem": dem,
                "slope": xr.DataArray(slope, dims=dem.dims),
                "aspect": xr.DataArray(aspect, dims=dem.dims),
                "tpi": xr.DataArray(tpi, dims=dem.dims)
            }
        )
        
        # Add attributes
        ds.dem.attrs["long_name"] = "Digital Elevation Model"
        ds.slope.attrs["long_name"] = f"Slope ({output_slope_units})"
        ds.aspect.attrs["long_name"] = "Aspect (radians)"
        ds.tpi.attrs["long_name"] = "Topographic Position Index"
        
        return ds

def save_terrain_dataset(
    terrain_ds: xr.Dataset,
    output_path: Union[str, Path],
    drop_dem_variable: bool = True
) -> Path:
    """Save terrain dataset to GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if drop_dem_variable and "dem" in terrain_ds:
        terrain_ds = terrain_ds.drop_vars("dem")
    
    terrain_ds.rio.to_raster(output_path)
    return output_path
