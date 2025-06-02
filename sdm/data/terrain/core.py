"""
Core terrain data processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import aiohttp
import tempfile

import numpy as np
import rasterio
from rasterio.warp import transform_geom
from scipy.ndimage import generic_filter
import xarray as xr

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
        
        logger.info(f"Saved slope raster to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error calculating slope: {e}", exc_info=True)
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
        
        logger.info(f"Saved aspect raster to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error calculating aspect: {e}", exc_info=True)
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
        
        logger.info(f"Saved terrain ruggedness raster to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error calculating terrain ruggedness: {e}", exc_info=True)
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
        
        logger.info(f"Saved terrain position raster to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error calculating terrain position: {e}", exc_info=True)
        raise 