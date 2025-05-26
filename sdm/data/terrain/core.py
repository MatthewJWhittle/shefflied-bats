"""
Core terrain data processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_geom
from scipy.ndimage import generic_filter

logger = logging.getLogger(__name__)

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