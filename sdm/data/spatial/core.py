"""
Core spatial data processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from scipy.ndimage import generic_filter

logger = logging.getLogger(__name__)

def calculate_coastal_distance(
    points_gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    output_path: Path
) -> gpd.GeoDataFrame:
    """Calculate distance to coastline for points.
    
    Args:
        points_gdf: GeoDataFrame containing points
        coastline_gdf: GeoDataFrame containing coastline
        output_path: Path to save results
        
    Returns:
        GeoDataFrame with coastal distances
    """
    try:
        # Ensure same CRS
        if points_gdf.crs != coastline_gdf.crs:
            coastline_gdf = coastline_gdf.to_crs(points_gdf.crs)
        
        # Calculate distances
        distances = []
        for idx, row in points_gdf.iterrows():
            point = row.geometry
            min_dist = float('inf')
            
            for _, coast_row in coastline_gdf.iterrows():
                coast = coast_row.geometry
                dist = point.distance(coast)
                min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        # Add distances to GeoDataFrame
        points_gdf['coastal_distance'] = distances
        
        # Save results
        points_gdf.to_file(output_path)
        logger.info(f"Saved coastal distances to: {output_path}")
        
        return points_gdf
        
    except Exception as e:
        logger.error(f"Error calculating coastal distances: {e}", exc_info=True)
        raise

def create_study_boundary(
    points_gdf: gpd.GeoDataFrame,
    buffer_distance: float,
    output_path: Path
) -> gpd.GeoDataFrame:
    """Create study boundary by buffering points.
    
    Args:
        points_gdf: GeoDataFrame containing points
        buffer_distance: Buffer distance in CRS units
        output_path: Path to save boundary
        
    Returns:
        GeoDataFrame containing boundary
    """
    try:
        # Create boundary
        boundary = points_gdf.geometry.unary_union.buffer(buffer_distance)
        
        # Create GeoDataFrame
        boundary_gdf = gpd.GeoDataFrame(
            geometry=[boundary],
            crs=points_gdf.crs
        )
        
        # Save boundary
        boundary_gdf.to_file(output_path)
        logger.info(f"Saved study boundary to: {output_path}")
        
        return boundary_gdf
        
    except Exception as e:
        logger.error(f"Error creating study boundary: {e}", exc_info=True)
        raise

def calculate_terrain_metrics(
    dem_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Dict[str, Path]:
    """Calculate terrain metrics from DEM.
    
    Args:
        dem_raster: Path to DEM raster
        output_path: Path to save results
        window_size: Size of moving window for calculations
        
    Returns:
        Dictionary of metric names and output paths
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
        
        # Save results
        output_paths = {}
        for name, data in [('slope', slope), ('aspect', aspect)]:
            output_file = output_path / f"{name}.tif"
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ) as dst:
                dst.write(data, 1)
            output_paths[name] = output_file
            logger.info(f"Saved {name} to: {output_file}")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"Error calculating terrain metrics: {e}", exc_info=True)
        raise

def calculate_land_cover_metrics(
    land_cover_raster: Path,
    output_path: Path,
    window_size: int = 3
) -> Dict[str, Path]:
    """Calculate land cover metrics.
    
    Args:
        land_cover_raster: Path to land cover raster
        output_path: Path to save results
        window_size: Size of moving window for calculations
        
    Returns:
        Dictionary of metric names and output paths
    """
    try:
        # Read land cover
        with rasterio.open(land_cover_raster) as src:
            lc = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        
        # Calculate diversity
        def diversity_func(window):
            if nodata in window:
                return nodata
            unique = np.unique(window[window != nodata])
            return len(unique)
        
        diversity = generic_filter(
            lc,
            diversity_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Calculate dominance
        def dominance_func(window):
            if nodata in window:
                return nodata
            values = window[window != nodata]
            if len(values) == 0:
                return nodata
            counts = np.bincount(values)
            return np.max(counts) / len(values)
        
        dominance = generic_filter(
            lc,
            dominance_func,
            size=window_size,
            mode='constant',
            cval=nodata
        )
        
        # Save results
        output_paths = {}
        for name, data in [('diversity', diversity), ('dominance', dominance)]:
            output_file = output_path / f"{name}.tif"
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata
            ) as dst:
                dst.write(data, 1)
            output_paths[name] = output_file
            logger.info(f"Saved {name} to: {output_file}")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"Error calculating land cover metrics: {e}", exc_info=True)
        raise 