"""
Core data processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_geom
import elapid as ela

logger = logging.getLogger(__name__)

def merge_environmental_layers(
    layer_paths: List[Path],
    output_path: Path,
    resolution: Optional[float] = None,
    crs: Optional[str] = None
) -> None:
    """Merge multiple environmental layers into a single raster.
    
    Args:
        layer_paths: List of paths to environmental layers
        output_path: Path to save merged raster
        resolution: Optional target resolution
        crs: Optional target CRS
    """
    try:
        # Read first layer to get reference
        with rasterio.open(layer_paths[0]) as src:
            if resolution is None:
                resolution = src.res[0]
            if crs is None:
                crs = src.crs
            
            # Calculate new transform
            transform = rasterio.transform.from_origin(
                src.bounds.left,
                src.bounds.top,
                resolution,
                resolution
            )
            
            # Calculate new dimensions
            width = int((src.bounds.right - src.bounds.left) / resolution)
            height = int((src.bounds.top - src.bounds.bottom) / resolution)
            
            # Create output raster
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=len(layer_paths),
                dtype=src.dtypes[0],
                crs=crs,
                transform=transform,
                nodata=src.nodata
            ) as dst:
                # Write each layer
                for i, layer_path in enumerate(layer_paths, 1):
                    with rasterio.open(layer_path) as src2:
                        # Read and resample data
                        data = src2.read(
                            1,
                            out_shape=(height, width),
                            resampling=rasterio.enums.Resampling.bilinear
                        )
                        dst.write(data, i)
        
        logger.info(f"Successfully merged {len(layer_paths)} layers to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error merging environmental layers: {e}", exc_info=True)
        raise

def process_occurrence_data(
    occurrence_gdf: gpd.GeoDataFrame,
    environmental_raster: Path,
    output_path: Path,
    min_points: int = 15,
    max_points: Optional[int] = None,
    random_state: int = 42
) -> gpd.GeoDataFrame:
    """Process occurrence data for SDM.
    
    Args:
        occurrence_gdf: GeoDataFrame containing occurrence points
        environmental_raster: Path to environmental raster
        output_path: Path to save processed data
        min_points: Minimum number of points required
        max_points: Optional maximum number of points to keep
        random_state: Random state for reproducibility
        
    Returns:
        Processed GeoDataFrame
    """
    try:
        # Check number of points
        if len(occurrence_gdf) < min_points:
            raise ValueError(f"Insufficient points: {len(occurrence_gdf)} < {min_points}")
        
        # Subsample if needed
        if max_points and len(occurrence_gdf) > max_points:
            occurrence_gdf = occurrence_gdf.sample(
                n=max_points,
                random_state=random_state
            )
        
        # Ensure points are in same CRS as raster
        with rasterio.open(environmental_raster) as src:
            if occurrence_gdf.crs != src.crs:
                occurrence_gdf = occurrence_gdf.to_crs(src.crs)
        
        # Save processed data
        occurrence_gdf.to_file(output_path)
        logger.info(f"Saved processed occurrence data to: {output_path}")
        
        return occurrence_gdf
        
    except Exception as e:
        logger.error(f"Error processing occurrence data: {e}", exc_info=True)
        raise

def process_background_data(
    boundary_gdf: gpd.GeoDataFrame,
    environmental_raster: Path,
    output_path: Path,
    n_points: int = 10000,
    random_state: int = 42
) -> gpd.GeoDataFrame:
    """Generate and process background points.
    
    Args:
        boundary_gdf: GeoDataFrame containing study boundary
        environmental_raster: Path to environmental raster
        output_path: Path to save processed data
        n_points: Number of background points to generate
        random_state: Random state for reproducibility
        
    Returns:
        GeoDataFrame containing background points
    """
    try:
        # Ensure boundary is in same CRS as raster
        with rasterio.open(environmental_raster) as src:
            if boundary_gdf.crs != src.crs:
                boundary_gdf = boundary_gdf.to_crs(src.crs)
        
        # Generate random points within boundary
        points = []
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
        
        np.random.seed(random_state)
        while len(points) < n_points:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = gpd.points_from_xy([x], [y])[0]
            
            if boundary_gdf.contains(point).any():
                points.append(point)
        
        # Create GeoDataFrame
        background_gdf = gpd.GeoDataFrame(
            geometry=points,
            crs=boundary_gdf.crs
        )
        
        # Save processed data
        background_gdf.to_file(output_path)
        logger.info(f"Saved processed background data to: {output_path}")
        
        return background_gdf
        
    except Exception as e:
        logger.error(f"Error processing background data: {e}", exc_info=True)
        raise

def extract_environmental_data(
    points_gdf: gpd.GeoDataFrame,
    environmental_raster: Path,
    output_path: Path
) -> gpd.GeoDataFrame:
    """Extract environmental data at point locations.
    
    Args:
        points_gdf: GeoDataFrame containing points
        environmental_raster: Path to environmental raster
        output_path: Path to save extracted data
        
    Returns:
        GeoDataFrame with environmental data
    """
    try:
        # Ensure points are in same CRS as raster
        with rasterio.open(environmental_raster) as src:
            if points_gdf.crs != src.crs:
                points_gdf = points_gdf.to_crs(src.crs)
            
            # Extract values
            values = []
            for idx, row in points_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                row, col = src.index(x, y)
                value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                values.append(value)
            
            # Add values to GeoDataFrame
            points_gdf['value'] = values
            
            # Save results
            points_gdf.to_file(output_path)
            logger.info(f"Saved extracted environmental data to: {output_path}")
            
            return points_gdf
            
    except Exception as e:
        logger.error(f"Error extracting environmental data: {e}", exc_info=True)
        raise

def annotate_points(
    bats: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    ev_raster: Union[str, Path],
    ev_columns: List[str]
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Annotate bat and background points with environmental variables.
    
    Args:
        bats: GeoDataFrame containing bat occurrence points
        background: GeoDataFrame containing background points
        ev_raster: Path to environmental variables raster
        ev_columns: List of environmental variable column names
        
    Returns:
        Tuple of (annotated bat points, annotated background points)
    """
    try:
        bats_ant = ela.annotate(bats, str(ev_raster), labels=ev_columns)
        background = ela.annotate(background, str(ev_raster), labels=ev_columns)
        logger.info("Successfully annotated points with environmental variables")
        return bats_ant, background
    except Exception as e:
        logger.error(f"Error annotating points: {e}")
        raise 