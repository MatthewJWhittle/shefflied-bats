"""
Vector data loading functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

def load_os_shps(
    datasets: List[str], 
    data_dir: Union[str, Path] = "data/raw/big-files/os-vector-map"
) -> Dict[str, gpd.GeoDataFrame]:
    """Load Ordnance Survey shapefiles.
    
    Args:
        datasets: List of dataset names to load
        data_dir: Directory containing shapefiles
        
    Returns:
        Dictionary mapping dataset names to GeoDataFrames
    """
    data_dir = Path(data_dir)
    results = {}
    
    for dataset in datasets:
        shp_path = data_dir / f"{dataset}.shp"
        if not shp_path.exists():
            logger.warning(f"Shapefile not found: {shp_path}")
            continue
            
        try:
            gdf = gpd.read_file(shp_path)
            results[dataset] = gdf
            logger.info(f"Loaded {dataset} from {shp_path}")
        except Exception as e:
            logger.error(f"Error loading {dataset}: {e}")
            
    return results

def load_bat_data(
    bats_path: Union[str, Path],
    accuracy_threshold: int = 100
) -> gpd.GeoDataFrame:
    """Load bat occurrence data.
    
    Args:
        bats_path: Path to bat data file
        accuracy_threshold: Maximum allowed coordinate uncertainty in meters
        
    Returns:
        GeoDataFrame containing bat occurrence data
    """
    bats_path = Path(bats_path)
    
    try:
        gdf = gpd.read_file(bats_path)
        if 'coordinateUncertaintyInMeters' in gdf.columns:
            gdf = gdf[gdf['coordinateUncertaintyInMeters'] <= accuracy_threshold]
        logger.info(f"Loaded {len(gdf)} bat records from {bats_path}")
        return gdf
    except Exception as e:
        logger.error(f"Error loading bat data: {e}")
        raise

def load_background_points(
    background_path: Union[str, Path]
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Load background points data.
    
    Args:
        background_path: Path to background points file
        
    Returns:
        Tuple of (GeoDataFrame containing points, Series of weights)
    """
    background_path = Path(background_path)
    
    try:
        gdf = gpd.read_file(background_path)
        weights = gdf['weight'] if 'weight' in gdf.columns else pd.Series(1, index=gdf.index)
        logger.info(f"Loaded {len(gdf)} background points from {background_path}")
        return gdf, weights
    except Exception as e:
        logger.error(f"Error loading background points: {e}")
        raise

