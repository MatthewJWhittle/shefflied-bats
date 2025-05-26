"""Model prediction functionality for SDM models."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from elapid import MaxentModel

logger = logging.getLogger(__name__)

def predict_rasters_with_elapid_model(
    model: MaxentModel,
    raster_paths: List[str],
    output_path: str,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None
) -> None:
    """Generate prediction raster using an Elapid MaxentModel.
    
    Args:
        model: Trained MaxentModel
        raster_paths: List of paths to input rasters
        output_path: Path to save prediction raster
        transform: Optional affine transform for output raster
        crs: Optional CRS for output raster
    """
    try:
        # Generate prediction
        model.predict_rasters(
            raster_paths=raster_paths,
            output_path=output_path,
            transform=transform,
            crs=crs
        )
        logger.info(f"Successfully generated prediction raster at: {output_path}")
    except Exception as e:
        logger.error(f"Error generating prediction raster: {e}", exc_info=True)
        raise

def predict_points_with_model(
    model: MaxentModel,
    points_gdf: gpd.GeoDataFrame,
    feature_columns: List[str]
) -> np.ndarray:
    """Generate predictions for point data.
    
    Args:
        model: Trained model
        points_gdf: GeoDataFrame containing points to predict
        feature_columns: List of feature column names
        
    Returns:
        Array of prediction probabilities
    """
    try:
        # Extract features
        X = points_gdf[feature_columns]
        
        # Generate predictions
        predictions = model.predict_proba(X)[:, 1]
        
        logger.info(f"Successfully generated predictions for {len(points_gdf)} points")
        return predictions
    except Exception as e:
        logger.error(f"Error generating point predictions: {e}", exc_info=True)
        raise

def save_prediction_raster(
    predictions: np.ndarray,
    output_path: Union[str, Path],
    transform: Affine,
    crs: str,
    nodata: float = -9999.0
) -> None:
    """Save prediction array as a GeoTIFF raster.
    
    Args:
        predictions: 2D array of predictions
        output_path: Path to save raster
        transform: Affine transform for raster
        crs: CRS for raster
        nodata: Nodata value
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=predictions.shape[0],
            width=predictions.shape[1],
            count=1,
            dtype=predictions.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(predictions, 1)
        
        logger.info(f"Successfully saved prediction raster to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving prediction raster: {e}", exc_info=True)
        raise
