"""Feature extraction and processing functionality for SDM models."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def extract_features_from_raster(
    points_gdf: pd.DataFrame,
    raster_path: Path,
    band: int = 1,
    column_name: Optional[str] = None
) -> pd.Series:
    """Extract raster values at point locations.
    
    Args:
        points_gdf: DataFrame containing point coordinates
        raster_path: Path to raster file
        band: Band number to extract (1-based)
        column_name: Name for the extracted values column
        
    Returns:
        Series containing extracted values
    """
    import rasterio
    from rasterio.warp import transform_geom
    
    try:
        with rasterio.open(raster_path) as src:
            # Transform points to raster CRS if needed
            if points_gdf.crs != src.crs:
                points_gdf = points_gdf.to_crs(src.crs)
            
            # Extract values
            values = []
            for idx, row in points_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                row, col = src.index(x, y)
                try:
                    value = src.read(band, window=((row, row+1), (col, col+1)))[0, 0]
                    values.append(value)
                except IndexError:
                    values.append(np.nan)
            
            # Create series
            if column_name is None:
                column_name = Path(raster_path).stem
            
            return pd.Series(values, index=points_gdf.index, name=column_name)
            
    except Exception as e:
        logger.error(f"Error extracting features from raster {raster_path}: {e}", exc_info=True)
        raise

def create_feature_pipeline(
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """Create a feature processing pipeline.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: Optional list of categorical feature names
        
    Returns:
        sklearn Pipeline for feature processing
    """
    transformers = []
    
    # Add numeric feature transformer
    if numeric_features:
        transformers.append(
            ('numeric', StandardScaler(), numeric_features)
        )
    
    # Add categorical feature transformer if needed
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=transformers))
    ])
    
    return pipeline

def select_features(
    data: pd.DataFrame,
    target: str,
    feature_columns: List[str],
    method: str = 'correlation',
    threshold: float = 0.1,
    max_features: Optional[int] = None
) -> List[str]:
    """Select features based on various methods.
    
    Args:
        data: DataFrame containing features and target
        target: Target column name
        feature_columns: List of feature column names
        method: Feature selection method ('correlation' or 'mutual_info')
        threshold: Threshold for feature selection
        max_features: Maximum number of features to select
        
    Returns:
        List of selected feature names
    """
    if method == 'correlation':
        # Calculate correlations
        correlations = data[feature_columns].corrwith(data[target]).abs()
        selected = correlations[correlations > threshold].index.tolist()
        
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(
            data[feature_columns],
            data[target],
            random_state=42
        )
        
        # Select features above threshold
        selected = [
            feature for feature, score in zip(feature_columns, mi_scores)
            if score > threshold
        ]
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Limit number of features if specified
    if max_features and len(selected) > max_features:
        if method == 'correlation':
            selected = correlations.nlargest(max_features).index.tolist()
        else:
            selected = [
                feature for feature, score in sorted(
                    zip(feature_columns, mi_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:max_features]
            ]
    
    logger.info(f"Selected {len(selected)} features using {method} method")
    return selected

def save_feature_importance(
    importance_scores: Dict[str, float],
    output_path: Path,
    species_name: str,
    activity_type: str
) -> None:
    """Save feature importance scores to a CSV file.
    
    Args:
        importance_scores: Dictionary of feature names and importance scores
        output_path: Path to save results
        species_name: Name of the species
        activity_type: Type of activity
    """
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })
        
        # Add metadata
        importance_df['species'] = species_name
        importance_df['activity_type'] = activity_type
        importance_df['timestamp'] = pd.Timestamp.now()
        
        # Save to CSV
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Saved feature importance scores to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving feature importance scores: {e}", exc_info=True)
        raise
