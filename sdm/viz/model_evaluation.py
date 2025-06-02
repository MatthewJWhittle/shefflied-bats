"""Model output visualization functionality."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.plot import show
import geopandas as gpd

logger = logging.getLogger(__name__)

def plot_feature_importance(
    importance_scores: Dict[str, float],
    output_path: Path,
    title: Optional[str] = None,
    top_n: Optional[int] = None
) -> None:
    """Plot feature importance scores.
    
    Args:
        importance_scores: Dictionary of feature names and importance scores
        output_path: Path to save plot
        title: Optional plot title
        top_n: Optional number of top features to show
    """
    try:
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Select top N features if specified
        if top_n:
            importance_df = importance_df.tail(top_n)
        
        # Create plot
        plt.figure(figsize=(10, len(importance_df) * 0.4))
        sns.barplot(data=importance_df, x='importance', y='feature')
        
        # Customize plot
        plt.title(title or 'Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}", exc_info=True)
        raise

def plot_prediction_raster(
    raster_path: Union[str, Path],
    output_path: Path,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None
) -> None:
    """Plot prediction raster with optional boundary overlay.
    
    Args:
        raster_path: Path to prediction raster
        output_path: Path to save plot
        title: Optional plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        boundary_gdf: Optional GeoDataFrame containing boundary to overlay
    """
    try:
        # Read raster
        with rasterio.open(raster_path) as src:
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot raster
            im = show(
                src,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Predicted Probability')
            
            # Add boundary if provided
            if boundary_gdf is not None:
                boundary_gdf.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=1
                )
            
            # Customize plot
            plt.title(title or 'Species Distribution Prediction')
            plt.tight_layout()
            
            # Save plot
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved prediction raster plot to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error plotting prediction raster: {e}", exc_info=True)
        raise

def plot_evaluation_metrics(
    metrics: Dict[str, float],
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Plot model evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        output_path: Path to save plot
        title: Optional plot title
    """
    try:
        # Create DataFrame
        metrics_df = pd.DataFrame({
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        })
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='metric', y='value')
        
        # Customize plot
        plt.title(title or 'Model Evaluation Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved evaluation metrics plot to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting evaluation metrics: {e}", exc_info=True)
        raise

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        title: Optional plot title
    """
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title or 'Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curve plot to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}", exc_info=True)
        raise
