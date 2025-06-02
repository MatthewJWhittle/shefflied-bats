"""Prediction visualization functionality."""

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

def plot_prediction_map(
    prediction_raster: Union[str, Path],
    output_path: Path,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    occurrence_gdf: Optional[gpd.GeoDataFrame] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    """Plot prediction map with optional boundary and occurrence points.
    
    Args:
        prediction_raster: Path to prediction raster
        output_path: Path to save plot
        boundary_gdf: Optional GeoDataFrame containing boundary
        occurrence_gdf: Optional GeoDataFrame containing occurrence points
        title: Optional plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    try:
        # Read raster
        with rasterio.open(prediction_raster) as src:
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Plot raster
            im = show(
                src,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Predicted Probability')
            
            # Add boundary if provided
            if boundary_gdf is not None:
                boundary_gdf.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=1
                )
            
            # Add occurrence points if provided
            if occurrence_gdf is not None:
                occurrence_gdf.plot(
                    ax=ax,
                    color='red',
                    markersize=50,
                    alpha=0.5,
                    label='Occurrence Points'
                )
                ax.legend()
            
            # Customize plot
            plt.title(title or 'Species Distribution Prediction')
            plt.tight_layout()
            
            # Save plot
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved prediction map to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error plotting prediction map: {e}", exc_info=True)
        raise

def plot_prediction_histogram(
    prediction_raster: Union[str, Path],
    output_path: Path,
    title: Optional[str] = None,
    bins: int = 50
) -> None:
    """Plot histogram of prediction values.
    
    Args:
        prediction_raster: Path to prediction raster
        output_path: Path to save plot
        title: Optional plot title
        bins: Number of histogram bins
    """
    try:
        # Read raster
        with rasterio.open(prediction_raster) as src:
            # Read data
            data = src.read(1)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=bins, density=True)
            
            # Customize plot
            plt.title(title or 'Distribution of Prediction Values')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved prediction histogram to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error plotting prediction histogram: {e}", exc_info=True)
        raise

def plot_prediction_comparison(
    raster_paths: List[Union[str, Path]],
    output_path: Path,
    titles: Optional[List[str]] = None,
    cmap: str = 'viridis',
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    """Plot multiple prediction rasters side by side.
    
    Args:
        raster_paths: List of paths to prediction rasters
        output_path: Path to save plot
        titles: Optional list of titles for each subplot
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    try:
        n_rasters = len(raster_paths)
        fig, axes = plt.subplots(1, n_rasters, figsize=(6 * n_rasters, 6))
        
        if n_rasters == 1:
            axes = [axes]
        
        for i, (raster_path, ax) in enumerate(zip(raster_paths, axes)):
            with rasterio.open(raster_path) as src:
                # Plot raster
                im = show(
                    src,
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Set title
                if titles and i < len(titles):
                    ax.set_title(titles[i])
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved prediction comparison to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting prediction comparison: {e}", exc_info=True)
        raise

def plot_prediction_uncertainty(
    mean_raster: Union[str, Path],
    std_raster: Union[str, Path],
    output_path: Path,
    title: Optional[str] = None,
    cmap: str = 'viridis'
) -> None:
    """Plot prediction mean and standard deviation.
    
    Args:
        mean_raster: Path to mean prediction raster
        std_raster: Path to standard deviation raster
        output_path: Path to save plot
        title: Optional plot title
        cmap: Colormap name
    """
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot mean
        with rasterio.open(mean_raster) as src:
            im1 = show(src, ax=ax1, cmap=cmap)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('Mean Prediction')
        
        # Plot standard deviation
        with rasterio.open(std_raster) as src:
            im2 = show(src, ax=ax2, cmap='hot')
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('Standard Deviation')
        
        # Set overall title
        if title:
            plt.suptitle(title)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved prediction uncertainty plot to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting prediction uncertainty: {e}", exc_info=True)
        raise
