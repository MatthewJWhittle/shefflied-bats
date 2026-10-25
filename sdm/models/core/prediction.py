"""Model prediction functionality for SDM models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
from elapid import MaxentModel
from rasterio.transform import Affine

from sdm.raster.io import cogify_geotiff_inplace, export_geotiff

logger = logging.getLogger(__name__)


def predict_rasters_with_elapid_model(
    model: MaxentModel,
    raster_paths: List[str],
    output_path: str,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None,
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
        model.predict_rasters(
            raster_paths=raster_paths,
            output_path=output_path,
            transform=transform,
            crs=crs,
        )
        logger.debug("Successfully generated prediction raster at: %s", output_path)
    except Exception as e:
        logger.error("Error generating prediction raster: %s", e, exc_info=True)
        raise


def predict_points_with_model(
    model: MaxentModel,
    points_gdf: gpd.GeoDataFrame,
    feature_columns: List[str],
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
        X = points_gdf[feature_columns]
        predictions = model.predict_proba(X)[:, 1]
        logger.debug(
            "Successfully generated predictions for %s points", len(points_gdf)
        )
        return predictions
    except Exception as e:
        logger.error("Error generating point predictions: %s", e, exc_info=True)
        raise


def save_prediction_raster(
    predictions: np.ndarray,
    output_path: Union[str, Path],
    transform: Affine,
    crs: str,
    nodata: float = -9999.0,
    *,
    as_cog: bool = False,
) -> Path:
    """Save prediction array as a GeoTIFF raster.

    Args:
        predictions: 2D array of predictions
        output_path: Path to save raster
        transform: Affine transform for raster
        crs: CRS for raster
        nodata: Nodata value
        as_cog: When True, rewrite the GeoTIFF as a Cloud Optimized GeoTIFF

    Returns:
        Path to the written raster.
    """
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=predictions.shape[0],
            width=predictions.shape[1],
            count=1,
            dtype=predictions.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(predictions, 1)

        if as_cog:
            cogify_geotiff_inplace(output_path)

        logger.debug("Successfully saved prediction raster to: %s", output_path)
        return output_path
    except Exception as e:
        logger.error("Error saving prediction raster: %s", e, exc_info=True)
        raise


def copy_prediction_as_cog(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
) -> Path:
    """Copy *src_path* to *dst_path* as a Cloud Optimized GeoTIFF."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    export_geotiff(src_path, dst_path, dst_crs=None, as_cog=True)
    return dst_path


def write_threshold_binary_raster(
    prediction_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    cutoff: float,
    as_cog: bool = True,
    suitable_value: int = 1,
    unsuitable_value: int = 0,
) -> Path:
    """Write a yes/no map from a continuous suitability raster and cut-off.

    Pixels with suitability ``>= cutoff`` become *suitable_value*; finite
    pixels below the cut-off become *unsuitable_value*; source nodata / non-finite
    cells stay nodata (255 for uint8).
    """
    prediction_path = Path(prediction_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(prediction_path) as src:
        data = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

        if nodata is None:
            valid = np.isfinite(data)
        elif isinstance(nodata, float) and np.isnan(nodata):
            valid = np.isfinite(data)
        else:
            valid = np.isfinite(data) & (data != nodata)

        binary = np.full(data.shape, 255, dtype=np.uint8)
        binary[valid & (data >= cutoff)] = suitable_value
        binary[valid & (data < cutoff)] = unsuitable_value

        profile.update(
            {
                "driver": "GTiff",
                "dtype": "uint8",
                "count": 1,
                "nodata": 255,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(binary, 1)

    if as_cog:
        cogify_geotiff_inplace(output_path)

    logger.debug(
        "Wrote threshold binary raster (cutoff=%s) to %s", cutoff, output_path
    )
    return output_path
