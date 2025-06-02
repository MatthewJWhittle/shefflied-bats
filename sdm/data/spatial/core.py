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
import topojson as tp
from shapely.geometry.base import BaseGeometry

from scipy.ndimage import generic_filter
import xarray as xr

from sdm.raster.utils import generate_point_grid

logger = logging.getLogger(__name__)

def calculate_coastal_distance(
    geom: BaseGeometry,
    boundary: gpd.GeoDataFrame,
    bounds: tuple,
    resolution: float,
    name: str = "distance",
) -> xr.Dataset:
    """
    Create a grid of points within the boundary and calculate the distance to the geometry.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the points to calculate the distance to.
        boundary (gpd.GeoDataFrame): The boundary to calculate the distance within.
        bounds (tuple): The bounds of the boundary.
        resolution (float): The resolution of the output data.
        name (str): The name of the distance variable.

    Returns:
        xr.DataArray: An xarray DataArray containing the distances.
    """
    if not isinstance(geom, (BaseGeometry)):
        raise ValueError("geom must be a Polygon or MultiPolygon")
    
    points_gdf = generate_point_grid(
        bbox=bounds, resolution=resolution, crs=boundary.crs
    )
    points_gdf.reset_index(drop=True, inplace=True)

    # Calculate the distance to the geometry
    distances = points_gdf.geometry.distance(geom)
    distances.reset_index(drop=True, inplace=True)
    points_gdf[name] = distances
    logging.debug(f"Missing values: {round(points_gdf[name].isna().mean(), 2) * 100}%")

    # Reshape the distances to a grid
    logging.info("Converting distance grid to xarray")
    distance_array = (
        points_gdf.sort_values(["y", "x"])
        .set_index(["y", "x"])
        .to_xarray()
        .rio.write_crs(boundary.crs)
        .drop_vars(["geometry"])
    )

    return distance_array

def create_study_boundary(
    counties_filepath: Path,
    target_crs: str,
    simplify_tolerance: Optional[float] = 100.0,
) -> gpd.GeoDataFrame:
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Filter to just the counties we want
    county_subset = {
        "South Yorkshire": ["Barnsley", "Doncaster", "Rotherham", "Sheffield"],
        "West Yorkshire": ["Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield"],
        "North Yorkshire": ["North Yorkshire", "York"],
        "East Riding of Yorkshire": [
            "East Riding of Yorkshire",
            "Kingston upon Hull, City of",
        ],
    }
    uk_counties = gpd.read_file(counties_filepath)

    if "CTYUA23NM" not in uk_counties.columns:
        raise ValueError("CTYUA23NM column not found in counties file. This is used to match the counties to the subset.")

    # convert to a dataframe
    counties = pd.DataFrame(data={"CTYUA23NM": county_subset.values(), "County" : county_subset.keys()})
    counties = counties.explode("CTYUA23NM").reset_index(drop=True)

    study_area = uk_counties.merge(counties, on="CTYUA23NM", how="inner")

    # check that all counties are present
    missing_counties = set(study_area.CTYUA23NM) - set(counties.CTYUA23NM)
    if len(missing_counties) > 0:
        raise ValueError(f"Missing counties: {missing_counties}")
    
    # Transform to flat projection
    study_area = study_area.to_crs(target_crs)

    # convert tot topojson, simplify, then back to gdf
    logging.info(f"Converting to topojson and simplifying to {simplify_tolerance}m")
    study_area_tp = tp.Topology(study_area, prequantize=False, topology=True)
    study_area_tp.toposimplify(simplify_tolerance)
    study_area = study_area_tp.to_gdf()

    # Merge the geometries
    logging.info("Dissolving geometries by county")
    study_area = study_area.dissolve(by="County", dropna=False, as_index=False)
    study_area = study_area[["County", "geometry"]]

    return study_area


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