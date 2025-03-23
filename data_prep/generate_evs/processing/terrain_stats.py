from pathlib import Path
from typing import Union, Tuple
import logging

import numpy as np
import xarray as xr
import richdem as rd
import rioxarray as rxr

from generate_evs.utils.config import setup_logging
from generate_evs.ingestion.geo_utils import squeeze_dataset


def calculate_slope_aspect(dem_rd: rd.rdarray) -> Tuple[rd.rdarray, rd.rdarray]:
    """
    Calculate slope and aspect from a RichDEM array.
    
    Args:
        dem_rd: RichDEM array containing elevation data
        
    Returns:
        Tuple containing slope (radians) and aspect arrays
    """
    slope = rd.TerrainAttribute(dem_rd, attrib='slope_radians')
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')
    return slope, aspect


def calculate_aspect_components(aspect: rd.rdarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate eastness and northness components from aspect.
    
    Args:
        aspect: RichDEM array containing aspect data (in radians)
        
    Returns:
        Tuple containing eastness and northness arrays
    """
    aspect_eastness = np.cos(aspect.data)
    aspect_northness = np.sin(aspect.data)
    return aspect_eastness, aspect_northness


def calculate_twi(dem_rd: rd.rdarray, slope_array: np.ndarray) -> np.ndarray:
    """
    Calculate Topographic Wetness Index.
    
    Args:
        dem_rd: RichDEM array containing elevation data
        slope_array: Numpy array containing slope data (in radians)
        
    Returns:
        Numpy array containing TWI values
    """
    # Ensure slope has no zeros to avoid division issues
    slope_safe = slope_array.copy()
    slope_safe[slope_safe < 0.001] = 0.001
    
    # Calculate flow accumulation
    flow_acc = rd.FlowAccumulation(dem_rd, method='D8')
    
    # Calculate TWI
    twi = np.log(flow_acc.data / slope_safe)
    return twi


def calculate_curvature(dem_rd: rd.rdarray) -> rd.rdarray:
    """
    Calculate terrain curvature.
    
    Args:
        dem_rd: RichDEM array containing elevation data
        
    Returns:
        RichDEM array containing curvature data
    """
    return rd.TerrainAttribute(dem_rd, attrib='curvature')


def calculate_roughness(slope_da: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    """
    Calculate terrain roughness as the standard deviation of slope within a moving window.
    
    Args:
        slope_da: DataArray containing slope data
        window_size: Size of the moving window for calculating roughness
        
    Returns:
        DataArray containing roughness values
    """
    return slope_da.rolling(x=window_size, y=window_size, center=True).std()


def calculate_tpi(dem_da: xr.DataArray, window_size: int = 3) -> xr.DataArray:
    """
    Calculate Topographic Position Index as the difference between elevation
    and mean elevation within a moving window.
    
    Args:
        dem_da: DataArray containing elevation data
        window_size: Size of the moving window for calculating TPI
        
    Returns:
        DataArray containing TPI values
    """
    return dem_da - dem_da.rolling(x=window_size, y=window_size).mean()


def calculate_weighted_aspect(slope_da: xr.DataArray, aspect_eastness_da: xr.DataArray, 
                              aspect_northness_da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Weight aspect components by slope to emphasize aspect in steeper areas.
    
    Args:
        slope_da: DataArray containing slope data
        aspect_eastness_da: DataArray containing eastness component
        aspect_northness_da: DataArray containing northness component
        
    Returns:
        Tuple containing weighted eastness and northness DataArrays
    """
    weighted_eastness = slope_da * aspect_eastness_da
    weighted_northness = slope_da * aspect_northness_da
    return weighted_eastness, weighted_northness


def process_terrain_stats(
    dem_path: Union[str, Path],
    dem_band: int = 1,
    window_size: int = 3,
) -> xr.Dataset:
    """
    Process DEM to calculate and compile terrain statistics.
    
    Args:
        dem_path: Path to the input DEM file
        output_path: Path where output should be saved (if None, output is not saved)
        boundary_path: Path to boundary file (optional)
        window_size: Size of the window for neighborhood calculations
        visualize: Whether to generate and display visualization plots
        
    Returns:
        Dataset containing calculated terrain statistics
    """
    # Load the DEM using rioxarray
    dem_rxr :xr.DataArray = rxr.open_rasterio(dem_path) # type: ignore
    dem_rxr = dem_rxr.isel(band=dem_band-1)
    dem_rd = rd.rdarray(dem_rxr.values, no_data=dem_rxr.rio.nodata)
    
    # Convert to dataset with 'dem' variable
    terrain = dem_rxr.to_dataset(name="dem")
    
    # Calculate terrain attributes
    slope, aspect = calculate_slope_aspect(dem_rd)
    aspect_eastness, aspect_northness = calculate_aspect_components(aspect)
    twi = calculate_twi(dem_rd, np.array(slope, copy=False))
    curvature = calculate_curvature(dem_rd)
    
    # Add calculated attributes to the dataset
    terrain["slope"] = (("y", "x"), slope.data)
    terrain["aspect_eastness"] = (("y", "x"), aspect_eastness)
    terrain["aspect_northness"] = (("y", "x"), aspect_northness)
    terrain["twi"] = (("y", "x"), twi)
    terrain["curvature"] = (("y", "x"), curvature.data)
    
    # Calculate neighborhood-based statistics
    terrain["roughness"] = calculate_roughness(terrain.slope, window_size)
    terrain["tpi"] = calculate_tpi(terrain.dem, window_size)
    
    # Calculate slope-weighted aspect
    weighted_eastness, weighted_northness = calculate_weighted_aspect(
        terrain.slope, terrain.aspect_eastness, terrain.aspect_northness
    )
    terrain["aspect_eastness_slope"] = weighted_eastness
    terrain["aspect_northness_slope"] = weighted_northness
    
    return terrain



def save_terrain_stats(
    terrain: xr.Dataset, 
    output_path: Union[str, Path],
) -> Path:
    """
    Save terrain statistics to file.
    
    Args:
        terrain: Dataset containing terrain statistics
        output_path: Path where output should be saved
        drop_dem: Whether to drop the DEM from the output (to avoid duplication)
        
    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Drop the DEM to avoid duplication if requested
    terrain_stats = terrain.drop_vars("dem")
    
    terrain_stats = squeeze_dataset(terrain_stats)
    # Write to file
    terrain_stats.rio.to_raster(output_path)
    
    return output_path


def main(
    dem_path: Union[str, Path],
    output_path: Union[str, Path],
    dem_band: int = 1,
    window_size: int = 3,
) -> Path:
    """
    Calculate terrain statistics from a DEM and save results.
    
    Args:
        dem_path: Path to the input DEM file
        output_path: Path where output should be saved (defaults to 'terrain-stats.tif' in same dir as DEM)
        boundary_path: Path to boundary file (optional)
        window_size: Size of the window for neighborhood calculations
        visualize: Whether to generate and display visualization plots
        drop_dem: Whether to drop the DEM from the output (to avoid duplication)
        
    Returns:
        Path to the saved terrain statistics file
    """
    setup_logging()
    logging.info(f"Calculating terrain statistics for {dem_path}")
    
    # Process terrain statistics
    terrain = process_terrain_stats(
        dem_path=dem_path,
        window_size=window_size,
        dem_band=dem_band,
    )
    
    # Save results
    logging.info(f"Saving terrain statistics to {output_path}")
    result_path = save_terrain_stats(terrain, output_path)
    
    logging.info("Terrain statistics processing complete")
    return result_path


if __name__ == "__main__":
    main(
        dem_path="data/evs/dtm_dsm_100m.tif",
        output_path="data/evs/terrain_stats.tif",
        window_size=3,
        dem_band=1,
    )
