"""
Climate data processing functionality.
"""

from typing import Dict, List, Union
from pathlib import Path

import xarray as xr
import geopandas as gpd

from .loaders import ClimateData

def fetch_worldclim_datasets(
    variables: List[str],
    boundary_gdf: gpd.GeoDataFrame,
    cache_folder: Union[str, Path]
) -> Dict[str, xr.DataArray]:
    """Fetch WorldClim datasets for specified variables."""
    climate_data = ClimateData(cache_folder=cache_folder)
    datasets = {}
    
    for var in variables:
        try:
            data = climate_data.get_dataset(var, aoi=boundary_gdf)
            datasets[var] = data
        except Exception as e:
            print(f"Error fetching {var}: {e}")
            continue
            
    return datasets

def reproject_climate_datasets(
    datasets: Dict[str, xr.DataArray],
    target_crs: str,
    target_transform,  # Can be Affine or tuple
    target_resolution: float
) -> Dict[str, xr.DataArray]:
    """Reproject climate datasets to target CRS and resolution."""
    reprojected = {}
    
    for var, data in datasets.items():
        try:
            reprojected[var] = data.rio.reproject(
                dst_crs=target_crs,
                transform=target_transform,
                resolution=target_resolution
            )
        except Exception as e:
            print(f"Error reprojecting {var}: {e}")
            continue
            
    return reprojected

def assign_climate_variable_names(
    datasets: Dict[str, xr.DataArray]
) -> Dict[str, xr.DataArray]:
    """Assign descriptive names to climate variables."""
    named_datasets = {}
    
    for var, data in datasets.items():
        try:
            # Add long_name attribute based on variable type
            if var == "bio":
                for i in range(1, 20):
                    if f"bio{i}" in data.dims:
                        data[f"bio{i}"].attrs["long_name"] = f"Bioclimatic variable {i}"
            elif var in ["tmin", "tmax", "tavg"]:
                data.attrs["long_name"] = f"{var.upper()} - {'Minimum' if var == 'tmin' else 'Maximum' if var == 'tmax' else 'Average'} temperature"
            elif var == "prec":
                data.attrs["long_name"] = "Precipitation"
            elif var == "wind":
                data.attrs["long_name"] = "Wind speed"
            elif var == "srad":
                data.attrs["long_name"] = "Solar radiation"
                
            named_datasets[var] = data
        except Exception as e:
            print(f"Error naming {var}: {e}")
            continue
            
    return named_datasets

def write_climate_data(
    climate_datasets: Dict[str, xr.DataArray],
    output_dir: Union[str, Path]
) -> Dict[str, Path]:
    """Write climate datasets to GeoTIFF files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    
    for var, data in climate_datasets.items():
        try:
            output_path = output_dir / f"{var}.tif"
            data.rio.to_raster(output_path)
            output_paths[var] = output_path
        except Exception as e:
            print(f"Error writing {var}: {e}")
            continue
            
    return output_paths

def calculate_climate_statistics(
    temp_average: xr.DataArray,
    precipitation: xr.DataArray,
    wind: xr.DataArray = None
) -> xr.Dataset:
    """Calculate annual climate statistics and return as xarray Dataset.
    
    Args:
        temp_average: Temperature data array (must have 12 bands for monthly data)
        precipitation: Precipitation data array (must have 12 bands for monthly data)
        wind: Wind data array (optional, must have 12 bands if provided)
        
    Returns:
        xarray Dataset containing climate statistics
    """
    # Check array shapes have three dimensions and that temperature array has 12 bands
    if temp_average.ndim != 3 or temp_average.shape[0] != 12:
        raise ValueError("Temperature array must have 12 bands")
    if precipitation.ndim != 3 or precipitation.shape[0] != 12:
        raise ValueError("Precipitation array must have 12 bands")
    if wind is not None and (wind.ndim != 3 or wind.shape[0] != 12):
        raise ValueError("Wind array must have 12 bands if provided")
    
    # Create a dataset with named variables (like the original)
    climate_stats = xr.zeros_like(temp_average[0])
    climate_stats = climate_stats.to_dataset(name="zeros")

    # Calculate temperature statistics
    climate_stats["temp_ann_var"] = temp_average.std(axis=0)
    climate_stats["temp_ann_avg"] = temp_average.mean(axis=0)
    climate_stats["temp_mat_avg"] = temp_average[3:6].mean(axis=0)  # Summer months (April-June)

    # Calculate precipitation statistics
    climate_stats["prec_ann_var"] = precipitation.std(axis=0)
    climate_stats["prec_ann_avg"] = precipitation.mean(axis=0)

    # Calculate wind statistics only if wind data is provided
    if wind is not None:
        climate_stats["wind_ann_var"] = wind.std(axis=0)
        climate_stats["wind_ann_avg"] = wind.mean(axis=0)
    else:
        # Create zero-filled wind statistics if wind data is not available
        climate_stats["wind_ann_var"] = xr.zeros_like(temp_average[0])
        climate_stats["wind_ann_avg"] = xr.zeros_like(temp_average[0])

    climate_stats = climate_stats.drop_vars("zeros")
    return climate_stats 