"""
Climate data processing functionality.
"""

from typing import Dict, List, Union, Optional
from pathlib import Path

import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np

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
    target_transform: tuple,
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
    climate_datasets: Dict[str, xr.DataArray],
    output_dir: Union[str, Path]
) -> None:
    """Calculate and log basic statistics for climate datasets."""
    output_dir = Path(output_dir)
    stats_file = output_dir / "climate_statistics.txt"
    
    with open(stats_file, "w") as f:
        f.write("Climate Data Statistics\n")
        f.write("======================\n\n")
        
        for var, data in climate_datasets.items():
            try:
                f.write(f"\n{var.upper()}:\n")
                f.write(f"  Mean: {float(data.mean()):.2f}\n")
                f.write(f"  Std: {float(data.std()):.2f}\n")
                f.write(f"  Min: {float(data.min()):.2f}\n")
                f.write(f"  Max: {float(data.max()):.2f}\n")
            except Exception as e:
                f.write(f"  Error calculating statistics: {e}\n")
                continue 