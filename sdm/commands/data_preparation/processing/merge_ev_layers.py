import logging
from pathlib import Path
from typing import List, Dict

import xarray as xr
import rioxarray as rxr
import numpy as np
from tqdm import tqdm
from rasterio.enums import Resampling

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.utils.text_utils import tidy_variable_name
from sdm.raster.utils import reproject_data, squeeze_dataset

def parse_dataset_input(dataset_inputs: List[str]) -> Dict[str, Path]:
    """Parses list of 'name=path' strings into a dict."""
    parsed_dict = {}
    for item in dataset_inputs:
        try:
            name, path_str = item.split('=', 1)
            parsed_dict[tidy_variable_name(name)] = Path(path_str)
        except ValueError:
            logging.error("Invalid dataset input format: '%s'. Expected 'name=path'. Skipping.", item)
    return parsed_dict


def load_and_preprocess_dataset(dataset_name: str, dataset_path: Path) -> xr.Dataset:
    """
    Load and preprocess a single dataset for merging.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset file
        
    Returns:
        Preprocessed xarray Dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        Exception: If the dataset cannot be opened
    """
    logging.info("Processing dataset: '%s' from %s", dataset_name, dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"File not found for dataset '{dataset_name}': {dataset_path}")
    
    try:
        data = rxr.open_rasterio(dataset_path, masked=True, band_as_variable=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open dataset '{dataset_name}' from {dataset_path}: {e}") from e

    # Rename variables based on original band descriptions or to a consistent format
    rename_map = {}
    for var_original_name in list(data.data_vars):
        band_description = data[var_original_name].attrs.get("long_name", var_original_name)
        new_var_name = tidy_variable_name(f"{dataset_name}_{band_description}")
        if new_var_name == dataset_name and len(data.data_vars) > 1:
            new_var_name = tidy_variable_name(f"{dataset_name}_{var_original_name}")
        elif len(data.data_vars) == 1:
            new_var_name = dataset_name
        
        rename_map[var_original_name] = new_var_name
    
    data = data.rename(rename_map)

    # Convert to float32 and handle nodata
    for var in data.data_vars:
        data[var] = data[var].astype(np.float32)
        if data[var].rio.nodata is None or np.isnan(data[var].rio.nodata) or abs(data[var].rio.nodata) > 1e30:
            data[var].rio.write_nodata(np.nan, inplace=True)
        else:
            data[var].rio.write_nodata(float(data[var].rio.nodata), inplace=True)
        data[var].attrs["long_name"] = str(var)
    
    return data


def reproject_dataset(data: xr.Dataset, target_crs: str, model_transform, target_resolution: int) -> xr.Dataset:
    """
    Reproject a dataset to the target coordinate system and resolution.
    
    Args:
        data: Input dataset
        target_crs: Target coordinate reference system
        model_transform: Target transform
        target_resolution: Target resolution in meters
        
    Returns:
        Reprojected dataset
    """
    logging.info("Reprojecting dataset to target grid...")
    return reproject_data(
        data,
        crs=target_crs,
        transform=model_transform,
        resolution=target_resolution,
        resampling=Resampling.bilinear
    )


def merge_datasets(processed_datasets: List[xr.Dataset]) -> xr.Dataset:
    """
    Merge multiple processed datasets into a single dataset.
    
    Args:
        processed_datasets: List of processed datasets to merge
        
    Returns:
        Merged dataset
        
    Raises:
        xr.MergeError: If datasets cannot be merged due to conflicts
    """
    logging.info("Merging %d processed datasets...", len(processed_datasets))
    try:
        merged_ds = xr.merge(processed_datasets)
    except xr.MergeError as e:
        logging.error("Failed to merge datasets. Check for coordinate or variable name conflicts: %s", e)
        raise
    
    # Final name tidying after merge
    final_rename_map = {var: tidy_variable_name(str(var)) for var in merged_ds.data_vars}
    merged_ds = merged_ds.rename(final_rename_map)
    for var in merged_ds.data_vars:
        merged_ds[var].attrs["long_name"] = str(var)
    
    return merged_ds


def clip_and_save_dataset(merged_ds: xr.Dataset, boundary_gdf, output_path: Path) -> None:
    """
    Clip the merged dataset to the boundary and save to file.
    
    Args:
        merged_ds: Merged dataset to clip and save
        boundary_gdf: Boundary GeoDataFrame for clipping
        output_path: Path to save the final dataset
        
    Raises:
        Exception: If the dataset cannot be saved
    """
    logging.info("Clipping final merged dataset to boundary...")
    merged_ds = merged_ds.rio.clip([boundary_gdf.union_all()], crs=boundary_gdf.crs, all_touched=True)

    logging.info("Squeezing and saving merged dataset to: %s", output_path)
    merged_ds = squeeze_dataset(merged_ds)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged_ds.rio.to_raster(output_path)
    except Exception as e:
        logging.error("Failed to save merged dataset to %s: %s", output_path, e)
        raise

def merge_ev_layers(
    dataset_inputs: List[str],
    output_path: Path = Path("data/evs/merged_environmental_variables.tif"),
    boundary_path: Path = Path("data/processed/boundary.geojson"),
    boundary_buffer_m: float = 0,
    verbose: bool = False
) -> None:
    """
    Core function to merge multiple raster datasets into a single multi-band GeoTIFF file.
    Can be called from other scripts or notebooks.

    Args:
        dataset_inputs: List of datasets to merge, each in 'name=path/to/file.tif' format
        output_path: Path to save the merged multi-band GeoTIFF
        boundary_path: Path to the boundary file for clipping the final merged layer
        boundary_buffer_m: Buffer (meters) for the boundary before final clipping
        verbose: Enable verbose logging
    """
    setup_logging(verbose=verbose)
    
    datasets_to_merge = parse_dataset_input(dataset_inputs)
    if not datasets_to_merge:
        raise ValueError("No valid datasets provided to merge.")

    logging.info("Merging %d datasets. Output: %s", len(datasets_to_merge), output_path)

    # Load boundary and project's spatial parameters for reprojection and final clipping
    boundary_gdf, model_transform, _, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=boundary_buffer_m
    )
    target_crs = boundary_gdf.crs
    target_resolution = spatial_config["resolution"]

    processed_datasets_to_merge = []

    for dataset_name, dataset_path in tqdm(datasets_to_merge.items(), desc="Processing datasets for merge"):
        try:
            # Load and preprocess the dataset
            data = load_and_preprocess_dataset(dataset_name, dataset_path)
            
            # Reproject the dataset
            reprojected_data = reproject_dataset(data, target_crs, model_transform, target_resolution)
            processed_datasets_to_merge.append(reprojected_data)
            
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning("Skipping dataset '%s': %s", dataset_name, e)
            continue

    if not processed_datasets_to_merge:
        raise ValueError("No datasets were successfully processed to merge.")

    # Merge all processed datasets
    merged_ds = merge_datasets(processed_datasets_to_merge)
    
    # Clip and save the final dataset
    clip_and_save_dataset(merged_ds, boundary_gdf, output_path)

    logging.info("Dataset merging complete.")

