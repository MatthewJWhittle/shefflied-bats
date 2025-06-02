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
            logging.error(f"Invalid dataset input format: '{item}'. Expected 'name=path'. Skipping.")
    return parsed_dict

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

    logging.info(f"Merging {len(datasets_to_merge)} datasets. Output: {output_path}")

    # Load boundary and project's spatial parameters for reprojection and final clipping
    boundary_gdf, model_transform, _, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=boundary_buffer_m
    )
    target_crs = boundary_gdf.crs
    target_resolution = spatial_config["resolution"]

    processed_datasets_to_merge = []

    for dataset_name, dataset_path in tqdm(datasets_to_merge.items(), desc="Processing datasets for merge"):
        logging.info(f"Processing dataset: '{dataset_name}' from {dataset_path}")
        if not dataset_path.exists():
            logging.warning(f"File not found for dataset '{dataset_name}': {dataset_path}. Skipping.")
            continue
        
        try:
            data = rxr.open_rasterio(dataset_path, masked=True, band_as_variable=True)
        except Exception as e:
            logging.error(f"Failed to open dataset '{dataset_name}' from {dataset_path}: {e}")
            continue

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

        logging.info(f"Reprojecting '{dataset_name}' to target grid...")
        reprojected_data = reproject_data(
            data,
            crs=target_crs,
            transform=model_transform,
            resolution=target_resolution,
            resampling=Resampling.bilinear
        )
        processed_datasets_to_merge.append(reprojected_data)

    if not processed_datasets_to_merge:
        raise ValueError("No datasets were successfully processed to merge.")

    logging.info(f"Merging {len(processed_datasets_to_merge)} processed datasets...")
    try:
        merged_ds = xr.merge(processed_datasets_to_merge)
    except xr.MergeError as e:
        logging.error(f"Failed to merge datasets. Check for coordinate or variable name conflicts: {e}")
        raise

    # Final name tidying after merge
    final_rename_map = {var: tidy_variable_name(str(var)) for var in merged_ds.data_vars}
    merged_ds = merged_ds.rename(final_rename_map)
    for var in merged_ds.data_vars:
        merged_ds[var].attrs["long_name"] = str(var)
        
    logging.info("Clipping final merged dataset to boundary...")
    merged_ds = merged_ds.rio.clip([boundary_gdf.unary_union], crs=boundary_gdf.crs, all_touched=True)

    logging.info(f"Squeezing and saving merged dataset to: {output_path}")
    merged_ds = squeeze_dataset(merged_ds)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged_ds.rio.to_raster(output_path)
    except Exception as e:
        logging.error(f"Failed to save merged dataset to {output_path}: {e}")
        raise

    logging.info("Dataset merging complete.")

