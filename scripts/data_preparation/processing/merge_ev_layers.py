import logging
from pathlib import Path
from typing import List, Dict # For type hints

import typer
import xarray as xr
import rioxarray as rxr # For direct open_rasterio with band_as_variable
import numpy as np
from tqdm import tqdm

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform # For boundary and spatial_config
from sdm.utils.text_utils import tidy_variable_name
from sdm.raster.utils import reproject_data, squeeze_dataset
from rasterio.enums import Resampling # For reprojection

app = typer.Typer()

def parse_dataset_input(dataset_inputs: List[str]) -> Dict[str, Path]:
    """Parses list of 'name=path' strings into a dict."""
    parsed_dict = {}
    for item in dataset_inputs:
        try:
            name, path_str = item.split('=', 1)
            parsed_dict[tidy_variable_name(name)] = Path(path_str)
        except ValueError:
            logging.error(f"Invalid dataset input format: '{item}'. Expected 'name=path'. Skipping.")
            # Optionally raise typer.BadParameter here
    return parsed_dict

@app.command()
def main(
    dataset_inputs: List[str] = typer.Option(
        ..., # Ellipsis makes it a required option
        help="List of datasets to merge, each in 'name=path/to/file.tif' format. "
             "Example: --dataset-inputs 'Layer1=./ev1.tif' --dataset-inputs 'Layer2=./ev2.tif'"
    ),
    output_path: Path = typer.Option(
        "data/evs/merged_environmental_variables.tif", 
        help="Path to save the merged multi-band GeoTIFF.",
        writable=True, resolve_path=True
    ),
    boundary_path: Path = typer.Option(
        "data/processed/boundary.geojson", 
        help="Path to the boundary file for clipping the final merged layer.",
        exists=True, readable=True, resolve_path=True
    ),
    boundary_buffer_m: float = typer.Option(0, help="Buffer (meters) for the boundary before final clipping. Set to 0 for no buffer."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Merges multiple raster datasets into a single multi-band GeoTIFF file.
    
    Each input dataset is reprojected to match the project's spatial configuration 
    (derived from the boundary and its associated spatial config, e.g. resolution, CRS).
    Variable names are tidied, and the final merged raster is clipped to the specified boundary.
    """
    setup_logging(verbose=verbose)
    
    datasets_to_merge = parse_dataset_input(dataset_inputs)
    if not datasets_to_merge:
        logging.error("No valid datasets provided to merge. Exiting.")
        raise typer.Exit(code=1)

    logging.info(f"Merging {len(datasets_to_merge)} datasets. Output: {output_path}")

    # Load boundary and project's spatial parameters for reprojection and final clipping
    boundary_gdf, model_transform, _, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=boundary_buffer_m # Buffer for final clip
    )
    target_crs = boundary_gdf.crs # CRS from potentially buffered boundary
    target_resolution = spatial_config["resolution"]

    processed_datasets_to_merge = []

    for dataset_name, dataset_path in tqdm(datasets_to_merge.items(), desc="Processing datasets for merge"):
        logging.info(f"Processing dataset: '{dataset_name}' from {dataset_path}")
        if not dataset_path.exists():
            logging.warning(f"File not found for dataset '{dataset_name}': {dataset_path}. Skipping.")
            continue
        
        try:
            # Open raster, treat bands as separate variables initially if multi-band source
            # masked=True handles nodata, band_as_variable=True for multi-band sources
            data = rxr.open_rasterio(dataset_path, masked=True, band_as_variable=True)
        except Exception as e:
            logging.error(f"Failed to open dataset '{dataset_name}' from {dataset_path}: {e}")
            continue

        # Rename variables based on original band descriptions or to a consistent format
        # Original script had complex renaming logic. Here, we simplify:
        # if input is single band, data var will be 'band_1'. Rename to dataset_name.
        # if multi-band, vars are band_1, band_2. Prefix with dataset_name.
        rename_map = {}
        for var_original_name in list(data.data_vars):
            band_description = data[var_original_name].attrs.get("long_name", var_original_name)
            # Create a new unique name: dataset_name + original band name/description
            new_var_name = tidy_variable_name(f"{dataset_name}_{band_description}")
            if new_var_name == dataset_name and len(data.data_vars) > 1: # Avoid ambiguity for multi-band
                new_var_name = tidy_variable_name(f"{dataset_name}_{var_original_name}") 
            elif len(data.data_vars) == 1: # Single band raster
                new_var_name = dataset_name # Use the provided dataset_name directly
            
            rename_map[var_original_name] = new_var_name
        
        data = data.rename(rename_map)

        # Convert to float32 (common for EVs) and handle nodata
        for var in data.data_vars:
            data[var] = data[var].astype(np.float32)
            # Ensure rio.nodata is set if not already, or convert existing to float32-compatible
            if data[var].rio.nodata is None or np.isnan(data[var].rio.nodata) or abs(data[var].rio.nodata) > 1e30:
                data[var].rio.write_nodata(np.nan, inplace=True) # Use NaN for float32 nodata
            else:
                 data[var].rio.write_nodata(float(data[var].rio.nodata), inplace=True) # Ensure it's float
            data[var].attrs["long_name"] = str(var) # Update long_name to match new var name

        logging.info(f"Reprojecting '{dataset_name}' to target grid...")
        # Reproject to the common model grid
        # Use nearest neighbor if categorical, bilinear for continuous. Assume continuous for merging generic EVs.
        reprojected_data = reproject_data(
            data,
            crs=target_crs,
            transform=model_transform,
            resolution=target_resolution,
            resampling=Resampling.bilinear # Default, suitable for continuous EV data
        )
        processed_datasets_to_merge.append(reprojected_data)

    if not processed_datasets_to_merge:
        logging.error("No datasets were successfully processed to merge. Exiting.")
        raise typer.Exit(code=1)

    logging.info(f"Merging {len(processed_datasets_to_merge)} processed datasets...")
    try:
        merged_ds = xr.merge(processed_datasets_to_merge)
    except xr.MergeError as e:
        logging.error(f"Failed to merge datasets. Check for coordinate or variable name conflicts: {e}")
        # You might want to inspect the datasets in `processed_datasets_to_merge` here.
        # for i, ds_item in enumerate(processed_datasets_to_merge):
        #    logging.debug(f"Dataset {i}: {ds_item.coords} {ds_item.data_vars.keys()}")
        raise typer.Exit(code=1)

    # Final name tidying after merge (should be minimal if done well before)
    final_rename_map = {var: tidy_variable_name(str(var)) for var in merged_ds.data_vars}
    merged_ds = merged_ds.rename(final_rename_map)
    for var in merged_ds.data_vars:
        merged_ds[var].attrs["long_name"] = str(var)
        
    logging.info("Clipping final merged dataset to boundary...")
    # Boundary (boundary_gdf) might have been buffered if boundary_buffer_m > 0
    merged_ds = merged_ds.rio.clip([boundary_gdf.unary_union], crs=boundary_gdf.crs, all_touched=True)

    logging.info(f"Squeezing and saving merged dataset to: {output_path}")
    merged_ds = squeeze_dataset(merged_ds) # Remove any singleton dimensions
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged_ds.rio.to_raster(output_path)
    except Exception as e:
        logging.error(f"Failed to save merged dataset to {output_path}: {e}")
        raise typer.Exit(code=1)

    logging.info("Dataset merging complete.")

if __name__ == "__main__":
    # Example usage, adjust paths as necessary for your environment
    # This CLI would typically be run from the command line.
    # Example: 
    # python scripts/merge_ev_layers.py --dataset-inputs "CEH_LC=data/evs/landcover/ceh_landcover_processed_100m.tif" \
    #                                 --dataset-inputs "VOM_Metrics=data/evs/terrain/vom_summary_metrics_100m.tif" \
    #                                 --output-path "data/evs/merged_demo.tif" \
    #                                 --boundary-path "data/processed/boundary.geojson" -v
    app() 