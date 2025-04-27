from typing import Union, Mapping
from pathlib import Path
import logging
import re

import xarray as xr
import rioxarray
from tqdm import tqdm
from rasterio import float32 as rio_float32

from data_prep.generate_evs.ingestion.geo_utils import reproject_data, squeeze_dataset
from data_prep.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)



def tidy_name(name: str) -> str:
    """
    Cleans up the name of the variable by replacing dashes and spaces with underscores, 
    converting to lowercase, and stripping whitespace.

    Args:
        name: The name of the variable

    Returns:
        The cleaned up name
    """
    # Replace dashes and spaces with underscores
    name = re.sub(r'[- ]', '_', name)
    # Convert to lowercase
    name = name.lower()
    # Strip whitespace
    name = name.strip()
    # Remove any leading or trailing underscores
    name = name.strip('_')
    # Remove any trailing underscores
    name = name.rstrip('_')
    return name

def main(
    datasets: Mapping[str, Union[str, Path]],
    output_path: Union[str, Path],
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
) -> Path:
    """
    Merges multiple datasets into a single multiband tiff file.

    Args:
        datasets: List of dictionaries, each with "name" and "path" keys
        output_path: Path to save the merged multiband tiff

    Returns:
        Path to the merged dataset
    """
    logging.info("Merging datasets...\n %s", str(datasets))

    # Load the boundary and spatial configuration
    boundary = load_boundary(boundary_path, buffer_distance=7_000)
    bounds = tuple(boundary.total_bounds)
    spatial_config = load_spatial_config()
    model_transform, _ = construct_transform_shift_bounds(
        bounds, spatial_config["resolution"]
    )
    output_path = Path(output_path)

    xr_datasets = []

    # Process each input dataset
    for name, path in tqdm(datasets.items(), desc="Processing datasets"):
        logging.info("Processing %s data...", name)

        # Load the dataset
        data: Union[xr.DataArray, xr.Dataset] = rioxarray.open_rasterio(
            path, 
            band_as_variable=True,
            masked=True,
            # Use a float32-compatible nodata value
            nodata=-9999.0
        )  # type: ignore

        # Ensure nodata value is float32 compatible for all variables
        for var in data.data_vars:
            if data[var].rio.nodata is not None and abs(data[var].rio.nodata) > 3.4e38:
                data[var].rio.write_nodata(-9999.0)

        # assign long_name to the data variable
        name_mapping =  {
            name: data[name].attrs["long_name"] for name in data.data_vars
            }
        name_mapping = {k: f"{name}_{v}" for k, v in name_mapping.items()}

        data = data.rename(name_mapping)
        # update long_name attribute
        for var in data.data_vars:
            data[var].attrs["long_name"] = str(var)
        
        # convert dtypes to float16 to save space
        for var in data.data_vars:
            data[var] = data[var].astype(rio_float32)

        # Reproject the data to the model grid
        logging.info("Reprojecting %s data...", name)
        data = reproject_data(
            data,
            crs=spatial_config["crs"],
            transform=model_transform,
            resolution=spatial_config["resolution"],
        )

        xr_datasets.append(data)

    # Merge all datasets
    logging.info("Merging %s datasets...", len(xr_datasets))
    merged_ds = xr.merge(xr_datasets)

    # Tidy up the names
    logging.info("Tidying up variable names...")
    name_mapping = {
        var: tidy_name(str(var)) for var in merged_ds.data_vars
    }
    merged_ds = merged_ds.rename(name_mapping)
    # update long_name attribute
    for var in merged_ds.data_vars:
        merged_ds[var].attrs["long_name"] = str(var)
        
    # clip to the boundary
    logging.info("Clipping to boundary...")
    merged_ds = merged_ds.rio.clip([boundary.unary_union], crs=boundary.crs)

    # Write to output file
    merged_ds = squeeze_dataset(merged_ds)
    merged_ds.rio.to_raster(output_path)

    return output_path



if __name__ == "__main__":
    main(
        datasets={
            "ceh_landcover" : "data/evs/ceh-land-cover-100m.tif",
            "vom" : "data/evs/vom_100m.tif",
            "terrain_stats" : "data/evs/terrain_stats.tif",
            "terrain" : "data/evs/dtm_dsm_100m.tif",
            "os_cover" : "data/evs/os-feature-cover.tif",
            "os-distance" : "data/evs/os-distance-to-feature.tif",
            "climate_stats" : "data/evs/climate_stats.tif",
            "climate_bioclim" : "data/evs/bioclim.tif",
            "bgs_coast": "data/evs/coastal_distance.tif",
            },
        output_path="data/evs/all-evs.tif",

    )