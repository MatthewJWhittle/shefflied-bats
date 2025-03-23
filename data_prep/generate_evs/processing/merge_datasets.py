from typing import Union
from pathlib import Path
import logging

import xarray as xr
import rioxarray
from tqdm import tqdm
from rasterio import float32 as rio_float32

from generate_evs.ingestion.geo_utils import reproject_data, squeeze_dataset
from generate_evs.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)


def main(
    datasets: dict[str, Union[str, Path]],
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
    boundary = load_boundary(boundary_path, buffer_distance=0)
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
        data: Union[xr.DataArray, xr.Dataset] = rioxarray.open_rasterio(path, band_as_variable=True)  # type: ignore


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
            "os_cover" : "data/evs/os-feature-cover-100m.tif",
            "os-distance" : "data/evs/os-distance-to-feature.tif",
            "climate" : "data/evs/climate_stats.tif",
            "bgs-coast": "data/evs/coastal_distance.tif",
            },
        output_path="data/evs/all-evs.tif",

    )