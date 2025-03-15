from pathlib import Path
from typing import Union
import logging

import geopandas as gpd
import xarray as xr
import rioxarray as rxr

from src.ingestion.geo_utils import reproject_data, squeeze_dataset, calculate_distances
from src.utils.load import load_boundary_and_transform
from src.utils.config import setup_logging



def main(
        boundary_path: Union[Path, str],
        output_dir: Union[Path, str],
        debug: bool = False,
) -> Path:
    """
    Main function for creating the coastal distance dataset.

    Data from:  https://www.bgs.ac.uk/download/bgs-geocoast-open/

    """
    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)
    logging.info("Creating coastal distance dataset...")
    boundary, model_transform, bounds, spatial_config = load_boundary_and_transform(boundary_path)

    logging.info("Loading coastal data...")
    coastal_data = gpd.read_file("data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp")
    coastal_data = coastal_data.to_crs(boundary.crs.to_epsg())

    logging.info("Dissolving...")
    simplify_tolerance = 100
    ## Dissolve the coastal data
    coastal_data = coastal_data.dissolve(dropna=False)

    logging.info("Simplifying...")
    ## simplify the coastal data to speed up the process
    coastal_data["geometry"] = coastal_data.simplify(simplify_tolerance)

    ## buffer the coast and calculate a difference to create a 'sea' zone
    logging.info("Create sea zone...")
    sea = coastal_data.buffer(10_000).simplify(simplify_tolerance).difference(coastal_data.geometry)
    sea_gdf = gpd.GeoDataFrame(geometry=sea, crs=coastal_data.crs)

    ## calculate the distance to the coast
    logging.info("Calculating distances...")
    coastal_distance = calculate_distances({"coast": sea_gdf}, boundary, resolution=spatial_config["resolution"])

    ## reproject the data
    logging.info("Reprojecting data...")
    coastal_distance : xr.Dataset = reproject_data(
        coastal_distance, boundary.crs, model_transform, spatial_config["resolution"]
    ) # type: ignore

    ## squeeze the data
    logging.info("Squeezing data...")
    coastal_distance = squeeze_dataset(coastal_distance)

    ## save the data
    logging.info("Saving data...")
    output_path = Path(output_dir) / "coastal_distance.tif"
    coastal_data.rio.to_raster(output_path)

    return output_path


if __name__ == "__main__":
    main(
        boundary_path="data/processed/boundary.geojson",
        output_dir="data/evs",
    )