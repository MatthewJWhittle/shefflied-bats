from pathlib import Path
from typing import Union
import logging

import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry.base import BaseGeometry

from src.ingestion.geo_utils import reproject_data, squeeze_dataset, generate_point_grid
from src.utils.load import load_boundary_and_transform
from src.utils.config import setup_logging


def calculate_distance(
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
    logging.debug("Missing values: %d%", round(points_gdf[name].isna().mean(), 2) * 100)

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
    boundary, model_transform, bounds, spatial_config = load_boundary_and_transform(
        boundary_path
    )

    logging.info("Loading coastal data...")
    coast_gdf = gpd.read_file(
        "data/raw/big-files/BGS GeoCoast/GeoCoast_v1_Authority_Area_Inundation.shp"
    )
    coast_gdf = coast_gdf.to_crs(boundary.crs.to_epsg())

    logging.info("Dissolving...")
    ## Simplification makes it faster
    simplify_tolerance = 1000
    ## Dissolve the coastal data
    coast_gdf = coast_gdf.dissolve(dropna=False)
    
    logging.info("Simplifying...")
    ## simplify the coastal data to speed up the process
    coast_gdf["geometry"] = coast_gdf.simplify(simplify_tolerance)

    ## buffer the coast and calculate a difference to create a 'sea' zone
    logging.info("Create sea zone...")
    sea = (
        coast_gdf.buffer(10_000)
        .simplify(simplify_tolerance)
        .difference(coast_gdf.geometry)
    )
    # then explode the polygon and drop anything with an area less than 1km^2
    sea = sea.explode().to_frame()
    sea = sea[sea.area > 1_000_000]
    sea_polygon = sea.unary_union

    ## calculate the distance to the coast
    logging.info("Calculating distances...")
    coastal_distance_xr = calculate_distance(
        sea_polygon,
        boundary,
        bounds,
        # Calculate the distance to the coast at 10x resolution to speed things up
        resolution=spatial_config["resolution"] * 10,
        name="distance_to_coast",
    )

    ## reproject the data
    logging.info("Reprojecting data...")
    coastal_distance_xr: xr.Dataset = reproject_data(
        coastal_distance_xr, boundary.crs, model_transform, spatial_config["resolution"]
    )  # type: ignore

    ## squeeze the data
    logging.info("Squeezing data...")
    coastal_distance_xr = squeeze_dataset(coastal_distance_xr)

    # mask the data to the boundary
    logging.info("Masking data...")
    coastal_distance_xr = coastal_distance_xr.rio.clip(
        [boundary.unary_union], crs=boundary.crs
    )

    ## save the data
    logging.info("Saving data...")
    output_path = Path(output_dir) / "coastal_distance.tif"
    coastal_distance_xr.rio.to_raster(output_path)

    return output_path


if __name__ == "__main__":
    main(
        boundary_path="data/processed/boundary.geojson",
        output_dir="data/evs",
        debug=True,
    )
