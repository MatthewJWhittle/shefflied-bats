from pathlib import Path
import logging
from typing import Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from typing import Union

def load_os_shps(datasets:list, dir:str="data/raw/big-files/os-vector-map"):
    dir_path = Path(dir)

    datasets_shp = ["**/*" + keyword + "*.shp" for keyword in datasets]

    dataset_files = [list(dir_path.glob(pattern)) for pattern in datasets_shp]
    
    os_data = []

    for dataset, files in zip(datasets, dataset_files):
        gdfs = [gpd.read_file(file) for file in files]
        gdf = pd.concat(gdfs)
        gdf["dataset"] = dataset
        os_data.append(gdf)

    # Return a dictionary of the data with the name as the key
    return {
        name:data for name, data in zip(datasets, os_data)
    }

def generate_parquets(datasets, dir = "data/processed/os-data", boundary:Union[Polygon, None]=None, overwrite=False):
    """
    This is a helper function for generating parquet files of the OS data. It comes as a shapefile which is very slow to load.
    This process is lazy and will only run if a parquet file doesn't already exist.

    Args:
        datasets (list): A list of datasets to load and generate parquets for
        dir (str): The directory to save the parquets
        boundary (shapely.geometry.Polygon): A boundary polygon with which to filter the data
        overwrite (bool): Whether to overwrite the files or skip them if they already exist. Setting this to false will avoid loading shp data which has already been converted.

    Returns:
        A list of the filepaths to requested parquet files. 

    """
    # Construct the filepaths
    parq_dir = Path(dir)
    out_paths = [parq_dir.joinpath(f"os-{name}.parquet") for name in datasets]
    requested_paths = out_paths.copy()

    # If the user doesn't want to overwrite data, then only load and process datasets without an shp
    if not overwrite:
        datasets = [name for name, path in zip(datasets, out_paths) if not path.exists()]
        out_paths = [path for path in out_paths if not path.exists()]

        if len(datasets) == 0:
            print("All datasets have parquet files already. Set overwrite = True to overwrite them")
            return requested_paths

    shps = load_os_shps(datasets)    

    # If a boundary has been supplied, filter the shps to the boundary
    if boundary:
        for name, gdf in shps.items():
            # Filter the gdf to only those within the boundary
            # Use a spatial index query for speed
            match_indices = gdf.sindex.query(boundary, predicate="intersects")
            shps[name] = gdf.iloc[match_indices]

    # Write the gdfs out as parquet files
    for gdf,path in zip(list(shps.values()), out_paths):
        gdf.to_parquet(path)

    return requested_paths


def process_roads(
    roads_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split roads into major and minor categories.

    Major roads include motorways and A roads, while minor roads include all others.

    Args:
        roads_gdf: GeoDataFrame containing road features with 'CLASSIFICA' column.

    Returns:
        Tuple containing (major_roads, minor_roads) as GeoDataFrames.
    """
    logging.info("Processing roads classification")
    road_classes = roads_gdf["CLASSIFICA"].value_counts()
    major_roads = ["Motorway", "A Road"]
    pattern = "|".join([f"{road_type}*" for road_type in major_roads])

    road_classes = road_classes.to_frame(name="count").reset_index(names=["CLASSIFICA"])
    road_classes["major_road"] = road_classes.CLASSIFICA.str.contains(
        pattern, regex=True
    )
    road_classes.drop(columns=["count"], inplace=True)

    roads = roads_gdf.merge(road_classes, on="CLASSIFICA", how="left")

    roads.major_road.fillna(False, inplace=True)
    roads["major_road"] = roads.major_road.astype(bool)

    major_roads = gpd.GeoDataFrame(roads[roads.major_road])
    minor_roads = gpd.GeoDataFrame(roads[~roads.major_road])
    logging.info("Classified %d major roads and %d minor roads", 
                 len(major_roads), len(minor_roads))
    return major_roads, minor_roads
