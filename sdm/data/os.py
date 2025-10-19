from pathlib import Path
import logging
from typing import Tuple, Union, List, Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon

def load_os_shps(
    datasets: List[str], 
    dir: Union[str, Path] = "data/raw/big-files/os-vector-map"
) -> Dict[str, gpd.GeoDataFrame]:
    """Load Ordnance Survey shapefiles for specified datasets.

    Args:
        datasets: List of dataset names to load (e.g., ["Building", "Water"]).
        dir: Directory containing OS data organized in subdirectories.

    Returns:
        Dictionary mapping dataset names to their corresponding GeoDataFrames.

    Raises:
        FileNotFoundError: If required shapefiles are not found.
    """
    logging.info("Loading OS shapefiles from %s", dir)
    dir_path = Path(dir)
    datasets_shp = [f"**/*{keyword}*.shp" for keyword in datasets]
    dataset_files = [list(dir_path.glob(pattern)) for pattern in datasets_shp]

    os_data = []
    for dataset, files in zip(datasets, dataset_files):
        logging.debug("Loading %d files for dataset %s", len(files), dataset)
        gdfs = [gpd.read_file(file) for file in files]
        gdf = gpd.GeoDataFrame(pd.concat(gdfs))
        gdf["dataset"] = dataset
        logging.info("Loaded %d features for dataset %s", len(gdf), dataset)
        os_data.append(gdf)

    return {name: data for name, data in zip(datasets, os_data)}


def generate_parquets(
    datasets: List[str],
    dir: str = "data/processed/os-data",
    boundary: Union[Polygon, None] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Generate parquet files for OS data.
    
    Args:
        datasets: List of dataset names to load and generate parquets for
        dir: Directory to save the parquets
        boundary: Boundary polygon with which to filter the data
        overwrite: Whether to overwrite existing files
        
    Returns:
        List of filepaths to requested parquet files
    """
    logging.info("Generating parquet files in %s", dir)
    parq_dir = Path(dir)
    out_paths = [parq_dir / f"os-{name}.parquet" for name in datasets]
    requested_paths = out_paths.copy()

    if not overwrite:
        datasets = [
            name for name, path in zip(datasets, out_paths) if not path.exists()
        ]
        out_paths = [path for path in out_paths if not path.exists()]

        if not datasets:
            logging.info("All datasets have parquet files already")
            return requested_paths

    shps = load_os_shps(datasets)

    if boundary:
        logging.info("Filtering data to boundary")
        for name, gdf in shps.items():
            original_len = len(gdf)
            match_indices = gdf.sindex.query(boundary, predicate="intersects")
            shps[name] = gdf.iloc[match_indices]
            logging.info("Filtered %s: %d → %d features", name, original_len, len(shps[name]))

    for gdf, path in zip(shps.values(), out_paths):
        logging.info("Saving parquet file to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
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

    roads = roads.assign(major_road=roads.major_road.fillna(False))
    roads["major_road"] = roads.major_road.astype(bool)

    major_roads = gpd.GeoDataFrame(roads[roads.major_road])
    minor_roads = gpd.GeoDataFrame(roads[~roads.major_road])
    logging.info("Classified %d major roads and %d minor roads", 
                 len(major_roads), len(minor_roads))
    return major_roads, minor_roads
