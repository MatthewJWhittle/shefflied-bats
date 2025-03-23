import logging
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from scipy.spatial import cKDTree
from shapely.geometry import box, Polygon
import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds as transform_from_bounds

from generate_evs.ingestion.geo_utils import (
    reproject_data,
    calculate_distances,
    )
from generate_evs.utils.config import setup_logging
from generate_evs.utils.load import (
    load_boundary,
    load_spatial_config,
    construct_transform_shift_bounds,
)


def rasterise_gdf(
    gdf: gpd.GeoDataFrame, 
    resolution: float, 
    output_file: Union[str, Path], 
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Path:
    """Rasterize a GeoDataFrame to a GeoTIFF file.

    Args:
        gdf: GeoDataFrame containing geometries to rasterize.
        resolution: Pixel size in the GeoDataFrame's CRS units.
        output_file: Path where the output GeoTIFF will be saved.
        bbox: Optional bounding box (minx, miny, maxx, maxy) to limit rasterization.

    Returns:
        Path to the created raster file.

    Raises:
        ValueError: If GeoDataFrame is empty or resolution is invalid.
    """
    # Define the raster size and transform
    # Here, I'm assuming a 1x1 meter resolution and using the bounds of the GeoDataFrame
    if bbox is None:
        x_min, y_min, x_max, y_max = gdf.total_bounds
    else:
        x_min, y_min, x_max, y_max = bbox

    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    transform = transform_from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Create a mask: rasterize the GeoDataFrame. This gives a value of True where the geometry covers a square
    mask = geometry_mask(
        gdf.geometry, transform=transform, invert=True, out_shape=(height, width)
    )
    # Convert the boolean mask to uint8 (or another supported data type)
    mask = mask.astype("uint8")

    # Write the mask to a raster file
    with rio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs=gdf.crs,
        transform=transform,
        nodata=255,
    ) as dest:
        dest.write(mask.astype(rio.uint8), 1)

    return Path(output_file)


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
        # add bounding box columns to use for filtering in parquet loading
        gdf = pd.concat([gdf, gdf.geometry.bounds], axis=1)
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
    """Generate parquet files for OS data."""
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

    roads.major_road.fillna(False, inplace=True)
    roads["major_road"] = roads.major_road.astype(bool)

    major_roads = gpd.GeoDataFrame(roads[roads.major_road])
    minor_roads = gpd.GeoDataFrame(roads[~roads.major_road])
    logging.info("Classified %d major roads and %d minor roads", 
                 len(major_roads), len(minor_roads))
    return major_roads, minor_roads


def calculate_feature_cover(
    feature_gdfs: Dict[str, gpd.GeoDataFrame],
    boundary: gpd.GeoDataFrame,
    target_resolution: int = 100
) -> xr.Dataset:
    """Calculate feature coverage density for each feature type.

    Args:
        feature_gdfs: Dictionary of feature name to GeoDataFrame mappings.
        boundary: GeoDataFrame containing the area of interest.
        target_resolution: Resolution in meters for the output raster.

    Returns:
        xarray Dataset containing coverage density for each feature type.

    Note:
        Coverage is calculated by rasterizing features at 1m resolution,
        then aggregating to target resolution using sum.
    """
    logging.info("Calculating feature cover at %dm resolution", target_resolution)

    boundary_union = boundary.unary_union
    base_resolution = 10  # 10m resolution for rasterization
    scale_factor = target_resolution // base_resolution
    def calculate_cover(gdf: gpd.GeoDataFrame, name: str) -> xr.Dataset:
        logging.debug("Processing cover for %s (%d features)", name, len(gdf))
        with NamedTemporaryFile() as f:
            rasterise_gdf(
                gdf, resolution=base_resolution, output_file=f.name, bbox=boundary_union.bounds
            )
            cover : xr.Dataset = rxr.open_rasterio(f.name, chunks="auto")
            cover_area = cover.coarsen(
                x=scale_factor, y=scale_factor, boundary="trim"
            ).sum()
            # log the % NA values
            logging.debug("NA values for %s: %.2f%%", name,
                          100 * cover_area.isnull().mean())
            logging.debug
        return cover_area.to_dataset(name=name)

    cover_datasets = []
    for name, gdf in feature_gdfs.items():
        logging.info("Calculating cover for %s", name)
        cover_datasets.append(calculate_cover(gdf, name))

    feature_cover = xr.merge(cover_datasets)
    logging.info("Feature cover calculation complete")
    #return feature_cover.rio.clip([boundary_union], crs=boundary.crs)
    return feature_cover



def squeeze_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Squeeze a Dataset by dropping extra dimensions."""
    for var in ds.data_vars:
        ds[var] = ds[var].squeeze()
    if "band" in ds.dims:
        ds = ds.drop_dims("band")
    return ds


def bbox_filter(bounds:tuple, bounds_vars = ("minx", "miny", "maxx", "maxy")) -> list:
    """
    Generate a set of filters to use in loading parquet files.
    """
    xmin, ymin, xmax, ymax = bounds
    xmin_label, ymin_label, xmax_label, ymax_label = bounds_vars
    filters = [
        (xmin_label, ">=", xmin),
        (ymin_label, ">=", ymin),
        (xmax_label, "<=", xmax),
        (ymax_label, "<=", ymax)
    ]

    return filters

def main(
    output_dir: Union[str, Path] = "data/evs",
    boundary_path: Union[str, Path] = "data/processed/boundary.geojson",
    buffer_distance: float = 7000,
    debug : bool = False,
    load_from_shp: bool = False
) -> Tuple[Path, Path]:
    """Process Ordnance Survey data to generate environmental variables.

    This function processes OS data to create two main outputs:
    1. Feature coverage density (percentage of area covered by each feature type)
    2. Distance to nearest feature (for each feature type)

    Args:
        output_dir: Directory where output rasters will be saved.
        boundary_path: Path to GeoJSON file defining the area of interest.
        buffer_distance: Distance in meters to buffer the boundary.
        debug: Enable debug logging.
        load_from_shp: Whether to load OS data from shapefiles instead of cached parquet files.

    Returns:
        Tuple containing:
        - feature_cover: xarray Dataset with feature coverage densities
        - distance_array: xarray Dataset with distance-to-feature matrices

    Raises:
        FileNotFoundError: If input files are not found.
        ValueError: If boundary or buffer distance are invalid.
    """

    setup_logging(log_level=logging.DEBUG if debug else logging.INFO)
    logging.info("Starting OS data processing pipeline")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", output_dir)

    logging.info("Loading boundary from %s with %dm buffer", boundary_path, buffer_distance)
    boundary = load_boundary(
        boundary_path, buffer_distance=buffer_distance, target_crs="EPSG:27700"
    )
    
    logging.info("Loading spatial configuration")
    spatial_config = load_spatial_config()
    boundary.to_crs(spatial_config["crs"], inplace=True)
    model_transform, bounds = construct_transform_shift_bounds(
        tuple(boundary.total_bounds), spatial_config["resolution"]
    )

    # Load OS data
    datasets = ["Building", "Water", "Woodland", "Road"]
    parquet_paths = generate_parquets(
        datasets, dir="data/raw/big-files/os-data", boundary=box(*bounds), overwrite=load_from_shp
    )
    os_data = {
        name: gpd.read_parquet(path) for name, path in zip(datasets, parquet_paths)
    }

    # Process roads
    major_roads, minor_roads = process_roads(os_data["Road"])

    # Prepare feature datasets
    feature_gdfs = {
        "major_roads": major_roads,
        "minor_roads": minor_roads,
        "woodland": os_data["Woodland"],
        "water": os_data["Water"],
        "buildings": os_data["Building"],
    }

    # Calculate and save feature cover
    feature_cover = calculate_feature_cover(feature_gdfs, boundary)
    feature_cover = reproject_data(
        feature_cover,
        spatial_config["crs"],
        transform=model_transform,
        resolution=spatial_config["resolution"],
    )
    logging.info("Writing feature cover raster")
    cover_path = output_dir / "os-feature-cover.tif"
    # write it as a dataset to keep band names
    feature_cover = squeeze_dataset(feature_cover) # type: ignore
    feature_cover.rio.to_raster(cover_path)

    # Calculate and save distance matrices
    distance_array = calculate_distances(feature_gdfs, boundary)
    distance_array = reproject_data(
        distance_array,
        spatial_config["crs"],
        transform=model_transform,
        resolution=spatial_config["resolution"],
    )
    logging.info("Writing distance matrix raster")
    distance_path = output_dir / "os-distance-to-feature.tif"
    distance_array = squeeze_dataset(distance_array) # type: ignore
    distance_array.rio.to_raster(
        distance_path
    )

    logging.info("OS data processing complete")
    logging.info("Output files saved to: %s", output_dir)
    return cover_path, distance_path


if __name__ == "__main__":
    main(
    )
