"""
This script is used to upload the data required by the app to GCS. It shohuld be run whenever the modelling is updated.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
from shutil import copyfile

import rioxarray as rxr

import xarray as xr
import geopandas as gpd
import pandas as pd
import google.cloud.storage as gcs
from google import auth
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import numpy as np
from shapely.geometry import box
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt


def list_predictions(dir: Path) -> list[Path]:
    files = list(dir.glob("*_*.tif"))
    return files


def load_predictions(files: list[Path]) -> xr.Dataset:
    predictions = []
    for f in files:
        prediction = rxr.open_rasterio(f)
        # stem the filename as the dataset name
        prediction = prediction.squeeze()
        prediction = prediction.to_dataset(name=f.stem)
        predictions.append(prediction)

    # merge the datasets
    predictions = xr.merge(predictions)
    return predictions



def load_predictions_cog_array(path) -> xr.Dataset:
    """
    Load the model predictions array.

    This function loads the tif and processes it to be in the correct format for the dashboard.
    """
    predictions = rxr.open_rasterio(path)
    predictions.coords["band"] = list(predictions.attrs["long_name"])
    # Set the nodata appropriately
    nodata = -1
    predictions = predictions.where(predictions >= 0, np.nan)

    # convert to a dataset to allow band name indexing
    predictions = predictions.to_dataset(dim="band")
    # Convert back to 0-1
    predictions = predictions / 100

    return predictions


def compress_predictions(input_dir: Path, output_file: Path, bbox: tuple[float, float, float, float]):
    prediction_files = list_predictions(input_dir)

    predictions = load_predictions(prediction_files)
    predictions = predictions.rio.reproject("EPSG:4326")
    predictions = predictions.rio.clip_box(*bbox)
    predictions = predictions.rio.pad_box(*bbox)

    # Convert the predictions to int using a scaling factor
    predictions = predictions * 100
    predictions = predictions.round()
    int_nodata = -1
    predictions = predictions.fillna(int_nodata)

    for array in predictions.data_vars.values():
        array.rio.set_nodata(int_nodata)
        array.rio.write_nodata(int_nodata)

    predictions = predictions.astype("int16")
    # write the scale factor to the metadata
    predictions.attrs["scale_factor"] = 100

    predictions.rio.to_raster(output_file, compress="lzw")

    bounds = predictions.rio.bounds()
    return bounds


def process_results(results_file: Path, output_file: Path):
    results = pd.read_csv(results_file)
    results.drop(
        columns=["occurrence", "cv_models", "final_model", "prediction_path"],
        inplace=True,
    )
    results["band_name"] = results["latin_name"] + "_" + results["activity_type"]
    results.to_csv(output_file)


def csv_to_parquet(input_file: Path, output_file: Path):
    df = pd.read_csv(input_file)
    df.to_parquet(output_file)


def load_south_yorkshire():
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Load the counties data
    counties = gpd.read_file(
        "data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"
    )
    # Filter to just the counties we want
    south_yorkshire = ["Barnsley", "Doncaster", "Rotherham", "Sheffield"]
    counties = counties[counties["CTYUA23NM"].isin(south_yorkshire)]
    counties = counties.to_crs(4326)
    # Return the dataframe
    return counties


def compress_boundary(output_file: Path) -> tuple[tuple[float, float, float, float],...]:
    # Load the counties data
    counties = load_south_yorkshire()

    counties.to_parquet(output_file)
    # Save the counties data
    counties  = counties.to_crs(27700)
    counties["geometry"] = counties.geometry.buffer(1000)
    bbox_27700 = counties.total_bounds
    counties = counties.to_crs(4326)
    bbox_4326 = counties.total_bounds
    return bbox_4326, bbox_27700


def compress_evs(input_file: Path):
    evs = rxr.open_rasterio(input_file)
    evs = evs.squeeze()
    evs.rio.to_raster(output_file, compress="lzw")


def compress_occurence(input_file: Path, output_file: Path):
    occurence = gpd.read_parquet(input_file)
    occurence = occurence[occurence["class"] == 1]
    occurence.to_parquet(output_file)


def calculate_offset(x: float, y: float, origin: tuple, resolution: float):
    x_origin, y_origin = origin  # Changed from grid_origin to origin

    # Calculate offsets to align with grid
    offset_x = (x - x_origin) % resolution
    offset_y = (y - y_origin) % resolution

    return offset_x, offset_y


def align_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    origin: tuple,
    resolution: float,
    tolerance: float = 1e-10,
    decimals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    x_origin, y_origin = origin

    # Calculate offsets to align with grid
    offset_x = (x - x_origin) % resolution
    # Adjust to the right edge if very close
    offset_x = np.where(abs(offset_x - resolution) >= tolerance, offset_x, resolution)

    offset_y = (y - y_origin) % resolution
    # Adjust to the top edge if very close
    offset_y = np.where(abs(offset_y - resolution) >= tolerance, offset_y, resolution)

    # Apply the calculated offset to align the coordinates
    aligned_x = x - offset_x
    aligned_y = y - offset_y

    # Round to the required number of decimal places
    aligned_x = aligned_x.round(decimals)
    aligned_y = aligned_y.round(decimals)

    return aligned_x, aligned_y


def points_to_grid_squares(
    points: gpd.GeoSeries, grid_size: int = 100, origin=(0, 0)
) -> gpd.GeoSeries:
    """
    This function takes a series of points and converts them to grid squares.
    The grid squares are a standard size defined by the grid_size parameter and aligned to the origin.
    """
    # Get the points as np arrays
    x = points.x.values
    y = points.y.values

    # find the bottom left corner of the grid square
    x_mins, y_mins = align_to_grid(x, y, origin, grid_size, decimals=1)
    x_maxs = x_mins + grid_size
    y_maxs = y_mins + grid_size

    # Create a new geoseires with the grid squares
    grid_squares = gpd.GeoSeries(
        data=[
            box(x_min, y_min, x_max, y_max)
            for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs)
        ],
        crs=points.crs,
    )

    return grid_squares


def copy_bat_records(
    input_file: Path,
    output_file: Path,
    raster_bounds: tuple[float, float, float, float],
):
    bats = gpd.read_parquet(input_file)  # type: gpd.GeoDataFrame
    assert (
        bats.crs == 27700
    ), "The bats dataframe must be in British National Grid to allow records to be filtered by raster bounds"
    bats = bats[bats.accuracy <= 100]

    source_data = bats.source_data.apply(json.loads).apply(pd.Series)

    bats_gdf_full = bats.join(source_data)
    bats_gdf_full = bats_gdf_full[
        [
            "grid_reference",
            "species_raw",
            "activity_type",
            "source_data",
            "date",
            "latin_name",
            "common_name",
            "genus",
            "x",
            "y",
            "accuracy",
            "Recorder",
            "Notes",
            "Evidence",
            "Source",
            "row_id",
            "geometry",
        ]
    ]

    # Drop the records that are outside the raster bounds
    bats_gdf_full = bats_gdf_full.cx[
        raster_bounds[0] : raster_bounds[2], raster_bounds[1] : raster_bounds[3]
    ]

    bats_gdf_full["geometry"] = points_to_grid_squares(
        bats_gdf_full.geometry,
        grid_size=250,
        origin=(raster_bounds[0], raster_bounds[1]),
    )

    # then group by the geometry, latin_name and activity_type and count the number of records
    bats_gdf_full = (
        bats_gdf_full.groupby(["geometry", "latin_name", "activity_type"])
        .size()
        .reset_index(name="count")
    )
    # convert back to geodataframe as this is lost during group by
    bats_gdf_full = gpd.GeoDataFrame(bats_gdf_full, geometry="geometry", crs=27700)

    bats_gdf_full.to_parquet(output_file)


class GCSAppDataBucket:
    def __init__(self, name):
        credentials, _ = auth.load_credentials_from_file("sy-bat-4c78e6926c83.json")
        self.client = gcs.Client(credentials=credentials)
        self.bucket = self.client.bucket(name)

    def upload_file(self, file: Path, destination: str):
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(file)
    
    def upload_string(self, string: str, destination: str):
        blob = self.bucket.blob(destination)
        blob.upload_from_string(string)


def translate_to_cog(src_path, dst_path, profile="webp", profile_options={}, **options):
    """Convert image to COG."""
    # Format creation option (see gdalwarp `-co` option)
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        **options,
    )
    return True


def normalize(array: np.ndarray, vmin=None, vmax=None):
    """
    This function normalizes an array to be between 0 and 1.
    """
    if vmin is None or vmax is None:
        array = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    else:
        # clip the values to the min and max
        array = array.clip(vmin, vmax)
        array = (array - vmin) / (vmax - vmin)
    return array


def write_tif_to_pngs(
    dataset: xr.Dataset,
    out_dir: Path,
    colormap="viridis",
    vmin=None,
    vmax=None,
    overwrite=False,
) -> tuple[dict[str, Path], tuple, int]:
    """
    This function converts each band of a tif file to a RGB array and writes it to a PNG file.
    """
    # Get the band names
    band_names = dataset.data_vars.keys()
    output_paths = {}

    colormap_fn = get_cmap(colormap)
    dataset = dataset.rio.reproject("EPSG:4326")
    nodata = dataset["Pipistrellus pygmaeus_All"].rio.nodata
    # Set the no data values to nan
    dataset = dataset.where(dataset != nodata, np.nan)
    for var in dataset.data_vars:
        dataset[var] = dataset[var].rio.write_nodata(np.nan)
    tif_bounds = dataset.rio.bounds()
    tif_crs = dataset.rio.crs.to_epsg()
    # Loop through each band
    for band_name in band_names:
        out_path = out_dir / f"{band_name}.png"
        output_paths[band_name] = out_path

        if out_path.exists() and not overwrite:
            # Skip if the file already exists and user doesn't want to overwrite
            continue

        # Get the band
        band = dataset[band_name].values
        # flip the y axis to match the tif
        #band = np.flipud(band)
        # Normalize the band
        band = normalize(band, vmin, vmax)
        # apply the colormap to convert to RGD
        rgb = colormap_fn(band)
        # Write the PNG file
        plt.imsave(out_path, rgb)

    return output_paths, tif_bounds, tif_crs


def generate_config(
        prediction_bbox, 
        bbox_crs,
):
    config = {
        "prediction_bbox": prediction_bbox,
        "bbox_crs": bbox_crs,
    }
    return config

def main(env_type: str = "dev"):
    # Set up the GCS bucket
    bucket = "sygb-data"
    folders = {"dev": "app_data_dev", "prod": "app_data"}
    target_folder = folders[env_type]

    gcs_bucket = GCSAppDataBucket(name=bucket)

    # Set up a temporary directory
    temp_dir = TemporaryDirectory(dir="data/temp")
    temp_dir_path = Path(temp_dir.name)
    ## Boundary
    # Convert the boundary to parquet
    boundary_parquet = temp_dir_path / "boundary.parquet"
    bbox_4326, bbox_27700 = compress_boundary(boundary_parquet)

    ## SDM Predictions
    predictions_dir = Path("data/sdm_predictions")
    predictions_output_file = temp_dir_path / "predictions.tif"
    raster_bounds = compress_predictions(predictions_dir, predictions_output_file, bbox=bbox_4326)
    # translate the predictions to COG
    predictions_cog_output_file = temp_dir_path / "predictions_cog.tif"
    translate_to_cog(
        predictions_output_file, predictions_cog_output_file, profile="lzw"
    )
    # write the predictions to PNG
    predictions_png_output_dir = temp_dir_path / "predictions_png"
    predictions_png_output_dir.mkdir(exist_ok=True)

    output_files, tif_bounds, tif_crs = write_tif_to_pngs(
        load_predictions_cog_array(path =  predictions_cog_output_file), 
        predictions_png_output_dir
    )

    ## Model Results
    results_file = Path("data/sdm_predictions/results.csv")
    results_output = temp_dir_path / "results.csv"
    process_results(results_file, results_output)

    ## Partial Dependence
    # Convert the partial_dependence_csv to parquet
    partial_dependence_csv = Path("data/sdm_predictions/partial-dependence-data.csv")
    partial_dependence_parquet = temp_dir_path / "partial-dependence-data.parquet"
    csv_to_parquet(partial_dependence_csv, partial_dependence_parquet)



    ## Occurence Data
    occurence_output_file = temp_dir_path / "training-occurrence-data.parquet"
    compress_occurence(
        Path("data/sdm_predictions/training-occurrence-data.parquet"),
        occurence_output_file,
    )

    ## Bat Records
    bat_records_output_file = temp_dir_path / "bat-records.parquet"
    copy_bat_records(
        Path("data/processed/sybg-bats.parquet"),
        bat_records_output_file,
        raster_bounds=bbox_27700,
    )

    metadata = generate_config(
        prediction_bbox=tif_bounds, 
        bbox_crs=tif_crs,
    )

    print("Uploading to GCS")
    # Write the config to json string
    metadata_json = json.dumps(metadata)
    # Upload the config to GCS
    gcs_bucket.upload_string(metadata_json, f"{target_folder}/metadata.json")


    # Upload to GCS
    # predictions_cog.tif
    gcs_bucket.upload_file(
        predictions_cog_output_file, f"{target_folder}/predictions_cog.tif"
    )
    # bat-records.parquet
    gcs_bucket.upload_file(
        bat_records_output_file, f"{target_folder}/bat-records.parquet"
    )

    # boundary.parquet
    gcs_bucket.upload_file(boundary_parquet, f"{target_folder}/boundary.parquet")

    # training-occurrence-data.parquet
    gcs_bucket.upload_file(
        occurence_output_file, f"{target_folder}/training-occurrence-data.parquet"
    )

    # partial-dependence-data.parquet
    gcs_bucket.upload_file(
        partial_dependence_parquet, f"{target_folder}/partial-dependence-data.parquet"
    )

    # results.csv
    gcs_bucket.upload_file(results_output, f"{target_folder}/results.csv")

    # predictions_png
    for band_name, output_file in output_files.items():
        gcs_bucket.upload_file(
            output_file, f"{target_folder}/predictions_png/{band_name}.png"
        )

    # Clean up the temporary directory
    temp_dir.cleanup()
    print("Done")


if __name__ == "__main__":
    main(env_type="prod")
