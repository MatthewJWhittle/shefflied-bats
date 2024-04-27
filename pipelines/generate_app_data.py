"""
This script is used to upload the data required by the app to GCS. It shohuld be run whenever the modelling is updated.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import rioxarray as rxr

import xarray as xr
import geopandas as gpd
import pandas as pd
import google.cloud.storage as gcs
from google import auth
from rio_cogeo.cogeo import cog_translate


from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


def list_predictions(dir:Path) -> list[Path]:
    files = list(dir.glob('*_*.tif'))
    return files

def load_predictions(files:list[Path]) -> xr.Dataset:
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



def compress_predictions(input_dir:Path, output_file:Path): 
    prediction_files = list_predictions(input_dir)

    predictions = load_predictions(prediction_files)

    # Convert the predictions to int using a scaling factor
    predictions = predictions * 100
    predictions = predictions.round()
    int_nodata = -1
    predictions = predictions.fillna(int_nodata)

    for array in predictions.data_vars.values():
        array.rio.set_nodata(int_nodata)
        array.rio.write_nodata(int_nodata)

    predictions = predictions.astype('int16')
    # write the scale factor to the metadata
    predictions.attrs['scale_factor'] = 100


    predictions.rio.to_raster(output_file, compress='lzw')

    bounds = predictions.rio.bounds()
    return bounds



def process_results(results_file:Path, output_file:Path):
    results = pd.read_csv(results_file)
    results.drop(columns=['occurrence', 'cv_models', 'final_model', 'prediction_path'], inplace=True)
    results["band_name"] = results["latin_name"] + "_" + results["activity_type"]
    results.to_csv(output_file)


def csv_to_parquet(input_file:Path, output_file:Path):
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


def compress_boundary(output_file:Path):
    if output_file.exists():
        return
    # Load the counties data
    counties = load_south_yorkshire()
    # Save the counties data
    counties.to_parquet(output_file)

def compress_evs(input_file:Path, output_file:Path):
    evs = rxr.open_rasterio(input_file)
    evs = evs.squeeze()
    evs.rio.to_raster(output_file, compress='lzw')

def compress_occurence(input_file:Path, output_file:Path):
    occurence = gpd.read_parquet(input_file)
    occurence = occurence[occurence["class"] == 1]
    occurence.to_parquet(output_file)

def copy_bat_records(input_file:Path, output_file:Path, raster_bounds:tuple[float]):
    bats = gpd.read_parquet(input_file) # type: gpd.GeoDataFrame
    assert bats.crs == 27700, "The bats dataframe must be in British National Grid to allow records to be filtered by raster bounds"
    bats = bats[bats.accuracy <= 100]

    source_data = bats.source_data.apply(json.loads).apply(pd.Series)

    bats_gdf_full = bats.join(source_data)
    bats_gdf_full = bats_gdf_full[[
        'grid_reference', 'species_raw', 'activity_type', 'source_data', 'date',
       'latin_name', 'common_name', 'genus', 'x', 'y', 'accuracy', 
       'Recorder', "Notes", "Evidence", "Source", "row_id", 'geometry',
    ]]

    # Drop the records that are outside the raster bounds
    bats_gdf_full = bats_gdf_full.cx[raster_bounds[0]:raster_bounds[2], raster_bounds[1]:raster_bounds[3]]

    bats_gdf_full.to_parquet(output_file)


class GCSAppDataBucket:
    def __init__(self, name):
        credentials, _ = auth.load_credentials_from_file("sy-bat-4c78e6926c83.json")
        self.client = gcs.Client(credentials=credentials)
        self.bucket = self.client.bucket(name)

    def upload_file(self, file:Path, destination:str):
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(file)


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

def main():
    # Set up the GCS bucket
    bucket = "sygb-data"
    gcs_bucket = GCSAppDataBucket(name = bucket)

    # Set up a temporary directory
    temp_dir = TemporaryDirectory(dir="data/temp")
    temp_dir_path = Path(temp_dir.name)

    ## SDM Predictions
    predictions_dir = Path("data/sdm_predictions")
    predictions_output_file =  temp_dir_path / "predictions.tif"
    raster_bounds = compress_predictions(predictions_dir, predictions_output_file)
    # translate the predictions to COG
    predictions_cog_output_file =  temp_dir_path / "predictions_cog.tif"
    translate_to_cog(predictions_output_file, predictions_cog_output_file, profile="lzw")
    # Copy the cog to the top level of the directory
    from shutil import copyfile
    copyfile(predictions_cog_output_file, "predictions_cog.tif")

    ## Model Results
    results_file = Path("data/sdm_predictions/results.csv")
    results_output = temp_dir_path / "results.csv"
    process_results(results_file, results_output)

    ## Partial Dependence
    # Convert the partial_dependence_csv to parquet
    partial_dependence_csv = Path("data/sdm_predictions/partial-dependence-data.csv")
    partial_dependence_parquet = temp_dir_path / "partial-dependence-data.parquet"
    csv_to_parquet(partial_dependence_csv, partial_dependence_parquet)

    ## Boundary
    # Convert the boundary to parquet
    boundary_parquet = temp_dir_path / "boundary.parquet"
    compress_boundary(boundary_parquet)

    ## Occurence Data
    occurence_output_file = temp_dir_path / "training-occurrence-data.parquet"
    compress_occurence(
        Path("data/sdm_predictions/training-occurrence-data.parquet"),
        occurence_output_file
    )

    ## Bat Records
    bat_records_output_file = temp_dir_path / "bat-records.parquet"
    copy_bat_records(
        Path("data/processed/sybg-bats.parquet"),
        bat_records_output_file,
        raster_bounds=raster_bounds
    )

    # Upload to GCS
    # predictions_cog.tif

    gcs_bucket.upload_file(predictions_cog_output_file, "app_data/predictions_cog.tif")
    # bat-records.parquet
    gcs_bucket.upload_file(bat_records_output_file, "app_data/bat-records.parquet")
    
    # boundary.parquet
    gcs_bucket.upload_file(boundary_parquet, "app_data/boundary.parquet")
    
    # training-occurrence-data.parquet
    gcs_bucket.upload_file(occurence_output_file, "app_data/training-occurrence-data.parquet")
    
    # partial-dependence-data.parquet
    gcs_bucket.upload_file(partial_dependence_parquet, "app_data/partial-dependence-data.parquet")
    
    # results.csv
    gcs_bucket.upload_file(results_output, "app_data/results.csv")

    # Clean up the temporary directory
    temp_dir.cleanup()



if __name__ == '__main__':
    main()