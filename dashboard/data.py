from pathlib import Path
from regex import B
import rioxarray as rxr
from sklearn.inspection import partial_dependence
from sympy import comp
import xarray as xr
import geopandas as gpd

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

import shutil
import pandas as pd

def copy_results(results_file:Path, output_dir:Path):
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / results_file.name
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
import json
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

def main():
    predictions_dir = Path("data/sdm_predictions")
    predictions_output_file = Path("dashboard/data/predictions.tif")
    raster_bounds = compress_predictions(predictions_dir, predictions_output_file)

    results_file = Path("data/sdm_predictions/results.csv")
    results_output_dir = Path("dashboard/data")
    copy_results(results_file, results_output_dir)

    # Convert the partial_dependence_csv to parquet
    partial_dependence_csv = Path("data/sdm_predictions/partial-dependence-data.csv")
    partial_dependence_parquet = Path("dashboard/data/partial-dependence-data.parquet")
    csv_to_parquet(partial_dependence_csv, partial_dependence_parquet)

    # Convert the boundary to parquet
    boundary_parquet = Path("dashboard/data/boundary.parquet")
    compress_boundary(boundary_parquet)


    compress_occurence(
        Path("data/sdm_predictions/training-occurrence-data.parquet"),
        Path("dashboard/data/training-occurrence-data.parquet")
    )

    copy_bat_records(
        Path("data/processed/sybg-bats.parquet"),
        Path("dashboard/data/bat-records.parquet"),
        raster_bounds=raster_bounds
    )



if __name__ == '__main__':
    main()