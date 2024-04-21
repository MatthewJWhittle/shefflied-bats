import os
import json
from pathlib import Path

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from shiny import App, render, ui, reactive
from ipyleaflet import (
    Map,
    GeoData,
    Popup,
    ImageOverlay,
)
from shinywidgets import register_widget, render_widget
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr
import numpy as np
from dotenv import load_dotenv
from pyproj import Transformer

from map import record_popup, generate_basemap
from app_config import species_name_mapping, feature_names
from ui import app_ui

app_dir = Path(__file__).parent
css_path = app_dir / "www" / "styles.css"

def load_deployment_config():
    path = f"{app_dir}/rsconnect-python/dashboard.json"
    with open(path, "r") as f:
        config = json.load(f)
    return config




def normalize(array:np.ndarray, vmin=None, vmax=None):
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


def array_to_rgb(array:np.ndarray, colormap="viridis", vmin=None, vmax=None):
    """
    This function converts an array to a RGB array using a colormap.
    """
    colormap_fn = get_cmap(colormap)
    # Normalize the band
    array = normalize(array, vmin, vmax)
    # apply the colormap to convert to RGD
    rgb = colormap_fn(array)
    return rgb

def write_tif_to_pngs(
        dataset:xr.Dataset,
        out_dir:Path, 
        colormap="viridis", 
        vmin=None, 
        vmax=None, 
        overwrite=False
        ) -> tuple[dict[str, Path], tuple, int]:
    """
    This function converts each band of a tif file to a RGB array and writes it to a PNG file.
    """
    # Get the band names
    band_names = dataset.data_vars.keys()
    output_paths  = {}

    colormap_fn = get_cmap(colormap)
    tif_bounds = dataset.rio.bounds()
    tif_crs = dataset.rio.crs.to_epsg()
    # Loop through each band
    for band_name in band_names:
        out_path = out_dir/f"{band_name}.png"
        output_paths[band_name] = out_path

        if out_path.exists() and not overwrite:
            # Skip if the file already exists and user doesn't want to overwrite
            continue

        # Get the band
        band = dataset[band_name].values
        # flip the y axis to match the tif
        band = np.flipud(band)
        # Normalize the band
        band = normalize(band, vmin, vmax)
        # apply the colormap to convert to RGD
        rgb = colormap_fn(band)
        # Write the PNG file
        plt.imsave(out_path, rgb)
        

    return output_paths, tif_bounds, tif_crs




load_dotenv()
running_env = os.getenv("ENV", "prod")
def get_tile_client_spec(env:str):
    urls = {
    "dev": "127.0.0.1",
    "prod": "https://api.shinyapps.io",
    }
    return urls[env]

#local_files = download_data()

def setup_paths():
    local_dir = f"{app_dir}/data"
    file_dependencies = {
        "results" : "results.csv",
        "bat_records" : "bat-records.parquet",
        "predictions" : "predictions_cog.tif",
        "boundary" : "boundary.parquet",
        "partial_dependence" : "partial-dependence-data.parquet",
    }
    local_files = {}
    for key, file in file_dependencies.items():
        local_files[key] = f"{local_dir}/{file}"
    
    return local_files


def load_training_data(path) -> gpd.GeoDataFrame:
    """
    This function loads the training data
    """
    # Load the training data
    training_data_gdf = gpd.read_parquet(path)
    training_data_gdf = training_data_gdf.to_crs(4326)
    # Return the dataframe
    return training_data_gdf



def load_predictions() -> xr.Dataset:
    """
    Load the model predictions array.

    This function loads the tif and processes it to be in the correct format for the dashboard.
    """
    predictions = rxr.open_rasterio(local_files["predictions"])
    predictions.coords["band"] = list(predictions.attrs["long_name"])
    # Set the nodata appropriately
    nodata = -1
    predictions = predictions.where(predictions >= 0, np.nan)

    # convert to a dataset to allow band name indexing
    predictions = predictions.to_dataset(dim="band")
    # Convert back to 0-1
    predictions = predictions / 100

    return predictions


def load_partial_dependence_data(path) -> pd.DataFrame:
    return pd.read_parquet(path)



def calculate_dependence_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the range of values for each feature.

    Args:
        df: pd.DataFrame - The dataframe of partial dependence values.

    Returns: pd.DataFrame - The dataframe of feature ranges.
    """
    # Group by the latin name, activity type and feature
    df = (
        df.groupby(["latin_name", "activity_type", "feature"])["average"]
        .apply(lambda x: x.max() - x.min())
        .reset_index()
    )
    # Return the dataframe
    return df

def layer_exists(
        m: Map,
          name: str):
    """
    Check if a layer exists in a map.

    Args:
        m: ipyleaflet.Map
        name: str
    
    Returns: bool - True if the layer exists, False otherwise
    
    """
    for layer in m.layers:
        if layer.name == name:
            return True
    return False


def get_layer(
        m: Map, 
        name: str
        ):
    """
    Get a layer from a map by name.

    Args:
        m: ipyleaflet.Map
        name: str

    Returns: ipyleaflet.Layer or None
    """
    for layer in m.layers:
        if layer.name == name:
            return layer
    return None


def load_south_yorkshire(path:Path):
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Load the counties data
    boundary = gpd.read_parquet(path)
    # Return the dataframe
    return boundary

## Load all static app data ----------
local_files = setup_paths()
results_df = pd.read_csv(local_files["results"])
training_data_gdf = load_training_data(
    local_files["bat_records"]
) 
partial_dependence_df = load_partial_dependence_data(local_files["partial_dependence"])
dependence_range = calculate_dependence_range(partial_dependence_df)

south_yorkshire = load_south_yorkshire(local_files["boundary"])


def project_bbox(bbox, from_crs, to_crs):
    """
    Project a bounding box from one CRS to another.

    Parameters:
    bbox (tuple or list): The bounding box to project, in the format (minx, miny, maxx, maxy).
    from_crs (str): The current CRS of the bounding box.
    to_crs (str): The CRS to project the bounding box to.

    Returns:
    tuple: The projected bounding box, in the format (minx, miny, maxx, maxy).
    """
    transformer = Transformer.from_crs(from_crs, to_crs)

    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])

    return (minx, miny, maxx, maxy)

## save the predictions to PNGs somewhere the app can GET them

png_dir = app_dir / "data" / "predictions_png"
png_dir.mkdir(exist_ok=True, parents=True)
png_paths, tif_bounds, tif_crs = write_tif_to_pngs(
    load_predictions(),
    png_dir,
    overwrite=False
)

tif_bounds = project_bbox(tif_bounds, f"EPSG:{tif_crs}", "EPSG:4326")

### App UI

main_app_ui = app_ui(
    css_path=css_path,
    species_name_mapping=species_name_mapping,
    results_df=results_df,
    
)


def server(input, output, session):
    @reactive.Calc
    def selected_results() -> pd.DataFrame:
        return results_df[
            (results_df["latin_name"] == input.species())
            & (results_df["activity_type"] == input.activity_type())
        ]


    @reactive.Calc
    def partial_dependence_range() -> pd.DataFrame:
        return dependence_range[
            (dependence_range.latin_name == input.species_mi())
            & (dependence_range.activity_type == input.activity_type_mi())
        ].copy()

    @output
    @render.data_frame
    def dependence_summary_table():
        pd_df = partial_dependence_range()
        pd_df["average"] = pd_df["average"].round(3)
        pd_df = pd_df[["feature", "average"]].sort_values("average", ascending=False)
        pd_df.feature.replace(feature_names, inplace=True)
        pd_df.rename(
            columns={"feature": "Feature", "average": "Influence Range"}, inplace=True
        )
        return pd_df

    @reactive.Calc
    def partial_dependence_df_model():
        return partial_dependence_df[
            (partial_dependence_df.latin_name == input.species_mi())
            & (partial_dependence_df.activity_type == input.activity_type_mi())
        ].copy()

    @output
    @render.plot
    def partial_dependence_plot():
        pd_df = partial_dependence_df_model()
        pd_df = pd_df[pd_df["feature"] == input.feature_mi()]
        ax = pd_df.plot(x="values", y="average")
        feature_label = feature_names[input.feature_mi()]
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Effect on Habitat Suitability Score")
        # drop the legend
        ax.get_legend().remove()
        # round y axis to 2 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        return ax

    @output
    @render.ui
    def model_description():
        model_result = selected_results()

        score_mean = model_result["mean_cv_score"].values[0]
        score_std = model_result["std_cv_score"].values[0]
        n_presence = model_result["n_presence"].values[0]
        n_background = model_result["n_background"].values[0]
        folds = model_result["folds"].values[0]

        description = f"""
        This model has an accuracy of **{round(score_mean * 100, 1)}%** (+/- {round(score_std * 100, 1)})%.
        
        Species Records: {n_presence}
        """

        return ui.markdown(description)

    @reactive.Calc
    def selected_training_data() -> gpd.GeoDataFrame:
        activity_type = input.activity_type()
        latin_name = input.species()
        
        subset_gdf = training_data_gdf[
            (training_data_gdf["latin_name"] == latin_name)
        ]
        # Only filter by activity type is 'All' isn't selected
        if activity_type != "All":
            subset_gdf = subset_gdf[
                (subset_gdf["activity_type"] == activity_type)
            ]
        return subset_gdf

    @reactive.Calc
    def predictions_png_path():
        band_name = selected_results()["band_name"].values[0]
        url =f"https://raw.githubusercontent.com/MatthewJWhittle/shefflied-bats/dashboard/dashboard/data/predictions_png/{band_name}.png?raw=true"
        
        # encode the url replacing spaces with %20
        url = url.replace(" ", "%20")
        #path = Path(app_dir) / "data/predictions_png/Pipistrellus pipistrellus_Foraging.png"
        return url

    @reactive.Calc
    def map_points():
        gdf = selected_training_data()

        #gdf = gdf[["geometry"]]
        return gdf

    ## Map ----------------------------------------------------------------------
    base_map = generate_basemap(south_yorkshire)
    register_widget("map", base_map)
    

    @output
    @render_widget
    def main_map() -> Map:
        png_path = predictions_png_path()

        if layer_exists(base_map, "HSM Predictions"):
            old_layer = get_layer(base_map, "HSM Predictions")
            base_map.remove_layer(old_layer)

        # Get the bounds of the tif
        sw_corner = [tif_bounds[0], tif_bounds[1]]
        ne_corner = [tif_bounds[2], tif_bounds[3]]
        bounds = [sw_corner, ne_corner]

        image_overlay = ImageOverlay(
            url=png_path,
            bounds=bounds,
            name="HSM Predictions",
        )


        base_map.add_layer(image_overlay)


        # Check if the training data layer is already added
        layer_name = "Species Records"
        geo_data = GeoData(
            geo_dataframe=map_points(), 
            name="Species Records", 
            point_style={'color': 'black', 'radius':6, 'fillColor': '#F96E46', 'opacity':0.8, 'weight':1.3,  'fillOpacity':0.8},
            hover_style={'fillColor': '#00E8FC' , 'fillOpacity': 1},
        )
        def generate_popup(event, properties, id, **kwargs):
            """
            Generate a popup for a feature on the map.
            """
            #print(properties)
            popup_content = record_popup(properties)
            popup = Popup(
                location=properties["geometry"]["coordinates"],
                child=popup_content,
            )
            # get the marker and open the popup
            base_map.add(popup)

        geo_data.on_click(generate_popup)

        if layer_exists(base_map, layer_name):
            old_layer = get_layer(base_map, layer_name)
            base_map.remove_layer(old_layer)
        
        base_map.add_layer(geo_data)
        

        return base_map
    

    # Model Summary --------------------------------------------------------------
    @output
    @render.data_frame
    def models_table():
        table = results_df[
            [
                "latin_name",
                "activity_type",
                "mean_cv_score",
                "std_cv_score",
                "n_presence",
                "n_background",
            ]
        ]
        table["mean_cv_score"] = (table["mean_cv_score"] * 100).round(1)
        table["std_cv_score"] = (table["std_cv_score"] * 100).round(1)
        table.rename(
            columns={
                "latin_name": "Species",
                "activity_type": "Activity Type",
                "mean_cv_score": "Accuracy (%)",
                "std_cv_score": "Accuracy Std (±%)",
                "n_presence": "Presence Points",
                "n_background": "Background Points",
            },
            inplace=True,
        )
        return table


app = App(main_app_ui, server)
