from shiny import App, render, ui, reactive
import shinyswatch
from ipyleaflet import (
    Map,
    basemaps,
    GeoData,
    LayersControl,
    basemap_to_tiles,
    Popup,
)
from localtileserver import get_leaflet_tile_layer, TileClient
from shinywidgets import output_widget, register_widget, render_widget
import geopandas as gpd
import pandas as pd
from pathlib import Path
import rioxarray as rxr
import xarray as xr
from map import record_popup, generate_basemap
css_path = Path(__file__).parent / "www" / "styles.css"


results_df = pd.read_csv("dashboard/data/results.csv")


training_data_gdf = gpd.read_parquet(
    "dashboard/data/bat-records.parquet"
)  # type: gpd.GeoDataFrame
training_data_gdf = training_data_gdf.to_crs(4326)


import numpy as np

def generate_popup(feature, **kwargs):
    print("Generating popup")
    print(f"Feature: {feature}")
    popup = Popup(
        location=feature["geometry"]["coordinates"],
        child=record_popup(feature["properties"]),
    )

def load_predictions() -> xr.Dataset:
    predictions = rxr.open_rasterio("dashboard/data/predictions.tif")
    predictions.coords["band"] = list(predictions.attrs["long_name"])
    # Set the nodata appropriately
    nodata = -1
    predictions = predictions.where(predictions >= 0, np.nan)

    # convert to a dataset to allow band name indexing
    predictions = predictions.to_dataset(dim="band")
    # Convert back to 0-1
    predictions = predictions / 100

    return predictions


def load_partial_dependence_data() -> pd.DataFrame:
    return pd.read_parquet("dashboard/data/partial-dependence-data.parquet")



def calculate_dependence_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the range of values for each feature
    """
    # Group by the latin name, activity type and feature
    df = (
        df.groupby(["latin_name", "activity_type", "feature"])["average"]
        .apply(lambda x: x.max() - x.min())
        .reset_index()
    )
    # Return the dataframe
    return df


partial_dependence_df = load_partial_dependence_data()

dependence_range = calculate_dependence_range(partial_dependence_df)

feature_names = partial_dependence_df["feature"].unique().tolist()


def layer_exists(m, name):
    for layer in m.layers:
        if layer.name == name:
            return True
    return False


def get_layer(m, name):
    for layer in m.layers:
        if layer.name == name:
            return layer


def load_south_yorkshire():
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Load the counties data
    boundary = gpd.read_parquet("dashboard/data/boundary.parquet")
    # Return the dataframe
    return boundary


south_yorkshire = load_south_yorkshire()

app_ui = ui.page_fluid(
    ui.include_css(css_path),
    shinyswatch.theme.minty(),
    ui.page_navbar(
        ui.nav(
            "Map",
            output_widget("map"),
            ui.panel_absolute(
                ui.tags.h3("Select a Model"),
                ui.input_selectize(
                    id="species",
                    label="",
                    choices=results_df["latin_name"].unique().tolist(),
                ),
                ui.input_selectize(
                    id="activity_type",
                    label="",
                    choices=results_df["activity_type"].unique().tolist(),
                ),
                ui.div(
                    ui.output_ui("model_description"),
                    class_="model-description-container",
                ),
                class_="model-selection-container",
            ),
        ),
        ui.nav(
            "Model Inspection",
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_selectize(
                        id="species_mi",
                        label="",
                        choices=results_df["latin_name"].unique().tolist(),
                    ),
                    ui.input_selectize(
                        id="activity_type_mi",
                        label="",
                        choices=results_df["activity_type"].unique().tolist(),
                    ),
                    ui.div(
                        ui.output_data_frame("dependence_summary_table"),
                        class_="dependence-summary-table-container",
                    ),
                ),
                ui.panel_main(
                    ui.row(
                        ui.div(
                            ui.input_selectize(
                                id="feature_mi",
                                label="",
                                choices=feature_names,
                            ),
                        ),
                        ui.row(
                            ui.output_plot("partial_dependence_plot"),
                            class_="partial-dependence-plot-container",
                        ),
                    )
                ),
            ),
        ),
        ui.nav(
            "Model Summary",
            ui.column(
                8,  # Width
                ui.tags.h3("Models"),
                ui.output_data_frame("models_table"),
                offset=2,
                style="display: flex; flex-direction: column; align-items: center; padding-top: 20px;",
            ),
        ),
        title="Sheffield Bat Group - HSM",
    ),
)


def server(input, output, session):
    @reactive.Calc
    def selected_results() -> pd.DataFrame:
        return results_df[
            (results_df["latin_name"] == input.species())
            & (results_df["activity_type"] == input.activity_type())
        ]

    @reactive.Calc
    def feature_names() -> list[str]:
        features = partial_dependence_df["feature"].unique().tolist()
        return features

    @reactive.Calc
    def partial_dependence_range() -> pd.DataFrame:
        return dependence_range[
            (dependence_range.latin_name == input.species_mi())
            & (dependence_range.activity_type == input.activity_type_mi())
        ].copy()

    @output
    @render.data_frame()
    def dependence_summary_table():
        pd_df = partial_dependence_range()
        pd_df["average"] = pd_df["average"].round(3)
        pd_df = pd_df[["feature", "average"]].sort_values("average", ascending=False)
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
        return pd_df.plot(x="values", y="average")

    @output
    @render.ui()
    def model_description():
        model_result = selected_results()

        score_mean = model_result["mean_cv_score"].values[0]
        score_std = model_result["std_cv_score"].values[0]
        n_presence = model_result["n_presence"].values[0]
        n_background = model_result["n_background"].values[0]
        folds = model_result["folds"].values[0]

        description = f"""
        This model has an accuracy of **{round(score_mean * 100, 1)}%** (+/- {round(score_std * 100, 1)})%. This was tested using ROC/AUC scores under {folds}-fold cross validation.
        
        Presence Points: {n_presence}

        Background Points: {n_background}
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
    def predictions_band():
        return selected_results()["band_name"].values[0]

    @reactive.Calc
    def map_points():
        gdf = selected_training_data()

        #gdf = gdf[["geometry"]]
        return gdf

    ## Map ----------------------------------------------------------------------
    base_map = generate_basemap(south_yorkshire)
    register_widget("map", base_map)
    

    @reactive.Calc
    def prediction_bands():
        return load_predictions().data_vars.keys()

    @reactive.Calc
    def tile_client():
        return TileClient(filename="dashboard/data/predictions.tif")

    @output
    @render_widget()
    def map() -> Map:
        band_names = prediction_bands()
        band_number = list(band_names).index(predictions_band())

        client = tile_client()
        tile_layer = get_leaflet_tile_layer(
            client,
            cmap="viridis",
            name="HSM Predictions",
            opacity=0.6,
            vmin=0,
            vmax=100,
            band=band_number + 1,
        )

        if layer_exists(base_map, "HSM Predictions"):
            old_layer = get_layer(base_map, "HSM Predictions")
            base_map.remove_layer(old_layer)

        base_map.add_layer(tile_layer)

        # Check if the training data layer is already added

        geo_data = GeoData(
            geo_dataframe=map_points(), 
            name="Training Data", 
            visible=False,
            style={'color': 'black', 'radius':6, 'fillColor': '#F96E46', 'opacity':0.5, 'weight':1.3,  'fillOpacity':0.6},
            hover_style={'fillColor': '#00E8FC' , 'fillOpacity': 0.2},
            point_style={'radius': 5, 'color': '#00E8FC', 'fillOpacity': 0.8, 'fillColor': 'blue', 'weight': 3},
        )
        geo_data.on_click(generate_popup)
        # set the layer to not be visible by deafult
        geo_data.visible = False

        if layer_exists(base_map, "Training Data"):
            old_layer = get_layer(base_map, "Training Data")
            base_map.remove_layer(old_layer)
        base_map.add_layer(geo_data)

        return base_map
    

    # Model Summary --------------------------------------------------------------
    @output
    @render.data_frame()
    def models_table():
        table = results_df[
            [
                "latin_name",
                "activity_type",
                "mean_cv_score",
                "std_cv_score",
                "n_presence",
                "n_background",
                "folds",
            ]
        ]
        table.rename(
            columns={
                "latin_name": "Species",
                "activity_type": "Activity Type",
                "mean_cv_score": "Accuracy",
                "std_cv_score": "Accuracy Std",
                "n_presence": "Presence Points",
                "n_background": "Background Points",
                "folds": "CV Folds",
            },
            inplace=True,
        )
        return table


app = App(app_ui, server)
