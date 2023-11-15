import pickle
from unittest import result
from shiny import App, render, ui, reactive
import shinyswatch
from ipyleaflet import (
    Map,
    basemaps,
    GeoData,
    LayersControl,
    WidgetControl,
    TileLayer,
    basemap_to_tiles,
)
from localtileserver import get_leaflet_tile_layer, TileClient
from shinywidgets import output_widget, render_widget
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from pathlib import Path
import rioxarray as rxr
import xarray as xr
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

css_path = Path(__file__).parent / "www" / "styles.css"


results_df = pd.read_csv("data/sdm_predictions/results.csv")




training_data_gdf = gpd.read_parquet(
    "data/sdm_predictions/training-occurrence-data.parquet"
)  # type: gpd.GeoDataFrame
training_data_gdf = training_data_gdf.to_crs(4326)




def load_ev_df() -> pd.DataFrame:
    evs = rxr.open_rasterio("data/sdm_predictions/evs.tif")
    evs.coords["band"] = list(evs.attrs["long_name"])
    evs = evs.to_dataset(dim="band")

    ev_df = evs.to_dataframe()
    ev_df.drop("spatial_ref", axis=1, inplace=True)
    ev_df.dropna(inplace=True)

    return ev_df

def load_partial_dependence_data() -> pd.DataFrame:
    return pd.read_csv("data/sdm_predictions/partial-dependence-data.csv")

partial_dependence_df = load_partial_dependence_data()
dependence_range = partial_dependence_df.groupby(["latin_name", "activity_type", "feature"]).apply(lambda x: x.max() - x.min()).reset_index()

feature_names = load_ev_df().columns.tolist()


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
    counties = gpd.read_file(
        "data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"
    )
    # Filter to just the counties we want
    south_yorkshire = ["Barnsley", "Doncaster", "Rotherham", "Sheffield"]
    counties = counties[counties["CTYUA23NM"].isin(south_yorkshire)]
    counties = counties.to_crs(4326)
    # Return the dataframe
    return counties


south_yorkshire = load_south_yorkshire()

app_ui = ui.page_fluid(
    ui.include_css(css_path),
    shinyswatch.theme.minty(),
    ui.page_navbar(
        ui.nav(
            "Results",
            ui.layout_sidebar(
                ui.panel_sidebar(
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
                    ui.div(
                        ui.tags.h3("Partial Dependence"),
                        ui.input_selectize(
                            id="feature",
                            label="Select a Feature",
                            choices=feature_names,
                        ),
                        ui.output_plot("partial_dependence_plot"),
                        class_="partial-dependence-container",
                    ),
                    width=3,
                ),
                ui.panel_main(
                    output_widget("map"),
                    width=9,
                ),
            ),
        ),
        ui.nav(
            "Tables",
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
    def selected_model():
        model = selected_results()["final_model"].values[0]
        return model
    
    @reactive.Calc
    def ev_df() -> pd.DataFrame:
        return load_ev_df()
    
    @reactive.Calc
    def feature_names() -> list[str]:
        return ev_df().columns.tolist()
    
    @reactive.Calc
    def partial_dependence_range() -> pd.DataFrame:
        return dependence_range[(dependence_range.latin_name == input.species()) & (dependence_range.activity_type == input.activity_type())].copy()

    @reactive.Calc
    def partial_dependence_df_model():
        return partial_dependence_df[(partial_dependence_df.latin_name == input.species()) & (partial_dependence_df.activity_type == input.activity_type())].copy()
    
    @output
    @render.plot
    def partial_dependence_plot():
        pd_df = partial_dependence_df_model()
        pd_df = pd_df[pd_df["feature"] == input.feature()]
        return pd_df.plot(x="values", y="average", title=f"Partial Dependence for {input.feature()}")
    

    

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
        subset_gdf = training_data_gdf[
            (training_data_gdf["latin_name"] == input.species())
            & (training_data_gdf["activity_type"] == input.activity_type())
        ]
        return subset_gdf

    @reactive.Calc
    def predictions_path():
        return selected_results()["prediction_path"].values[0]

    @reactive.Calc
    def map_points():
        gdf = selected_training_data()
        gdf = gdf[gdf["class"] == 1]
        gdf = gdf[["geometry"]]
        return gdf

    @reactive.Calc
    def map_base():
        m = Map(width="100%", height="100%", zoom=18)

        imagery = basemap_to_tiles(basemaps.Esri.WorldImagery)
        imagery.base = True
        imagery.name = "Imagery"

        m.add_layer(imagery)

        m.add_control(LayersControl(position="topright"))

        sy_geo = GeoData(
            geo_dataframe=south_yorkshire,
            name="South Yorkshire Boundary",
            style={
                "color": "pink",
                "fillColor": "pink",
                "opacity": 0.6,
                "weight": 1.9,
                "dashArray": "2",
                "fillOpacity": 0,
            },
        )
        m.add_layer(sy_geo)

        bbox = south_yorkshire.total_bounds

        m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

        return m

    @output
    @render_widget()
    def map() -> Map:
        m = map_base()

        client = TileClient(filename=predictions_path())
        tile_layer = get_leaflet_tile_layer(
            client, cmap="viridis", name="HSM Predictions", opacity=0.6
        )

        if layer_exists(m, "HSM Predictions"):
            old_layer = get_layer(m, "HSM Predictions")
            m.remove_layer(old_layer)

        m.add_layer(tile_layer)

        # Check if the training data layer is already added

        geo_data = GeoData(geo_dataframe=map_points(), name="Training Data", visible=False)
        if layer_exists(m, "Training Data"):
            old_layer = get_layer(m, "Training Data")
            m.remove_layer(old_layer)
        m.add_layer(geo_data)

        return m

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
