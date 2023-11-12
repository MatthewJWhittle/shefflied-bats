from shiny import App, render, ui, reactive
import shinyswatch
import leafmap
from ipyleaflet import Map, basemaps, GeoData, LayersControl, WidgetControl
import localtileserver as lts
from shinywidgets import output_widget, render_widget
import geopandas as gpd
import pandas as pd

results_df = pd.read_csv("data/sdm_predictions/results.csv")
training_data_gdf = gpd.read_parquet(
    "data/sdm_predictions/training-occurrence-data.parquet"
) # type: gpd.GeoDataFrame
training_data_gdf = training_data_gdf.to_crs(4326)

app_ui = ui.page_navbar(
    shinyswatch.theme.minty(),
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
            style = "display: flex; flex-direction: column; align-items: center; padding-top: 20px;"
        ),
    ),
    title="Sheffield Bat Group - HSM",
)


def server(input, output, session):
    @reactive.Calc
    def selected_results() -> pd.DataFrame:
        return results_df[
            (results_df["latin_name"] == input.species())
            & (results_df["activity_type"] == input.activity_type())
        ]

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
        subset_gdf = training_data_gdf.query(
            f"latin_name == '{input.species()}' & activity_type == '{input.activity_type()}'"
        ) 
        return subset_gdf

    @reactive.Calc
    def predictions_path():
        return selected_results()["prediction_path"].values[0]
    

    @reactive.Calc
    def map_points():
        gdf = selected_training_data()
        gdf = gdf.sample(100)
        gdf = gdf[["geometry"]]
        return gdf

    @output
    @render_widget()
    def map() -> leafmap.Map:
        m = leafmap.Map(width="100%", height="85vh")
        m.add_basemap("HYBRID")

        m.add_raster(
            predictions_path(),
            cmap="viridis",
            layer_name="HSM Predictions",
            opacity=0.6,
        )
        print(map_points())
        print(m)

        m.add_gdf(map_points(), layer_name="Training Data")
        print(m)

        # Add a layer control
        m.add_layer_control()
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