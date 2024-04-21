from shiny import ui, module
from shiny import reactive
import shinyswatch
from shinywidgets import output_widget
from app_config import feature_names

partial_dependence_explanation = "A partial dependence plot (PDP) is a visualization tool used in habitat suitability modelling to show the relationship between a specific variable and the predicted suitability, while averaging out the effects of all other features in the model. It helps to understand how the predicted suitability depends on the feature of interest (e.g. woodland cover), regardless of the values of the other features. However, PDPs do not capture complex interactions between features, so they might not fully represent how various environmental factors together affect bat habitat suitability"


def app_ui(css_path, species_name_mapping, results_df):
    return ui.page_fluid(
        ui.include_css(css_path),
        shinyswatch.theme.minty(),
        ui.page_navbar(
            ui.nav_panel(
                "HSM Maps",
                output_widget("main_map"),
                ui.panel_absolute(
                    ui.tags.h3("Select a Species"),
                    ui.input_selectize(
                        id="species",
                        label="",
                        choices=species_name_mapping,
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
            ui.nav_panel(
                "Explore Variable Importance",
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_selectize(
                            id="species_mi",
                            label="",
                            choices=species_name_mapping,
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
                            ui.column(
                                3,
                                # text
                                ui.tags.p("Plot Influence of:"),
                            ),
                        ui.column(
                            3,
                            ui.div(
                                ui.input_selectize(
                                    id="feature_mi",
                                    label="",
                                    choices=feature_names,
                                ),
                            ),
                            ),
                            ui.row(
                                ui.div(

                                ui.output_plot("partial_dependence_plot"),
                                class_="partial-dependence-plot-container",
                                ),
                                ui.tags.p(partial_dependence_explanation),
                            ),
                        )
                    ),
                ),
            ),
            ui.nav_panel(
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
