"""Example login flow ot be implemented in a Shiny app."""
from shiny import App, reactive, render, req, ui, module



@module.ui
def app_ui():
    return ui.page_fixed(
        ui.output_ui("app")
    )

@module.ui
def app_content_ui():
    return ui.TagList(
        ui.h2(f"Hello, {input.username()}!"),
        ui.p("Welcome to my Shiny app.")
    )

@module.ui
def login_ui():
    return ui.modal(
        ui.input_text("username", "Username", width="200px"),
        ui.input_password("password", "password", width="200px"),
        ui.input_action_button(
            "trigger_password",
            "Log In",
            width="150px",
            height="5%",
        ),
        ui.output_text("ui_auth_error").add_class("text-danger"),
        size="xl",
        easy_close=False,
        footer=None,
    )

app_ui = ui.page_fixed(
    ui.output_ui("app")
)


@module.server
def login_server(input, output, session):
    logged_in = reactive.value(False)
    auth_error = reactive.value("")

    @render.ui
    def app():
        req(logged_in.get())
        return app_content_ui()
    
    @reactive.effect
    def _():
        ui.modal_show(
            ui.modal(
                ui.input_text("username", "Username", width="200px"),
                ui.input_password("password", "password", width="200px"),
                ui.input_action_button(
                    "trigger_password",
                    "Log In",
                    width="150px",
                    height="5%",
                ),
                ui.output_text("ui_auth_error").add_class("text-danger"),
                size="xl",
                easy_close=False,
                footer=None,
            ),
        )

    @reactive.effect
    @reactive.event(input.trigger_password)
    def __():
        if input.username() == input.password():
            # actually validate the username and password here
            logged_in.set(True)
            auth_error.set("")

        if logged_in.get():
            ui.modal_remove()
        else:
            auth_error.set("Invalid user credentials.")


    @render.text
    def ui_auth_error():
        req(auth_error.get())
        return auth_error.get()

    @reactive.effect
    @reactive.event(input.username)
    @reactive.event(input.password)
    def _():
        # Clear the auth error when the user updates credentials
        auth_error.set("")


app = App(app_ui, )