"Basic app to test local tile server deployment"

import os

from shiny import App, ui
from ipyleaflet import Map

from shinywidgets import output_widget, render_widget
from localtileserver import TileClient, get_leaflet_tile_layer

ui = ui.page_fluid(
    ui.div(
        ui.h1("Local Tile Server Test"),
        output_widget("tile_map"),
))
# get the shiny app url from the environment variable
app_url = "https://matthewjwhittle.shinyapps.io/tile-server-test/__sockjs__/{port}"



def server(input, output, session):
    @render_widget("tile_map")
    def tile_map():
        m = Map()
        url = "https://github.com/giswqs/data/raw/main/raster/landsat7.tif"
        tile_client = TileClient(
            url,
            client_prefix=app_url,

            )
        tile_layer = get_leaflet_tile_layer(tile_client)
        m.add_layer(tile_layer)
        m.zoom = tile_client.default_zoom
        m.center = tile_client.center()
        return m
    
app = App(ui=ui, server=server)