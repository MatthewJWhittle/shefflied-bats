
import geopandas as gpd
from matplotlib.pyplot import cla
import pandas as pd
from shiny.ui import tags
from ipyleaflet import Popup



def record_popup(feature) -> str:
    """
    Generates an HTML string for a popup based on the provided feature of bat data.
    """

    popup = tags.div(
        tags.h4(f"{feature['common_name']} ({feature['latin_name']})"),
        tags.table(
            tags.tr(
                tags.td("Activity:"),
                tags.td(feature['activity_type'])
            ),
            tags.tr(
                tags.td("Date:"),
                tags.td(feature['date'][:10] if pd.notna(feature['date']) else 'N/A')
            ),
            tags.tr(
                tags.td("Recorder:"),
                tags.td(feature['Recorder'])
            ),
            tags.tr(
                tags.td("Notes:"),
                tags.td(feature['Notes'] if pd.notna(feature['Notes']) else 'N/A')
            ),
        ),
        className="map-popup"

    )

    return popup._repr_html_()

from ipyleaflet import Map, basemaps, basemap_to_tiles, GeoData, LayersControl
def generate_basemap(south_yorkshire: gpd.GeoDataFrame):
        ## Map ----------------------------------------------------------------------
    m = Map(width="100%", height="100%", zoom=13)
    print("Init Map")

    imagery = basemap_to_tiles(basemaps.Esri.WorldImagery)
    imagery.base = True
    imagery.name = "Imagery"

    m.add_layer(imagery)

    bbox = south_yorkshire.total_bounds

    m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    m.add_control(LayersControl(position="bottomleft", collapsed=False))

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

    return m


def main():
    bat_records  = gpd.read_parquet("dashboard/data/bat-records.parquet")
    
    bat_records['popup'] = bat_records.apply(record_popup, axis=1)
    # geet one to check
    bat_records.iloc[0].popup
    pass

if __name__ == '__main__':
    main()
