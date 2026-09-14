import geopandas as gpd
import pandas as pd
import topojson as tp # type: ignore # Add type ignore if linter complains about missing stubs
from pathlib import Path # For path type hints if used for file paths
from typing import Union, Optional # For type hints

from sdm.data.ons_download import (
    DEFAULT_CACHE_FILE,
    LEGACY_MANUAL_FILE,
    county_name_column,
    resolve_counties_file,
)

# Default path prefers live cache, then legacy manual drop.
UK_COUNTIES_FILE = DEFAULT_CACHE_FILE
LEGACY_UK_COUNTIES_FILE = LEGACY_MANUAL_FILE

def load_uk_counties_data(
    counties_filepath: Union[str, Path] = UK_COUNTIES_FILE,
    live_download: bool = True,
) -> gpd.GeoDataFrame:
    """Loads the UK counties and unitary authorities GeoJSON file."""
    resolved = resolve_counties_file(
        Path(counties_filepath) if counties_filepath else None,
        live_download=live_download,
    )
    return gpd.read_file(resolved)

def load_south_yorkshire(counties_filepath: Union[str, Path] = UK_COUNTIES_FILE) -> gpd.GeoDataFrame:
    """
    Loads UK counties data and filters for those in South Yorkshire.
    Assumes the input GeoDataFrame has a 'CTYUA23NM' column.
    """
    counties_gdf = load_uk_counties_data(counties_filepath)
    name_column = county_name_column(counties_gdf)
    south_yorkshire_names = ["Barnsley", "Doncaster", "Rotherham", "Sheffield"]
    south_yorkshire_gdf = counties_gdf[counties_gdf[name_column].isin(south_yorkshire_names)].copy()
    if south_yorkshire_gdf.empty:
        raise ValueError(f"No South Yorkshire counties found. Check {name_column} column and names.")
    return south_yorkshire_gdf

def load_yorkshire_study_area(
    counties_filepath: Union[str, Path] = UK_COUNTIES_FILE,
    target_crs: Union[str, int] = "EPSG:27700", 
    simplify_tolerance: Optional[float] = 100.0 # Allow None for no simplification
) -> gpd.GeoDataFrame:
    """
    Loads UK counties data, filters for a broader Yorkshire study area,
    reprojects, optionally simplifies using topojson, and dissolves by County.
    Assumes the input GeoDataFrame has a 'CTYUA23NM' column.
    """
    uk_counties_gdf = load_uk_counties_data(counties_filepath)
    name_column = county_name_column(uk_counties_gdf)
    
    county_subset_map = {
        "South Yorkshire": ["Barnsley", "Doncaster", "Rotherham", "Sheffield"],
        "West Yorkshire": ["Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield"],
        "North Yorkshire": ["North Yorkshire", "York"],
        "East Riding of Yorkshire": [
            "East Riding of Yorkshire",
            "Kingston upon Hull, City of",
        ],
    }

    # Create a DataFrame for merging county names with their broader region
    counties_df_list = []
    for region, names in county_subset_map.items():
        for name in names:
            counties_df_list.append({name_column: name, "CountyRegion": region})
    region_mapping_df = pd.DataFrame(counties_df_list)

    # Merge with the GeoDataFrame of UK counties
    study_area_gdf = uk_counties_gdf.merge(region_mapping_df, on=name_column, how="inner")

    if study_area_gdf.empty:
        raise ValueError(f"Study area is empty after merging. Check {name_column} column and names.")

    # Verify that all expected CTYUA names were found
    expected_ctyua_names = set(region_mapping_df[name_column])
    found_ctyua_names = set(study_area_gdf[name_column])
    missing_ctyua_from_map = expected_ctyua_names - found_ctyua_names
    if missing_ctyua_from_map:
        # This indicates that some names in county_subset_map were not found in the counties file
        raise ValueError(f"Counties listed in the map but not found in the shapefile: {missing_ctyua_from_map}")

    # Reproject to target CRS (e.g., British National Grid)
    study_area_gdf = study_area_gdf.to_crs(target_crs)

    # Optional simplification using topojson
    if simplify_tolerance is not None:
        # to_json to convert to GeoJSON string, then to_dict, then Topology
        study_area_tp = tp.Topology(study_area_gdf, prequantize=False, topology=True) # Removed to_json and to_dict
        study_area_tp.toposimplify(simplify_tolerance)
        study_area_gdf = study_area_tp.to_gdf() # This should retain attributes if possible
        # If attributes are lost, a merge might be needed here based on an ID

    # Dissolve by the broader CountyRegion and keep only necessary columns
    # Ensure geometry is valid before dissolve
    study_area_gdf['geometry'] = study_area_gdf.geometry.buffer(0)
    dissolved_gdf = study_area_gdf.dissolve(by="CountyRegion", as_index=False)
    dissolved_gdf = dissolved_gdf[["CountyRegion", "geometry"]]
    # Rename CountyRegion to County to match original script's output column name
    dissolved_gdf.rename(columns={"CountyRegion": "County"}, inplace=True)

    return dissolved_gdf 