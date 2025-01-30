import geopandas as gpd
import pandas as pd
import topojson as tp


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
    # Return the dataframe
    return counties


def load_study_area():
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Load the counties data
    uk_counties = gpd.read_file(
        "data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson"
    )
    # Filter to just the counties we want
    county_subset = {
        "South Yorkshire": ["Barnsley", "Doncaster", "Rotherham", "Sheffield"],
        "West Yorkshire": ["Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield"],
        "North Yorkshire": ["North Yorkshire", "York"],
        "East Riding of Yorkshire": [
            "East Riding of Yorkshire",
            "Kingston upon Hull, City of",
        ],
    }

    # convert to a dataframe
    counties = pd.DataFrame(data={"CTYUA23NM": county_subset.values(), "County" : county_subset.keys()})
    counties = counties.explode("CTYUA23NM").reset_index(drop=True)

    study_area = uk_counties.merge(counties, on="CTYUA23NM", how="inner")

    # check that all counties are present
    missing_counties = set(study_area.CTYUA23NM) - set(counties.CTYUA23NM)
    if len(missing_counties) > 0:
        raise ValueError(f"Missing counties: {missing_counties}")
    
    # Return the dataframe

    # Transform to flat projection
    study_area = study_area.to_crs(27700)

    # convert tot topojson, simplify, then back to gdf
    study_area_tp = tp.Topology(study_area, prequantize=False, topology=True)
    study_area_tp.toposimplify(100)
    study_area = study_area_tp.to_gdf()

    # Merge the geometries
    study_area = study_area.dissolve(by="County", dropna=False, as_index=False)
    study_area = study_area[["County", "geometry"]]

    return study_area


def main():
    counties = load_study_area()
    
    # Save the file
    filepath = "data/processed/boundary.geojson"
    # Save the file
    counties.to_file(filepath, driver="GeoJSON")

    return filepath



if __name__ == "__main__":
    main()