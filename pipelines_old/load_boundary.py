from os import name
import geopandas as gpd
import pandas as pd

def load_south_yorkshire():
    """
    This function loads the counties data which is a large file and filters it for those in south yorkshire
    """
    # Load the counties data
    counties = gpd.read_file("data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson")
    # Filter to just the counties we want
    south_yorkshire = ["Barnsley", "Doncaster", "Rotherham", "Sheffield"]
    counties = counties[counties["CTYUA23NM"].isin(south_yorkshire)]
    # Return the dataframe
    return counties

def main():
    counties = load_south_yorkshire()
    
    # Transform to flat projection
    counties = counties.to_crs(27700)

    # Add a buffer to avoid gaps in the union
    counties.geometry = counties.geometry.buffer(1000)

    # Simplify the polygon an union the geometries
    counties_simple = counties.geometry.simplify(100).unary_union
    
    # Convert the geometry to a geodataframe
    boundary = gpd.GeoDataFrame({"County":"South Yorkshire"}, geometry=[counties_simple], index=[0], crs = counties.crs) # type: ignore

    # Save the file
    boundary.to_file("data/processed/boundary.geojson", driver="GeoJSON")

if __name__ == "__main__":
    main()