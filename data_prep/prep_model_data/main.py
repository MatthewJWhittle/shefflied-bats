"""
Main function to prepare all data for model training.

This script 
1. loads (tidied) occurence data & the study bounday
2. Generates background points
3. Loads the environmental data & creates a grid gdf (centroid of each cell)
3. Stacks the occurence data with the background points
4. Extracts the environmental data for the combined occurence and background points
5. Saves the final data to a specified output directory
"""
from typing import Union
from pathlib import Path

import geopandas as gpd




def main(
        occurence_data_path: Union[Path, str],
        boundary_path : Union[Path, str],
        evs_data_path: Union[Path, str],
        output_dir: Union[Path, str],
)