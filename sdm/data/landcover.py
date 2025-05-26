"""
Land cover data processing functionality.
"""

from typing import Dict, List, Union, Optional
import xarray as xr
import numpy as np

def get_ceh_land_cover_codes_v2023() -> Dict[str, str]:
    """Get CEH Land Cover Map 2023 category codes and labels."""
    return {
        "1": "Broadleaved Woodland",
        "2": "Coniferous Woodland",
        "3": "Arable and Horticulture",
        "4": "Improved Grassland",
        "5": "Neutral Grassland",
        "6": "Calcareous Grassland",
        "7": "Acid Grassland",
        "8": "Fen, Marsh and Swamp",
        "9": "Heather",
        "10": "Heather Grassland",
        "11": "Bog",
        "12": "Inland Rock",
        "13": "Saltwater",
        "14": "Freshwater",
        "15": "Supra-littoral Rock",
        "16": "Supra-littoral Sediment",
        "17": "Littoral Rock",
        "18": "Littoral Sediment",
        "19": "Saltmarsh",
        "20": "Urban",
        "21": "Suburban",
        "22": "Inland Bare Ground",
        "23": "Coastal Bare Ground",
        "24": "Marine, Littoral",
    }

def define_broad_habitat_categories() -> Dict[str, List[str]]:
    """Define mapping of CEH land cover categories to broad habitat types."""
    return {
        "Woodland": ["Broadleaved_Woodland", "Coniferous_Woodland"],
        "Grassland": ["Improved_Grassland", "Neutral_Grassland", "Calcareous_Grassland", "Acid_Grassland", "Heather_Grassland"],
        "Heathland": ["Heather"],
        "Wetland_Bog": ["Fen_Marsh_and_Swamp", "Bog"],
        "Arable": ["Arable_and_Horticulture"],
        "Urban": ["Urban", "Suburban"],
        "Other_Bare": ["Inland_Bare_Ground", "Coastal_Bare_Ground", "Inland_Rock"],
        "Coastal_Marine": ["Saltwater", "Freshwater", "Supra-littoral_Rock", "Supra-littoral_Sediment", 
                          "Littoral_Rock", "Littoral_Sediment", "Saltmarsh", "Marine_Littoral"]
    }

def create_binary_raster_from_category(
    source_raster: xr.DataArray,
    category_value: int,
    output_var_name: str
) -> xr.Dataset:
    """Create a binary raster for a specific land cover category."""
    binary = (source_raster == category_value).astype(np.float32)
    return xr.Dataset({output_var_name: binary})

def aggregate_categorical_rasters(
    categorical_raster_ds: xr.Dataset,
    aggregation_map: Dict[str, List[str]],
    categories_to_drop: Optional[List[str]] = None
) -> xr.Dataset:
    """Aggregate categorical rasters into broader habitat categories."""
    aggregated = {}
    categories_to_drop = categories_to_drop or []
    
    for broad_cat, component_cats in aggregation_map.items():
        if broad_cat in categories_to_drop:
            continue
            
        # Sum all component categories
        components = [categorical_raster_ds[cat] for cat in component_cats if cat in categorical_raster_ds]
        if components:
            aggregated[broad_cat] = sum(components)
    
    return xr.Dataset(aggregated) 