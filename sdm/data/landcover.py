"""
Land cover data processing functionality.
"""

from typing import Dict, List, Optional
import xarray as xr
import numpy as np

def get_ceh_land_cover_codes_v2023() -> Dict[str, str]:
    """Get CEH Land Cover Map 2023 category codes and labels."""
    return {
        "1": "Broadleaved woodland",
        "2": "Coniferous woodland",
        "3": "Arable",
        "4": "Improved grassland",
        "5": "Neutral grassland",
        "6": "Calcareous grassland",
        "7": "Acid grassland",
        "8": "Fen, Marsh and Swamp",
        "9": "Heather and shrub",
        "10": "Heather grassland",
        "11": "Bog",
        "12": "Inland rock",
        "13": "Saltwater",
        "14": "Freshwater",
        "15": "Supralittoral rock",
        "16": "Supralittoral sediment",
        "17": "Littoral rock",
        "18": "Littoral sediment",
        "19": "Saltmarsh",
        "20": "Urban",
        "21": "Suburban",
    }

def define_broad_habitat_categories() -> Dict[str, List[str]]:
    """Define mapping of CEH land cover categories to broad habitat types."""
    return {
        "Grassland": [
            "Neutral grassland",
            "Calcareous grassland",
            "Acid grassland",
        ],
        "Marine, Littoral": [
            "Saltwater",
            "Supralittoral rock",
            "Supralittoral sediment",
            "Littoral rock",
            "Littoral sediment",
            "Saltmarsh",
        ],
        "Upland Heathland": ["Heather and shrub", "Heather grassland"],
        "Wetland": ["Bog", "Fen, Marsh and Swamp"],
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