import re

def tidy_variable_name(name: str) -> str:
    """
    Cleans up a string to be a suitable variable name by:
    - Replacing dashes, spaces, and other common separators with underscores.
    - Converting to lowercase.
    - Stripping leading/trailing whitespace and underscores.
    - Ensuring no multiple consecutive underscores.

    Args:
        name (str): The input string.

    Returns:
        str: The cleaned up string, suitable for use as a variable name.
    """
    if not isinstance(name, str):
        try:
            name = str(name) # Attempt to convert to string if not already
        except Exception:
            # Or raise a TypeError if strict string input is required
            raise TypeError(f"Input name must be a string or convertible to a string, got {type(name)}")

    # Replace common separators and problematic characters with underscores
    name = re.sub(r'[\s\-/\\.:;,()\[\]{}]', '_', name)
    # Convert to lowercase
    name = name.lower()
    # Remove any characters not alphanumeric or underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Replace multiple underscores with a single underscore
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    
    # Ensure it doesn't start with a number if it's meant for some contexts (e.g. Python var names)
    # For raster band names, this might be less critical.
    # if name and name[0].isdigit():
    #     name = f"_{name}" # Prepend underscore
        
    return name


# Human-friendly display names for EV/variable names (e.g. charts, SHAP, presentations).
# Use get_variable_display_name() for lookup with fallback.
VARIABLE_DISPLAY_NAMES = {
    # Climate – stats
    "climate_stats_temp_ann_var": "Temp variance (annual)",
    "climate_stats_temp_ann_avg": "Temp (annual avg)",
    "climate_stats_temp_mat_avg": "Temp (warmest month)",
    "climate_stats_prec_ann_var": "Precip variance (annual)",
    "climate_stats_prec_ann_avg": "Precip (annual avg)",
    "climate_stats_wind_ann_avg": "Wind (annual avg)",
    # Climate – bioclim
    "climate_bioclim_bio_2": "Diurnal temp range",
    "climate_bioclim_bio_3": "Isothermality",
    "climate_bioclim_bio_4": "Temp seasonality",
    "climate_bioclim_bio_7": "Annual temp range",
    "climate_bioclim_bio_8": "Temp (wettest quarter)",
    "climate_bioclim_bio_9": "Temp (coldest quarter)",
    # Terrain
    "terrain_dtm": "Elevation",
    "terrain_stats_slope": "Slope",
    "terrain_stats_roughness": "Roughness",
    "terrain_stats_tpi": "TPI",
    "terrain_stats_twi": "Wetness index",
    "terrain_stats_aspect_northness_slope": "Northness",
    "terrain_stats_aspect_eastness_slope": "Eastness",
    # CEH landcover – point
    "ceh_landcover_wetland": "Wetland",
    "ceh_landcover_upland_heathland": "Upland heath",
    "ceh_landcover_grassland": "Grassland",
    "ceh_landcover_arable": "Arable",
    "ceh_landcover_broadleaved_woodland": "Broadleaved woodland",
    "ceh_landcover_suburban": "Suburban",
    "ceh_landcover_coniferous_woodland": "Coniferous woodland",
    "ceh_landcover_urban": "Urban",
    "ceh_landcover_improved_grassland": "Improved grassland",
    # CEH landcover – 500m
    "ceh_landcover_wetland_500m": "Wetland (500m)",
    "ceh_landcover_upland_heathland_500m": "Upland heath (500m)",
    "ceh_landcover_grassland_500m": "Grassland (500m)",
    "ceh_landcover_arable_500m": "Arable (500m)",
    "ceh_landcover_broadleaved_woodland_500m": "Broadleaved woodland (500m)",
    "ceh_landcover_suburban_500m": "Suburban (500m)",
    "ceh_landcover_coniferous_woodland_500m": "Coniferous woodland (500m)",
    "ceh_landcover_urban_500m": "Urban (500m)",
    "ceh_landcover_improved_grassland_500m": "Improved grassland (500m)",
    # CEH landcover – 1000m
    "ceh_landcover_wetland_1000m": "Wetland (1000m)",
    "ceh_landcover_upland_heathland_1000m": "Upland heath (1000m)",
    "ceh_landcover_grassland_1000m": "Grassland (1000m)",
    "ceh_landcover_arable_1000m": "Arable (1000m)",
    "ceh_landcover_broadleaved_woodland_1000m": "Broadleaved woodland (1000m)",
    "ceh_landcover_suburban_1000m": "Suburban (1000m)",
    "ceh_landcover_coniferous_woodland_1000m": "Coniferous woodland (1000m)",
    "ceh_landcover_urban_1000m": "Urban (1000m)",
    "ceh_landcover_improved_grassland_1000m": "Improved grassland (1000m)",
    # CEH landcover – 1500m
    "ceh_landcover_wetland_1500m": "Wetland (1500m)",
    "ceh_landcover_upland_heathland_1500m": "Upland heath (1500m)",
    "ceh_landcover_grassland_1500m": "Grassland (1500m)",
    "ceh_landcover_arable_1500m": "Arable (1500m)",
    "ceh_landcover_broadleaved_woodland_1500m": "Broadleaved woodland (1500m)",
    "ceh_landcover_suburban_1500m": "Suburban (1500m)",
    "ceh_landcover_coniferous_woodland_1500m": "Coniferous woodland (1500m)",
    "ceh_landcover_urban_1500m": "Urban (1500m)",
    "ceh_landcover_improved_grassland_1500m": "Improved grassland (1500m)",
    # OS distance
    "os_distance_distance_to_buildings": "Distance to buildings",
    "os_distance_distance_to_minor_roads": "Distance to minor roads",
    "os_distance_distance_to_major_roads": "Distance to major roads",
    "os_distance_distance_to_water": "Distance to water",
    # OS cover
    "os_cover_buildings": "Buildings",
    "os_cover_water": "Water",
    "os_cover_water_500m": "Water (500m)",
    "os_cover_water_1000m": "Water (1000m)",
    "os_cover_water_1500m": "Water (1500m)",
    # BGS
    "bgs_coast_distance_to_coast": "Distance to coast",
    # VOM vegetation height – point
    "vom_vegetation_height_max": "Veg height (max)",
    "vom_vegetation_height_std": "Veg height (std)",
    "vom_vegetation_height_mean": "Veg height (mean)",
    # VOM – 500m
    "vom_vegetation_height_max_500m": "Veg height max (500m)",
    "vom_vegetation_height_std_500m": "Veg height std (500m)",
    "vom_vegetation_height_mean_500m": "Veg height mean (500m)",
    # VOM – 1000m
    "vom_vegetation_height_max_1000m": "Veg height max (1000m)",
    "vom_vegetation_height_std_1000m": "Veg height std (1000m)",
    "vom_vegetation_height_mean_1000m": "Veg height mean (1000m)",
    # VOM – 1500m
    "vom_vegetation_height_max_1500m": "Veg height max (1500m)",
    "vom_vegetation_height_std_1500m": "Veg height std (1500m)",
    "vom_vegetation_height_mean_1500m": "Veg height mean (1500m)",
}


def get_variable_display_name(name: str) -> str:
    """
    Return a short, human-friendly label for an EV/variable name (e.g. for charts or slides).
    Unknown names are returned unchanged.

    Args:
        name: Internal variable name (e.g. ceh_landcover_upland_heathland_1500m).

    Returns:
        Display label (e.g. Upland heath (1500m)), or name if not in the mapping.
    """
    return VARIABLE_DISPLAY_NAMES.get(str(name).strip(), name)