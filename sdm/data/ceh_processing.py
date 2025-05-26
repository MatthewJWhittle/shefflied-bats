from typing import Dict, List

def get_ceh_land_cover_codes_v2023() -> Dict[str, str]: # Renamed for clarity and version
    """
    Return a mapping of CEH land cover codes (e.g., for gblcm2023_10m.tif) to descriptive names.
    Source: Derived from inspecting data or accompanying metadata for CEH Global Land Cover 2023.
    Users should verify these against official CEH documentation for the specific product used.
    """
    return {
        "1": "Broadleaved woodland",
        "2": "Coniferous woodland",
        "3": "Arable",
        "4": "Improved grassland",
        "5": "Neutral grassland", # Often grouped under "Other Grassland" or specific types
        "6": "Calcareous grassland",
        "7": "Acid grassland",
        "8": "Fen, Marsh and Swamp",
        "9": "Heather and shrub", # Often grouped under "Heathland" or specific types
        "10": "Heather grassland",
        "11": "Bog", # Often grouped under "Peatland" or specific types
        "12": "Inland rock",
        "13": "Saltwater",
        "14": "Freshwater",
        "15": "Supralittoral rock",
        "16": "Supralittoral sediment",
        "17": "Littoral rock",
        "18": "Littoral sediment",
        "19": "Saltmarsh",
        "20": "Urban",
        "21": "Suburban", # Or "Built-up areas" depending on specific CEH product version
        # Add other codes as per the specific CEH product documentation if needed
    }

def define_broad_habitat_categories() -> Dict[str, List[str]]: # Renamed for clarity
    """
    Define broad habitat categories as aggregations of specific land cover types
    based on the descriptive names from get_ceh_land_cover_codes_v2023().
    Adjust these mappings based on project-specific aggregation needs.
    """
    # Uses the descriptive names returned by get_ceh_land_cover_codes_v2023()
    return {
        "Woodland": [
            "Broadleaved woodland",
            "Coniferous woodland"
        ],
        "Arable_ImprovedGrass": [
            "Arable",
            "Improved grassland"
        ],
        "Other_Grassland": [
            "Neutral grassland",
            "Calcareous grassland",
            "Acid grassland",
        ],
        "Heathland_Shrub": [
            "Heather and shrub", 
            "Heather grassland"
        ],
        "Wetland_Bog": [
            "Bog", 
            "Fen, Marsh and Swamp",
            "Freshwater" # Grouping freshwater bodies here, could be separate
        ],
        "Coastal_Marine": [
            "Saltwater",
            "Supralittoral rock",
            "Supralittoral sediment",
            "Littoral rock",
            "Littoral sediment",
            "Saltmarsh",
        ],
        "BuiltEnvironment": [
            "Urban",
            "Suburban"
        ],
        "Other_Bare": [
            "Inland rock"
        ]
        # Note: Some original categories like "Inland rock", "Freshwater" might be dropped
        # or assigned to broader groups depending on modeling needs.
        # The original script dropped "Inland rock", "Marine, Littoral", "Freshwater" later.
        # This definition provides a more comprehensive grouping first.
    } 