"""
Data loading and processing functionality for SDM.
"""

from .loaders import (
    ClimateData,
    ImageTileDownloader,
    load_os_shps,
    load_bat_data,
    load_background_points
)

from .processing import (
    merge_environmental_layers,
    process_occurrence_data,
    process_background_data,
    extract_environmental_data
)

from .spatial import (
    create_boundary,
    create_study_boundary,
)

from .terrain import (
    create_terrain_wcs_downloaders,
    WCSDownloader,
)

__all__ = [
    # Loaders
    'ClimateData',
    'ImageTileDownloader',
    'load_os_shps',
    'load_bat_data',
    'load_background_points',
    
    # Processing
    'merge_environmental_layers',
    'process_occurrence_data',
    'process_background_data',
    'extract_environmental_data',
    
    # Spatial
    'create_boundary',
    'create_study_boundary',
    
    # Terrain
    'create_terrain_wcs_downloaders',
    'WCSDownloader',
]
