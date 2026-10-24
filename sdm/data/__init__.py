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

from .processing import annotate_points

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
    'annotate_points',
    
    # Spatial
    'create_boundary',
    'create_study_boundary',
    
    # Terrain
    'create_terrain_wcs_downloaders',
    'WCSDownloader',
]
