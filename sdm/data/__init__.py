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
    calculate_coastal_distance,
    create_study_boundary,
    calculate_terrain_metrics,
    calculate_land_cover_metrics
)

from .terrain import (
    calculate_slope,
    calculate_aspect,
    calculate_terrain_ruggedness,
    calculate_terrain_position
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
    'calculate_coastal_distance',
    'create_study_boundary',
    'calculate_terrain_metrics',
    'calculate_land_cover_metrics',
    
    # Terrain
    'calculate_slope',
    'calculate_aspect',
    'calculate_terrain_ruggedness',
    'calculate_terrain_position',
]



