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
from .wcs_tilearray import (
    EA_DSM_COVERAGE_ID,
    EA_DSM_WCS_ENDPOINT,
    EA_DTM_COVERAGE_ID,
    EA_DTM_WCS_ENDPOINT,
    EA_VOM_COVERAGE_ID,
    EA_VOM_WCS_ENDPOINT,
    fetch_wcs_dataarray,
    fetch_wcs_dataset,
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
    
    # Terrain / WCS
    'create_terrain_wcs_downloaders',
    'WCSDownloader',
    'EA_DSM_COVERAGE_ID',
    'EA_DSM_WCS_ENDPOINT',
    'EA_DTM_COVERAGE_ID',
    'EA_DTM_WCS_ENDPOINT',
    'EA_VOM_COVERAGE_ID',
    'EA_VOM_WCS_ENDPOINT',
    'fetch_wcs_dataarray',
    'fetch_wcs_dataset',
]
