"""
Spatial data processing functionality.
"""

from .core import (
    calculate_coastal_distance,
    create_study_boundary,
    calculate_terrain_metrics,
    calculate_land_cover_metrics
)

__all__ = [
    'calculate_coastal_distance',
    'create_study_boundary',
    'calculate_terrain_metrics',
    'calculate_land_cover_metrics',
] 