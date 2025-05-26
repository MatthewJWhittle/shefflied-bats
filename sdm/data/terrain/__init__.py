"""
Terrain data processing functionality.
"""

from .core import (
    calculate_slope,
    calculate_aspect,
    calculate_terrain_ruggedness,
    calculate_terrain_position
)

__all__ = [
    'calculate_slope',
    'calculate_aspect',
    'calculate_terrain_ruggedness',
    'calculate_terrain_position',
] 