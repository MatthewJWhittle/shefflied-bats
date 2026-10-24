"""
Core data processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
import elapid as ela

logger = logging.getLogger(__name__)


def annotate_points(
    bats: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    ev_raster: Union[str, Path],
    ev_columns: List[str]
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Annotate bat and background points with environmental variables.
    
    Args:
        bats: GeoDataFrame containing bat occurrence points
        background: GeoDataFrame containing background points
        ev_raster: Path to environmental variables raster
        ev_columns: List of environmental variable column names
        
    Returns:
        Tuple of (annotated bat points, annotated background points)
    """
    try:
        bats_ant = ela.annotate(bats, str(ev_raster), labels=ev_columns)
        background_ant = ela.annotate(background, str(ev_raster), labels=ev_columns)
        logger.debug("Successfully annotated points with environmental variables")
        return bats_ant, background_ant
    except Exception as e:
        logger.error(f"Error annotating points: {e}")
        raise
