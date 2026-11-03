"""Training boundary simplify must run on the locked numpy (topojson np.in1d crash)."""

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from sdm.commands.modelling.train_sdm_models import simplify_boundary


def test_simplify_boundary_runs_on_locked_numpy():
    """Train calls simplify_boundary before modelling; must not raise on numpy>=2.4.

    topojson 1.9 calls removed np.in1d under numpy 2.4+; training dies at Topology.
    """
    # Coastline-like dissolved ring: enough vertices to hit Topology/toposimplify.
    parts = [
        box(400_000 + i * 2_000, 350_000 + (i % 3) * 500, 402_000 + i * 2_000, 480_000)
        for i in range(60)
    ]
    boundary = gpd.GeoDataFrame(geometry=[unary_union(parts)], crs="EPSG:27700")
    area_before = float(boundary.geometry.area.sum())

    assert tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 4)
    simplified = simplify_boundary(boundary, tolerance=100)

    assert len(simplified) == 1
    assert not simplified.geometry.isna().any()
    assert simplified.geometry.is_valid.all()
    assert simplified.geometry.area.sum() > 0.5 * area_before
    assert simplified.crs == boundary.crs
