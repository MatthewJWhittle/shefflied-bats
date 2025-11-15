# tests/test_twi_fast_valid.py
import math
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# Import the D8-based API (update the path to match your package)
from sdm.data.terrain.stats import (
    priority_flood_fill_numba,
    horn_tan_slope,
    d8_flow_dirs,
    resolve_flats_bfs,
    flow_accumulation_d8,
    twi_from_array,
    compute_twi_raster,
)

# ---------- helpers ----------
def plane_dem(h, w, ax, ay, base=0.0):
    """
    z = base + ax*x + ay*y
    For a downslope direction at angle theta (0=E, CCW):
    ax = -cos(theta), ay = -sin(theta).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return base + ax * xx + ay * yy


# ---------- tests ----------

def test_priority_flood_fills_pit():
    dem = np.array([[10,10,10],
                    [10, 0,10],
                    [10,10,10]], dtype=np.float32)
    filled = priority_flood_fill_numba(dem)
    assert np.isclose(filled[1,1], 10.0)        # pit raised to rim
    assert np.all(filled[dem == 10] >= 10.0)    # rim not lowered


def test_horn_tan_slope_on_uniform_plane():
    # z = -2*x -1*y -> tan(beta)=sqrt(5)
    dem = plane_dem(9, 9, ax=-2.0, ay=-1.0)
    tanb = horn_tan_slope(dem, cell=1.0)
    inner = (slice(2,7), slice(2,7))
    assert np.allclose(np.mean(tanb[inner]), math.sqrt(5.0), atol=1e-5)


def test_d8_dirs_on_east_sloping_plane_are_east():
    dem = plane_dem(9, 9, ax=-1.0, ay=0.0)  # slope to the east
    filled = priority_flood_fill_numba(dem)
    dirs = d8_flow_dirs(filled, cell=1.0)
    # Our D8 ordering is [N, NE, E, SE, S, SW, W, NW], so E == 2
    interior = dirs[1:-1, 1:-1]
    assert np.all(interior == 2)


def test_flats_resolver_routes_plateau_toward_outlet():
    # Flat 100 m plateau with one lower cell on the east edge
    dem = np.full((21, 21), 100.0, dtype=np.float32)
    dem[:, -1] = 99.0  # outlet along east edge
    filled = priority_flood_fill_numba(dem)
    dirs = d8_flow_dirs(filled, cell=1.0)
    # interior flats will be -1 before resolving
    assert (dirs[1:-1, 1:-2] == -1).any()
    dirs2 = resolve_flats_bfs(filled, dirs, tol=0.0)
    # central pixel should now have a direction
    assert dirs2[10, 10] != -1
    # and most of the plateau interior should be assigned
    frac_assigned = np.mean(dirs2[1:-1, 1:-2] != -1)
    assert frac_assigned > 0.95


def test_accumulation_conservation_on_plane():
    dem = plane_dem(15, 15, ax=-1.0, ay=0.0)
    filled = priority_flood_fill_numba(dem)
    dirs = d8_flow_dirs(filled, cell=1.0)
    acc = flow_accumulation_d8(dirs)
    sinks = (dirs == -1)
    # Sum of accumulation at sinks equals number of valid cells
    assert np.isclose(acc[sinks].sum(), np.isfinite(dem).sum(), rtol=1e-6)


def test_twi_increases_with_acc_when_slope_constant():
    # On a uniform slope, tan(beta) is constant -> higher acc => higher TWI along a row
    dem = plane_dem(21, 21, ax=-1.0, ay=0.0)
    twi = twi_from_array(dem, cellsize=1.0, slope_eps=1e-6, do_fill=True, flats_tol=0.0)
    row = 10
    assert np.isfinite(twi[row, 1])
    assert twi[row, -2] > twi[row, 2]


def test_nodata_propagates():
    dem = plane_dem(9, 9, ax=-1.0, ay=0.0)
    dem[0:3, 0:3] = np.nan
    twi = twi_from_array(dem, cellsize=1.0, slope_eps=1e-6, do_fill=True)
    assert np.isnan(twi[0:3, 0:3]).all()


@pytest.mark.skipif(rasterio is None, reason="rasterio not installed")
def test_raster_io_roundtrip(tmp_path):
    dem = plane_dem(64, 64, ax=-1.0, ay=0.0).astype(np.float32)
    transform = from_origin(0.0, 0.0, 10.0, 10.0)
    in_tif = tmp_path / "dem.tif"
    out_tif = tmp_path / "twi.tif"

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": np.nan,
        "compress": "LZW",
    }
    with rasterio.open(in_tif, "w", **profile) as dst:
        dst.write(dem, 1)

    compute_twi_raster(str(in_tif), str(out_tif), slope_eps=1e-6, flats_tol=0.0)

    with rasterio.open(out_tif) as src:
        twi = src.read(1)
        assert twi.shape == dem.shape
        # Finite wherever input was finite
        mask = np.isfinite(dem)
        assert np.isfinite(twi[mask]).all()
