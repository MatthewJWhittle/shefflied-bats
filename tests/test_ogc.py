import pytest
import xarray as xr
import numpy as np

from sdm.data.terrain.core import WCSDownloader


@pytest.fixture
def wcs_test_params() -> dict:
    """Return standard WCS parameters for testing."""
    return {
        "endpoint": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        "coverage_id": "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
        "request_tile_pixels": (100, 100),
    }


@pytest.fixture
def test_bbox() -> tuple:
    """Return a small bounding box in British National Grid."""
    xmin, ymin = 422558, 391118
    size = 100  # 100m square area
    return xmin, ymin, xmin + size, ymin + size


@pytest.fixture
def test_downloader(wcs_test_params):
    """Return a WCSDownloader instance for testing."""
    return WCSDownloader(**wcs_test_params)


def test_downloader_initialization(wcs_test_params):
    """Test basic downloader initialization."""
    downloader = WCSDownloader(**wcs_test_params)
    assert downloader.endpoint == wcs_test_params["endpoint"]
    assert downloader.coverage_id == wcs_test_params["coverage_id"]
    # Check that the downloader was initialized with correct attributes
    assert hasattr(downloader, 'endpoint')
    assert hasattr(downloader, 'coverage_id')
    # Check tile dimensions
    assert hasattr(downloader, 'tile_width')
    assert hasattr(downloader, 'tile_height')


@pytest.mark.asyncio
async def test_memory_download(test_downloader, test_bbox):
    """Test basic download functionality using memory storage."""
    result = await test_downloader.get_coverage(
        bbox=test_bbox,
        resolution=10.0,
    )

    assert isinstance(result, xr.Dataset)
    assert result.sizes["x"] > 0 and result.sizes["y"] > 0
    assert not np.all(np.isnan(result[test_downloader.coverage_id].values))

    result_box = result[test_downloader.coverage_id].rio.bounds()
    assert result_box[0] <= test_bbox[0]
    assert result_box[1] <= test_bbox[1]
    assert result_box[2] >= test_bbox[2]
    assert result_box[3] >= test_bbox[3]


@pytest.mark.asyncio
async def test_download_resolution(test_downloader, test_bbox):
    """Test download functionality with a specified resolution."""
    resolution = 10.0
    result = await test_downloader.get_coverage(
        bbox=test_bbox,
        resolution=resolution,
    )

    result_res = result.rio.resolution()
    assert result_res[0] == resolution
    assert result_res[1] == -resolution


@pytest.mark.asyncio
async def test_download_to_temp_storage(test_bbox, wcs_test_params):
    """Test download functionality using temporary storage."""
    temp_downloader = WCSDownloader(
        endpoint=wcs_test_params["endpoint"],
        coverage_id=wcs_test_params["coverage_id"],
        request_tile_pixels=(10, 10),
        use_temp_storage=True
    )
    
    resolution = 10.0
    result = await temp_downloader.get_coverage(
        bbox=test_bbox,
        resolution=resolution,
    )

    assert isinstance(result, xr.Dataset)
    # Temp tiles are deleted on return; data must already be loaded.
    assert result[temp_downloader.coverage_id].chunks is None
    assert np.isfinite(result[temp_downloader.coverage_id].values).any()

    # check the resolution
    result_res = result.rio.resolution()
    assert result_res[0] == resolution
    assert result_res[1] == -resolution

    # check the bounding box
    result_box = result[temp_downloader.coverage_id].rio.bounds()
    assert result_box[0] <= test_bbox[0]
    assert result_box[1] <= test_bbox[1]
    assert result_box[2] >= test_bbox[2]
    assert result_box[3] >= test_bbox[3]


@pytest.mark.asyncio
async def test_temp_storage_loads_before_tile_cleanup(monkeypatch):
    """Temp tiles must not be the only copy of data after get_coverage returns.

    Rebuild failure mode (#38): ``open_mfdataset`` returns a lazy dask view of
    ``tile_*.tif``, then the ``finally`` block deletes those files. Later reads
    that reopen by path raise ``RasterioIOError``. The returned dataset must
    therefore be fully loaded (or otherwise independent of the temp tiles).
    """
    import rasterio
    from rasterio.transform import from_origin

    coverage_id = "test_coverage_dtm"
    downloader = WCSDownloader.__new__(WCSDownloader)
    downloader.endpoint = "https://example.invalid/wcs"
    downloader.coverage_id = coverage_id
    downloader.tile_width = 4
    downloader.tile_height = 4
    downloader.use_temp_storage = True
    downloader.temp_dir = None
    downloader.axis_labels = ["x", "y"]
    downloader.native_crs = "http://www.opengis.net/def/crs/EPSG/0/27700"

    tile_specs = [
        (420000.0, 380000.0, 420040.0, 380040.0, 1.0),
        (420040.0, 380000.0, 420080.0, 380040.0, 2.0),
    ]

    async def fake_download_tile(session, tile_bbox, resolution):
        minx, miny, maxx, maxy = tile_bbox
        value = next(v for a, b, c, d, v in tile_specs if (a, b, c, d) == tile_bbox)
        assert downloader.temp_dir is not None
        path = downloader.temp_dir / f"tile_{hash(str(tile_bbox))}.tif"
        transform = from_origin(minx, maxy, resolution, resolution)
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        data = np.full((height, width), value, dtype=np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:27700",
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        return str(path)

    monkeypatch.setattr(downloader, "_download_tile", fake_download_tile)

    result = await downloader.get_coverage(
        bbox=(420000.0, 380000.0, 420080.0, 380040.0),
        resolution=10.0,
        max_concurrent=2,
    )

    assert isinstance(result, xr.Dataset)
    assert coverage_id in result.data_vars
    # Temp dir is cleaned in finally; data must not still be a lazy file view.
    assert downloader.temp_dir is None or not downloader.temp_dir.exists()
    assert result[coverage_id].chunks is None, (
        "lazy dask view of deleted temp tiles; load or persist before cleanup"
    )
    values = np.asarray(result[coverage_id].values)
    assert values.size > 0
    assert set(np.unique(values[np.isfinite(values)])).issuperset({1.0, 2.0})


def _wcs_downloader_for_tile_layout(
    tile_pixels: tuple[int, int] = (1024, 1024),
) -> WCSDownloader:
    """Build a WCSDownloader without DescribeCoverage / network."""
    downloader = WCSDownloader.__new__(WCSDownloader)
    downloader.endpoint = "https://example.invalid/wcs"
    downloader.coverage_id = "test_coverage"
    downloader.tile_width, downloader.tile_height = tile_pixels
    downloader.use_temp_storage = False
    downloader.temp_dir = None
    downloader.axis_labels = ["x", "y"]
    downloader.native_crs = "http://www.opengis.net/def/crs/EPSG/0/27700"
    return downloader


def test_wcs_tiles_use_fixed_national_grid_independent_of_study_origin():
    """Shared cells must request the same boxes regardless of study bounds.

    Rebuild failure mode (#38 / 67312d9): tile loops started at the study-area
    corner, so the same ground cell produced different GetCoverage boxes when
    the AOI shifted. Tile edges must fall on multiples of
    ``tile_pixels × resolution`` metres from (0, 0) in EPSG:27700.
    """
    resolution = 10.0
    tile_pixels = (1024, 1024)
    tile_m = tile_pixels[0] * resolution  # 10_240 m
    downloader = _wcs_downloader_for_tile_layout(tile_pixels)

    # Two overlapping study boxes that do not share a SW corner (neither on
    # a national tile edge). Synthetic metres only — not a real site.
    bbox_a = (12_300.0, 45_600.0, 35_000.0, 68_000.0)
    bbox_b = (18_100.0, 50_200.0, 40_500.0, 72_400.0)

    tiles_a = set(downloader.tile_bboxes_for_coverage(bbox_a, resolution))
    tiles_b = set(downloader.tile_bboxes_for_coverage(bbox_b, resolution))
    shared = tiles_a & tiles_b

    assert shared, "overlapping study areas must share at least one national tile"
    for minx, miny, maxx, maxy in shared:
        assert minx % tile_m == 0
        assert miny % tile_m == 0
        assert maxx - minx == tile_m
        assert maxy - miny == tile_m

    # Same shared tile identity from either AOI (set intersection already
    # requires identical request boxes).
    assert shared == {
        t for t in tiles_a
        if t[0] < bbox_b[2] and t[2] > bbox_b[0] and t[1] < bbox_b[3] and t[3] > bbox_b[1]
    }
    assert shared == {
        t for t in tiles_b
        if t[0] < bbox_a[2] and t[2] > bbox_a[0] and t[1] < bbox_a[3] and t[3] > bbox_a[1]
    }

    # These AOIs are deliberately off the national grid so corner-anchored
    # tiling (67312d9) would produce different request boxes.
    def _corner_anchored(bbox):
        minx, miny, maxx, maxy = bbox
        out = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                out.append((x, y, min(x + tile_m, maxx), min(y + tile_m, maxy)))
                y += tile_m
            x += tile_m
        return set(out)

    assert _corner_anchored(bbox_a) != tiles_a
    assert _corner_anchored(bbox_b) != tiles_b

