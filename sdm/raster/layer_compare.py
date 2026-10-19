"""Compare rebuilt environmental layers against a current reference set.

Pass marks follow the revised architecture review on issue #38
(comment that replaces earlier per-layer tolerance rules): closeness,
not exactness. Grid alignment is a hard precondition; continuous and
categorical layers are judged by share-close; intended method changes
are flagged rather than failed.

Comparison of multi-band stacks uses lazy rioxarray + dask arrays and a
single vectorised ``.compute()`` — no per-cell / per-tile Python loops.

Model AUC checks remain stub hooks only.
"""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import dask
import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers .rio accessor
import rioxarray as rxr
import xarray as xr
from scipy import ndimage

from sdm.utils.step_metrics import process_tree_rss_bytes
from sdm.utils.text_utils import tidy_variable_name

RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff"}
DEFAULT_STACK_NAME = "evs-to-model.tif"
SHARE_WITHIN_TOLERANCE_PASS = 0.95
CONTINUOUS_RANGE_FRACTION = 0.01
# Spatial chunks for lazy rioxarray / dask opens.
DEFAULT_CHUNKS: dict[str, int | str] = {"x": 512, "y": 512}

# Compactness heuristics for out-of-tolerance cells.
MAX_COMPONENTS_FOR_COMPACT = 8
MIN_BBOX_FILL_RATIO = 0.25

# ---------------------------------------------------------------------------
# Value-kind name rules (architecture review on #38 / #43).
#
# Exact / categorical: in-cell category and percentage-cover layers stored as
# whole numbers (e.g. ceh_landcover_arable, os_cover_buildings).
#
# Continuous (1% of layer range): neighbourhood averages named *_NNNm, plus
# distances, heights, and climate.
# ---------------------------------------------------------------------------
EXACT_MATCH_NAME_PREFIXES: tuple[str, ...] = ("os_cover",)
EXACT_MATCH_NAME_INFIXES: tuple[str, ...] = ("_cover_",)
NEIGHBOURHOOD_BUFFER_RE = re.compile(r"_(500|1000|1500)m$")
CEH_NEIGHBOURHOOD_BUFFER_RE = NEIGHBOURHOOD_BUFFER_RE
PREFERRED_STACK_NAMES = frozenset({"evs-to-model.tif", "evs-to-model.tiff"})


class DuplicateLayerError(ValueError):
    """Raised when the same layer name appears in multiple raster files."""


class LayerValueKind(str, Enum):
    """How cell values are compared."""

    EXACT = "exact"  # category / percentage-cover
    CONTINUOUS = "continuous"  # distances, heights, climate


class LayerVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    FLAGGED = "flagged"  # intended method change (configurable name/prefix list)
    FLAGGED_SOURCE_CHANGE = "flagged_source_change"
    MISSING = "missing"


@dataclass(frozen=True)
class LayerRef:
    """One named layer slice inside a GeoTIFF (possibly multi-band)."""

    name: str
    path: Path
    band_index: int  # 1-based


@dataclass
class GridCheck:
    """Grid / shape precondition (closeness tests are separate)."""

    name_match: bool
    projection_match: bool
    cell_size_match: bool
    origin_match: bool
    extent_match: bool
    nodata_value_match: bool = True
    detail: str = ""
    aligned: bool = False

    @property
    def ok(self) -> bool:
        # No-data cell mismatches are closeness metrics, not grid failures.
        return all(
            (
                self.name_match,
                self.projection_match,
                self.cell_size_match,
                self.origin_match,
                self.extent_match,
            )
        )

    def summary(self) -> str:
        if self.ok:
            return "ok (aligned)" if self.aligned else "ok"
        parts = []
        for label, value in (
            ("name", self.name_match),
            ("projection", self.projection_match),
            ("cell_size", self.cell_size_match),
            ("origin", self.origin_match),
            ("extent", self.extent_match),
        ):
            if not value:
                parts.append(label)
        base = "mismatch:" + ",".join(parts)
        return f"{base}; {self.detail}" if self.detail else base


@dataclass
class PatchInfo:
    """One connected component of out-of-tolerance cells."""

    component_id: int
    cell_count: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    bbox_fill_ratio: float


@dataclass
class LayerCompareResult:
    layer_name: str
    value_kind: LayerValueKind
    grid_check: str
    grid_ok: bool
    share_within_tolerance: Optional[float]
    n_nodata_mismatch: Optional[int] = None
    # Continuous: |diff| / layer range over jointly valid cells.
    norm_abs_diff_mean: Optional[float] = None
    norm_abs_diff_median: Optional[float] = None
    norm_abs_diff_p95: Optional[float] = None
    norm_abs_diff_max: Optional[float] = None
    correlation: Optional[float] = None
    # Raw absolute difference (jointly valid); useful for maps / debugging.
    max_difference: Optional[float] = None
    difference_map_path: str = ""
    differences_local: Optional[bool] = None
    n_mismatch_components: Optional[int] = None
    patches: list[PatchInfo] = field(default_factory=list)
    patch_table_path: str = ""
    source_version_current: str = ""
    source_version_rebuilt: str = ""
    reason: str = ""
    verdict: LayerVerdict = LayerVerdict.FAIL
    notes: str = ""


@dataclass
class ModelCompareHookResult:
    """Placeholder for post-rebuild model checks (AUC / cut-off map).

    Not executed by the layer comparison CLI; recorded for later rebuild runs.
    """

    model_id: str
    auc_current: Optional[float] = None
    auc_rebuilt: Optional[float] = None
    share_cutoff_switch: Optional[float] = None
    verdict: str = "not_run"
    notes: str = (
        "Stub only — retrain and compare after a real rebuilt layer set "
        "(architecture review items 5–7 on issue #38)."
    )


@dataclass
class CompareRunSummary:
    layers: list[LayerCompareResult] = field(default_factory=list)
    model_hooks: list[ModelCompareHookResult] = field(default_factory=list)
    duration_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None


@dataclass
class StackCompareTiming:
    """Wall time and peak RSS for one vectorised stack comparison."""

    duration_seconds: float
    peak_rss_bytes: Optional[int]
    n_bands: int
    height: int
    width: int
    note: str = ""


def is_neighbourhood_average(layer_name: str) -> bool:
    """True for neighbourhood buffer averages (*_500m / *_1000m / *_1500m)."""
    name = tidy_variable_name(layer_name)
    return bool(NEIGHBOURHOOD_BUFFER_RE.search(name))


def is_ceh_neighbourhood_average(layer_name: str) -> bool:
    """True for CEH landcover neighbourhood averages (*_500m / *_1000m / *_1500m)."""
    name = tidy_variable_name(layer_name)
    return "landcover" in name and is_neighbourhood_average(name)


def is_exact_match_layer(layer_name: str) -> bool:
    """Return True when the layer is categorical / percentage (cell match).

    Whole-number in-cell categories and percentage-cover layers only.
    Any ``*_NNNm`` neighbourhood average is continuous (1% of range).
    """
    name = tidy_variable_name(layer_name)
    if is_neighbourhood_average(name):
        return False
    if "landcover" in name:
        return True
    if any(name.startswith(prefix) for prefix in EXACT_MATCH_NAME_PREFIXES):
        return True
    if any(infix in name for infix in EXACT_MATCH_NAME_INFIXES):
        return True
    return False


def classify_value_kind(layer_name: str) -> LayerValueKind:
    """Whole-number category/percentage → exact; continuous otherwise."""
    if is_exact_match_layer(layer_name):
        return LayerValueKind.EXACT
    return LayerValueKind.CONTINUOUS


def matches_flagged_layer(
    layer_name: str,
    *,
    flag_names: Sequence[str] = (),
    flag_prefixes: Sequence[str] = (),
) -> bool:
    """True when the layer is an intended method-change (flagged, not failed)."""
    name = tidy_variable_name(layer_name)
    named = {tidy_variable_name(n) for n in flag_names}
    if name in named:
        return True
    for prefix in flag_prefixes:
        tidied = tidy_variable_name(prefix)
        if tidied and name.startswith(tidied):
            return True
    return False


def _is_preferred_stack_path(path: Path) -> bool:
    return Path(path).name.lower() in PREFERRED_STACK_NAMES


def _iter_raster_paths(
    folder: Path,
    *,
    stack_only: bool = False,
    file_glob: Optional[str] = None,
) -> list[Path]:
    folder = Path(folder)
    if stack_only and file_glob:
        raise ValueError("Use only one of stack_only and file_glob")
    if stack_only:
        paths = [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.name.lower() in {"evs-to-model.tif", "evs-to-model.tiff"}
        ]
        return sorted(paths)
    if file_glob:
        if "**" in file_glob or "/" in file_glob or "\\" in file_glob:
            paths = [p for p in folder.glob(file_glob) if p.is_file()]
            if not paths:
                paths = [p for p in folder.rglob(file_glob) if p.is_file()]
        else:
            paths = [p for p in folder.rglob(file_glob) if p.is_file()]
        return sorted(paths)
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES
    )


def discover_layers(
    folder: Path,
    *,
    stack_only: bool = False,
    file_glob: Optional[str] = None,
) -> dict[str, LayerRef]:
    """Index named layers under a folder (single-band files and multi-band stacks).

    When the same layer name appears in more than one file, prefer
    ``evs-to-model.tif`` and emit a warning. Ambiguous duplicates (neither or
    both sides are the preferred stack) raise :class:`DuplicateLayerError`.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Layer folder not found: {folder}")

    found: dict[str, LayerRef] = {}
    for path in _iter_raster_paths(folder, stack_only=stack_only, file_glob=file_glob):
        with rasterio.open(path) as src:
            for band_index in range(1, src.count + 1):
                description = None
                if src.descriptions and src.descriptions[band_index - 1]:
                    description = src.descriptions[band_index - 1]
                if description:
                    name = tidy_variable_name(description)
                elif src.count == 1:
                    name = tidy_variable_name(path.stem)
                else:
                    name = tidy_variable_name(f"{path.stem}_band_{band_index}")
                ref = LayerRef(name=name, path=path, band_index=band_index)
                existing = found.get(name)
                if existing is None:
                    found[name] = ref
                    continue
                existing_preferred = _is_preferred_stack_path(existing.path)
                new_preferred = _is_preferred_stack_path(path)
                if new_preferred and not existing_preferred:
                    logging.warning(
                        "Duplicate layer %r: preferring %s over %s",
                        name,
                        path,
                        existing.path,
                    )
                    found[name] = ref
                    continue
                if existing_preferred and not new_preferred:
                    logging.warning(
                        "Duplicate layer %r: keeping preferred stack %s; ignoring %s",
                        name,
                        existing.path,
                        path,
                    )
                    continue
                raise DuplicateLayerError(
                    f"Duplicate layer name {name!r} in {existing.path} and {path}. "
                    "Use --stack-only / --glob to select one file, or remove the "
                    "extra raster so names are unique."
                )
    return found


def ordered_layer_names(
    priority_names: Sequence[str],
    discovered_names: Iterable[str],
) -> list[str]:
    """Priority list first (as given), then remaining discovered names sorted."""
    priority = [tidy_variable_name(n) for n in priority_names]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in priority:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(set(discovered_names)):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _crs_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return a == b
    except Exception:
        return str(a) == str(b)


def _nodata_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return bool(np.isclose(float(a), float(b), equal_nan=True))
    except (TypeError, ValueError):
        return a == b


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """True where cells are valid (GDAL / rasterio ``read(masked=True)``).

    A cell is invalid only when it is NaN, or exactly equal to the nodata
    sentinel. Approximate (``isclose``) matching wrongly treats near-nodata
    values such as ``-9999.05`` as missing and is much slower on large stacks.
    """
    if nodata is None:
        if np.issubdtype(data.dtype, np.floating):
            return ~np.isnan(data)
        return np.ones(data.shape, dtype=bool)

    try:
        nodata_is_nan = bool(np.isnan(nodata))
    except (TypeError, ValueError):
        nodata_is_nan = False

    if nodata_is_nan:
        return ~np.isnan(data)

    mask = data != nodata
    if np.issubdtype(data.dtype, np.floating):
        mask &= ~np.isnan(data)
    return mask


def _open_band_lazy(ref: LayerRef, *, chunks: Mapping[str, int | str]) -> xr.DataArray:
    """Open one named band as a 2-D lazy DataArray (y, x) with dask chunks."""
    da = rxr.open_rasterio(ref.path, chunks=dict(chunks), masked=True)
    if "band" in da.dims:
        da = da.isel(band=ref.band_index - 1, drop=True)
    da = da.squeeze(drop=True)
    # Standardise spatial dim names for concat / ops.
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "y"
    if "longitude" in da.dims:
        rename["longitude"] = "x"
    if rename:
        da = da.rename(rename)
    da.name = ref.name
    return da


def open_layer_stack(
    refs: Sequence[LayerRef],
    *,
    chunks: Mapping[str, int | str] | None = None,
) -> xr.DataArray:
    """Lazy multi-band DataArray (band, y, x) from ordered layer refs."""
    if not refs:
        raise ValueError("open_layer_stack requires at least one layer ref")
    chunk_map = dict(chunks or DEFAULT_CHUNKS)
    bands = [_open_band_lazy(ref, chunks=chunk_map) for ref in refs]
    stacked = xr.concat(bands, dim="band")
    stacked = stacked.assign_coords(band=[ref.name for ref in refs])
    return stacked


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    for pair in (("y", "x"), ("latitude", "longitude")):
        if all(d in da.dims for d in pair):
            return pair
    raise ValueError(f"Cannot find spatial dims on DataArray with dims {da.dims}")


def check_grid_arrays(
    current: xr.DataArray,
    rebuilt: xr.DataArray,
    *,
    expected_names: Sequence[str] | None = None,
) -> GridCheck:
    """Compare CRS / transform / shape between two (possibly multi-band) arrays."""
    cur_names = (
        list(current.coords["band"].values)
        if "band" in current.dims
        else [str(current.name or "")]
    )
    reb_names = (
        list(rebuilt.coords["band"].values)
        if "band" in rebuilt.dims
        else [str(rebuilt.name or "")]
    )
    if expected_names is not None:
        expected = [tidy_variable_name(n) for n in expected_names]
        name_match = (
            [tidy_variable_name(str(n)) for n in cur_names] == expected
            and [tidy_variable_name(str(n)) for n in reb_names] == expected
        )
    else:
        name_match = [tidy_variable_name(str(n)) for n in cur_names] == [
            tidy_variable_name(str(n)) for n in reb_names
        ]

    try:
        cur_crs = current.rio.crs
        reb_crs = rebuilt.rio.crs
    except Exception:
        cur_crs = reb_crs = None
    projection_match = _crs_equal(cur_crs, reb_crs)

    try:
        cur_res = current.rio.resolution()
        reb_res = rebuilt.rio.resolution()
        cell_size_match = bool(np.allclose(cur_res, reb_res, rtol=0.0, atol=0.0))
    except Exception:
        cell_size_match = False

    try:
        cur_t = current.rio.transform()
        reb_t = rebuilt.rio.transform()
        origin_match = bool(
            np.isclose(cur_t.c, reb_t.c, rtol=0.0, atol=0.0)
            and np.isclose(cur_t.f, reb_t.f, rtol=0.0, atol=0.0)
        )
    except Exception:
        origin_match = False

    y_dim, x_dim = _spatial_dims(current)
    try:
        ry_dim, rx_dim = _spatial_dims(rebuilt)
        extent_match = (
            current.sizes[y_dim] == rebuilt.sizes[ry_dim]
            and current.sizes[x_dim] == rebuilt.sizes[rx_dim]
            and bool(
                np.allclose(
                    current.rio.bounds(),
                    rebuilt.rio.bounds(),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
    except Exception:
        extent_match = False

    try:
        nodata_value_match = _nodata_equal(current.rio.nodata, rebuilt.rio.nodata)
    except Exception:
        nodata_value_match = False

    return GridCheck(
        name_match=name_match,
        projection_match=projection_match,
        cell_size_match=cell_size_match,
        origin_match=origin_match,
        extent_match=extent_match,
        nodata_value_match=nodata_value_match,
    )


def check_grid(
    current_meta: Mapping,
    rebuilt_meta: Mapping,
    *,
    expected_name: str,
) -> GridCheck:
    """Grid check from rasterio-style metadata dicts (single-layer helpers/tests)."""
    cur_name = tidy_variable_name(current_meta.get("description") or current_meta["name"])
    reb_name = tidy_variable_name(rebuilt_meta.get("description") or rebuilt_meta["name"])
    expected = tidy_variable_name(expected_name)

    name_match = cur_name == expected and reb_name == expected
    projection_match = _crs_equal(current_meta["crs"], rebuilt_meta["crs"])
    cell_size_match = bool(
        np.allclose(current_meta["res"], rebuilt_meta["res"], rtol=0.0, atol=0.0)
    )
    cur_t = current_meta["transform"]
    reb_t = rebuilt_meta["transform"]
    origin_match = bool(
        np.isclose(cur_t.c, reb_t.c, rtol=0.0, atol=0.0)
        and np.isclose(cur_t.f, reb_t.f, rtol=0.0, atol=0.0)
    )
    extent_match = (
        current_meta["width"] == rebuilt_meta["width"]
        and current_meta["height"] == rebuilt_meta["height"]
        and bool(
            np.allclose(
                [
                    current_meta["bounds"].left,
                    current_meta["bounds"].bottom,
                    current_meta["bounds"].right,
                    current_meta["bounds"].top,
                ],
                [
                    rebuilt_meta["bounds"].left,
                    rebuilt_meta["bounds"].bottom,
                    rebuilt_meta["bounds"].right,
                    rebuilt_meta["bounds"].top,
                ],
                rtol=0.0,
                atol=0.0,
            )
        )
    )
    nodata_value_match = _nodata_equal(current_meta["nodata"], rebuilt_meta["nodata"])
    return GridCheck(
        name_match=name_match,
        projection_match=projection_match,
        cell_size_match=cell_size_match,
        origin_match=origin_match,
        extent_match=extent_match,
        nodata_value_match=nodata_value_match,
    )


def align_to_reference(rebuilt: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Crop / snap rebuilt onto the reference grid (``rio.reproject_match``)."""
    return rebuilt.rio.reproject_match(reference)


def analyze_mismatch_patches(
    out_of_tolerance: np.ndarray,
    jointly_valid: np.ndarray | None = None,
) -> tuple[bool, list[PatchInfo], int]:
    """Return (differences_local, patches, n_components).

    ``out_of_tolerance`` should already include nodata-mismatch cells that count
    as not-close. When ``jointly_valid`` is given it is AND-ed in (legacy tests).
    """
    mismatch = np.asarray(out_of_tolerance, dtype=bool)
    if jointly_valid is not None:
        mismatch = mismatch & np.asarray(jointly_valid, dtype=bool)
    n_mismatch = int(mismatch.sum())
    if n_mismatch == 0:
        return True, [], 0

    labeled, n_components = ndimage.label(mismatch)
    objects = ndimage.find_objects(labeled)
    patches: list[PatchInfo] = []
    sparse_component = False

    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        region = labeled[slc]
        cell_count = int((region == component_id).sum())
        row_min = int(slc[0].start)
        row_max = int(slc[0].stop - 1)
        col_min = int(slc[1].start)
        col_max = int(slc[1].stop - 1)
        bbox_h = row_max - row_min + 1
        bbox_w = col_max - col_min + 1
        bbox_area = max(bbox_h * bbox_w, 1)
        fill_ratio = cell_count / bbox_area
        patches.append(
            PatchInfo(
                component_id=component_id,
                cell_count=cell_count,
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                bbox_fill_ratio=float(fill_ratio),
            )
        )
        if cell_count > 1 and fill_ratio < MIN_BBOX_FILL_RATIO:
            sparse_component = True

    too_many = n_components > MAX_COMPONENTS_FOR_COMPACT
    differences_local = not too_many and not sparse_component
    return differences_local, patches, int(n_components)


def _build_closeness_graph(
    current: xr.DataArray,
    rebuilt: xr.DataArray,
    *,
    value_kinds: Sequence[LayerValueKind],
) -> xr.Dataset:
    """Lazy vectorised closeness metrics over (band, y, x) — call ``.compute()`` once."""
    y_dim, x_dim = _spatial_dims(current)
    spatial = (y_dim, x_dim)

    cur = current.astype("float64")
    reb = rebuilt.astype("float64")

    cur_valid = cur.notnull()
    reb_valid = reb.notnull()
    both_valid = cur_valid & reb_valid
    # Empty in one, present in the other → not-close; both empty → excluded.
    nodata_mismatch = cur_valid ^ reb_valid
    comparable = cur_valid | reb_valid

    abs_diff = abs(reb - cur)
    layer_min = cur.where(both_valid).min(dim=spatial, skipna=True)
    layer_max = cur.where(both_valid).max(dim=spatial, skipna=True)
    layer_range = (layer_max - layer_min).clip(min=0)

    is_exact = xr.DataArray(
        [kind is LayerValueKind.EXACT for kind in value_kinds],
        dims=("band",),
        coords={"band": current.coords["band"]},
    )
    is_continuous = ~is_exact

    # Denominator only for continuous bands; exact bands skip normalised stats.
    safe_range = xr.where((layer_range > 0) & is_continuous, layer_range, 1.0)
    norm_abs = xr.where(is_continuous, abs_diff / safe_range, np.nan)

    exact_close = both_valid & (cur == reb)
    continuous_close = both_valid & (norm_abs <= CONTINUOUS_RANGE_FRACTION)
    # Zero-range continuous: only exact equality counts as close.
    continuous_close = continuous_close | (
        both_valid & is_continuous & (layer_range == 0) & (abs_diff == 0)
    )
    close = xr.where(is_exact, exact_close, continuous_close)
    # Nodata mismatches are never close.
    close = close & ~nodata_mismatch

    comparable_count = comparable.sum(dim=spatial)
    share = close.sum(dim=spatial) / comparable_count.clip(min=1)
    share = xr.where(comparable_count == 0, 1.0, share)

    norm_for_stats = norm_abs.where(both_valid & is_continuous)
    corr = xr.corr(cur.where(both_valid), reb.where(both_valid), dim=spatial)

    not_close = comparable & ~close

    return xr.Dataset(
        {
            "share_within_tolerance": share,
            "n_nodata_mismatch": nodata_mismatch.sum(dim=spatial),
            "norm_abs_diff_mean": norm_for_stats.mean(dim=spatial, skipna=True),
            "norm_abs_diff_median": norm_for_stats.median(dim=spatial, skipna=True),
            "norm_abs_diff_p95": norm_for_stats.quantile(
                0.95, dim=spatial, skipna=True
            ),
            "norm_abs_diff_max": norm_for_stats.max(dim=spatial, skipna=True),
            "correlation": corr,
            "max_difference": abs_diff.where(both_valid).max(dim=spatial, skipna=True),
            "not_close": not_close,
            "both_valid": both_valid,
        }
    )


def write_difference_map(
    current: np.ndarray,
    rebuilt: np.ndarray,
    *,
    profile: Mapping,
    nodata,
    out_path: Path,
) -> Path:
    """Write abs(rebuilt - current); nodata where either input is nodata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cur_valid = _valid_mask(current, nodata)
    reb_valid = _valid_mask(rebuilt, nodata)
    jointly_valid = cur_valid & reb_valid

    out_nodata = -9999.0
    diff = np.full(current.shape, out_nodata, dtype=np.float32)
    diff[jointly_valid] = np.abs(
        current.astype(np.float64)[jointly_valid] - rebuilt.astype(np.float64)[jointly_valid]
    ).astype(np.float32)

    out_profile = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": 1,
        "dtype": "float32",
        "crs": profile.get("crs"),
        "transform": profile.get("transform"),
        "nodata": out_nodata,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(diff, 1)
        dst.descriptions = ["abs_difference"]
    return out_path


def _write_difference_map_from_arrays(
    current_band: xr.DataArray,
    rebuilt_band: xr.DataArray,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    both = current_band.notnull() & rebuilt_band.notnull()
    abs_diff = abs(rebuilt_band.astype("float64") - current_band.astype("float64"))
    out_nodata = -9999.0
    written = abs_diff.where(both, other=out_nodata).astype("float32")
    written = written.rio.write_nodata(out_nodata, inplace=False)
    written.rio.to_raster(out_path, compress="deflate")
    return out_path


def decide_verdict(
    *,
    grid_ok: bool,
    share_within: Optional[float],
    differences_local: Optional[bool],
    source_changed: bool,
    method_flagged: bool,
    both_present: bool,
) -> LayerVerdict:
    if not both_present:
        return LayerVerdict.MISSING
    if not grid_ok:
        return LayerVerdict.FAIL
    if share_within is None:
        return LayerVerdict.FAIL
    # Intended method changes (e.g. OS distance after #42): report stats, don't fail.
    if method_flagged:
        return LayerVerdict.FLAGGED
    if source_changed:
        return LayerVerdict.FLAGGED_SOURCE_CHANGE
    values_ok = share_within >= SHARE_WITHIN_TOLERANCE_PASS
    # Local-patch check applies when there are not-close cells; None means none.
    local_ok = differences_local is True or differences_local is None
    if values_ok and local_ok:
        return LayerVerdict.PASS
    return LayerVerdict.FAIL


def compare_layer_stacks(
    current: xr.DataArray,
    rebuilt: xr.DataArray,
    *,
    diff_dir: Path,
    source_versions: Mapping[str, Mapping[str, str]] | None = None,
    flag_names: Sequence[str] = (),
    flag_prefixes: Sequence[str] = (),
    align: bool = False,
) -> tuple[list[LayerCompareResult], StackCompareTiming]:
    """Compare two multi-band stacks with one vectorised xarray ``.compute()``."""
    source_versions = source_versions or {}
    diff_dir = Path(diff_dir)
    diff_dir.mkdir(parents=True, exist_ok=True)

    if "band" not in current.dims:
        current = current.expand_dims(band=[str(current.name or "layer")])
    if "band" not in rebuilt.dims:
        rebuilt = rebuilt.expand_dims(band=[str(rebuilt.name or "layer")])

    grid = check_grid_arrays(current, rebuilt)
    aligned = False
    if not grid.ok and align:
        logging.info("Aligning rebuilt stack onto current reference grid (--align)")
        rebuilt = align_to_reference(rebuilt, current)
        grid = check_grid_arrays(current, rebuilt)
        grid.aligned = True
        aligned = True
    elif grid.ok:
        grid.aligned = False

    band_names = [tidy_variable_name(str(n)) for n in current.coords["band"].values]
    reb_names = [tidy_variable_name(str(n)) for n in rebuilt.coords["band"].values]
    if band_names != reb_names:
        # Reorder rebuilt bands to match current when names are a permutation.
        if set(band_names) == set(reb_names):
            rebuilt = rebuilt.sel(band=list(current.coords["band"].values))
            reb_names = band_names
            grid = check_grid_arrays(current, rebuilt)
            grid.aligned = aligned
        else:
            grid.name_match = False

    kinds = [classify_value_kind(n) for n in band_names]
    y_dim, x_dim = _spatial_dims(current)
    height = int(current.sizes[y_dim])
    width = int(current.sizes[x_dim])

    if not grid.ok:
        results = [
            LayerCompareResult(
                layer_name=name,
                value_kind=kind,
                grid_check=grid.summary(),
                grid_ok=False,
                share_within_tolerance=None,
                verdict=LayerVerdict.FAIL,
                notes="grid precondition failed"
                + ("; --align could not reconcile grids" if align else ""),
            )
            for name, kind in zip(band_names, kinds)
        ]
        return results, StackCompareTiming(
            duration_seconds=0.0,
            peak_rss_bytes=process_tree_rss_bytes(),
            n_bands=len(band_names),
            height=height,
            width=width,
            note="grid_failed_before_compute",
        )

    graph = _build_closeness_graph(current, rebuilt, value_kinds=kinds)

    peak = process_tree_rss_bytes()
    started = time.perf_counter()

    def _sample_peak() -> None:
        nonlocal peak
        sample = process_tree_rss_bytes()
        if sample is not None and (peak is None or sample > peak):
            peak = sample

    _sample_peak()
    # One fused compute for metrics + both stacks (feeds difference maps / patches).
    current_np, rebuilt_np, computed = dask.compute(current, rebuilt, graph)
    _sample_peak()
    duration = time.perf_counter() - started
    current_np = xr.DataArray(current_np) if not isinstance(current_np, xr.DataArray) else current_np
    rebuilt_np = xr.DataArray(rebuilt_np) if not isinstance(rebuilt_np, xr.DataArray) else rebuilt_np
    if not isinstance(computed, xr.Dataset):
        raise TypeError("closeness graph compute did not return an xarray Dataset")

    not_close = np.asarray(computed["not_close"].values, dtype=bool)
    results: list[LayerCompareResult] = []

    for band_i, (name, kind) in enumerate(zip(band_names, kinds)):
        versions = source_versions.get(name, {})
        source_current = str(versions.get("current", "") or "")
        source_rebuilt = str(versions.get("rebuilt", "") or "")
        source_changed = bool(
            source_current and source_rebuilt and source_current != source_rebuilt
        )
        method_flagged = matches_flagged_layer(
            name, flag_names=flag_names, flag_prefixes=flag_prefixes
        )

        share = float(computed["share_within_tolerance"].isel(band=band_i).values)
        n_nodata = int(computed["n_nodata_mismatch"].isel(band=band_i).values)

        def _opt_float(key: str) -> Optional[float]:
            val = computed[key].isel(band=band_i).values
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            try:
                out = float(val)
            except (TypeError, ValueError):
                return None
            return None if np.isnan(out) else out

        norm_mean = _opt_float("norm_abs_diff_mean") if kind is LayerValueKind.CONTINUOUS else None
        norm_median = (
            _opt_float("norm_abs_diff_median") if kind is LayerValueKind.CONTINUOUS else None
        )
        # quantile adds a quantile coord — squeeze if present
        p95_da = computed["norm_abs_diff_p95"]
        if "quantile" in p95_da.dims:
            p95_da = p95_da.squeeze("quantile", drop=True)
        p95_val = p95_da.isel(band=band_i).values
        try:
            norm_p95 = float(p95_val)
            if np.isnan(norm_p95):
                norm_p95 = None
        except (TypeError, ValueError):
            norm_p95 = None
        if kind is not LayerValueKind.CONTINUOUS:
            norm_p95 = None
        norm_max = _opt_float("norm_abs_diff_max") if kind is LayerValueKind.CONTINUOUS else None
        correlation = _opt_float("correlation")
        max_diff = _opt_float("max_difference")

        band_not_close = not_close[band_i]
        local, patches, n_components = analyze_mismatch_patches(band_not_close)

        out_file = diff_dir / f"{name}_difference.tif"
        _write_difference_map_from_arrays(
            current_np.isel(band=band_i),
            rebuilt_np.isel(band=band_i),
            out_file,
        )
        patch_file = diff_dir / f"{name}_patches.csv"
        _write_patch_table(patch_file, patches)

        notes_parts: list[str] = []
        if method_flagged:
            notes_parts.append("intended method change (flag list)")
        if source_changed:
            notes_parts.append("source versions differ")
        if aligned:
            notes_parts.append("rebuilt aligned to current grid")

        verdict = decide_verdict(
            grid_ok=True,
            share_within=share,
            differences_local=local,
            source_changed=source_changed,
            method_flagged=method_flagged,
            both_present=True,
        )

        results.append(
            LayerCompareResult(
                layer_name=name,
                value_kind=kind,
                grid_check=grid.summary(),
                grid_ok=True,
                share_within_tolerance=share,
                n_nodata_mismatch=n_nodata,
                norm_abs_diff_mean=norm_mean,
                norm_abs_diff_median=norm_median,
                norm_abs_diff_p95=norm_p95,
                norm_abs_diff_max=norm_max,
                correlation=correlation,
                max_difference=max_diff,
                difference_map_path=str(out_file),
                differences_local=local,
                n_mismatch_components=n_components,
                patches=patches,
                patch_table_path=str(patch_file),
                source_version_current=source_current,
                source_version_rebuilt=source_rebuilt,
                verdict=verdict,
                notes="; ".join(notes_parts),
            )
        )

    timing = StackCompareTiming(
        duration_seconds=round(duration, 4),
        peak_rss_bytes=peak,
        n_bands=len(band_names),
        height=height,
        width=width,
        note="vectorised xarray/dask closeness graph + single compute",
    )
    return results, timing


def compare_layer_pair(
    layer_name: str,
    current_ref: Optional[LayerRef],
    rebuilt_ref: Optional[LayerRef],
    *,
    diff_dir: Path,
    source_version_current: str = "",
    source_version_rebuilt: str = "",
    reason: str = "",
    flag_names: Sequence[str] = (),
    flag_prefixes: Sequence[str] = (),
    align: bool = False,
) -> LayerCompareResult:
    """Compare one layer pair via the vectorised stack path (single-band)."""
    kind = classify_value_kind(layer_name)
    if current_ref is None or rebuilt_ref is None:
        missing_side = []
        if current_ref is None:
            missing_side.append("current")
        if rebuilt_ref is None:
            missing_side.append("rebuilt")
        return LayerCompareResult(
            layer_name=layer_name,
            value_kind=kind,
            grid_check="missing:" + ",".join(missing_side),
            grid_ok=False,
            share_within_tolerance=None,
            source_version_current=source_version_current,
            source_version_rebuilt=source_version_rebuilt,
            reason=reason,
            verdict=LayerVerdict.MISSING,
            notes=f"Layer absent from: {', '.join(missing_side)}",
        )

    current = open_layer_stack([current_ref])
    rebuilt = open_layer_stack([rebuilt_ref])
    versions = {
        layer_name: {
            "current": source_version_current,
            "rebuilt": source_version_rebuilt,
        }
    }
    results, _timing = compare_layer_stacks(
        current,
        rebuilt,
        diff_dir=diff_dir,
        source_versions=versions,
        flag_names=flag_names,
        flag_prefixes=flag_prefixes,
        align=align,
    )
    result = results[0]
    if reason:
        result.reason = reason
    return result


def _write_patch_table(path: Path, patches: Sequence[PatchInfo]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "component_id",
        "cell_count",
        "row_min",
        "row_max",
        "col_min",
        "col_max",
        "bbox_fill_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(patch) for patch in patches)


def model_compare_hooks(model_ids: Sequence[str] | None = None) -> list[ModelCompareHookResult]:
    """Document later model checks without running a retrain."""
    ids = list(model_ids) if model_ids else ["<release-model-id>"]
    return [ModelCompareHookResult(model_id=mid) for mid in ids]


def result_to_row(result: LayerCompareResult) -> dict:
    row = asdict(result)
    row.pop("patches", None)
    row["value_kind"] = result.value_kind.value
    row["verdict"] = result.verdict.value
    return row
