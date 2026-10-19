"""Compare rebuilt environmental layers against a current reference set.

Pass marks follow the consolidated architecture review on issue #38
(comment that replaces earlier tolerance comments): same shape exactly;
≥95% of cells within layer-type tolerance; difference maps for local
patches; source-version changes flagged rather than failed.

Model AUC checks (items 5–7 in that review) are stubbed as hooks only —
they are not required for layer comparison runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import rasterio
from scipy import ndimage

from sdm.utils.text_utils import tidy_variable_name

RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff"}
DEFAULT_STACK_NAME = "evs-to-model.tif"
SHARE_WITHIN_TOLERANCE_PASS = 0.95
CONTINUOUS_RANGE_FRACTION = 0.01

# Compactness heuristics for out-of-tolerance cells.
MAX_COMPONENTS_FOR_COMPACT = 8
# Fill ratio of mismatch cells inside their row/col bounding box. Low values
# flag elongated / road-network-shaped components even when component count is 1.
MIN_BBOX_FILL_RATIO = 0.25

# ---------------------------------------------------------------------------
# Exact-match (category / percentage-cover) name rules.
#
# Open question for architecture review: neighbourhood CEH landcover buffers
# (ceh_landcover_*_{500,1000,1500}m) are stored as decimal averages, not pure
# categories. They remain on EXACT match for now; change
# ``exact_match_layer_names`` / ``is_exact_match_layer`` in one place when the
# review decides otherwise.
# ---------------------------------------------------------------------------
EXACT_MATCH_NAME_CONTAINS: tuple[str, ...] = ("landcover",)
EXACT_MATCH_NAME_PREFIXES: tuple[str, ...] = ("os_cover",)
EXACT_MATCH_NAME_INFIXES: tuple[str, ...] = ("_cover_",)


class LayerValueKind(str, Enum):
    """How cell values are compared."""

    EXACT = "exact"  # category / percentage-cover
    CONTINUOUS = "continuous"  # distances, heights, climate


class LayerVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
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
    name_match: bool
    projection_match: bool
    cell_size_match: bool
    origin_match: bool
    extent_match: bool
    nodata_value_match: bool
    nodata_cells_match: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return all(
            (
                self.name_match,
                self.projection_match,
                self.cell_size_match,
                self.origin_match,
                self.extent_match,
                self.nodata_value_match,
                self.nodata_cells_match,
            )
        )

    def summary(self) -> str:
        if self.ok:
            return "ok"
        parts = []
        for label, value in (
            ("name", self.name_match),
            ("projection", self.projection_match),
            ("cell_size", self.cell_size_match),
            ("origin", self.origin_match),
            ("extent", self.extent_match),
            ("nodata_value", self.nodata_value_match),
            ("nodata_cells", self.nodata_cells_match),
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
    max_difference: Optional[float]
    difference_map_path: str
    differences_local: Optional[bool]
    n_mismatch_components: Optional[int]
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


def is_exact_match_layer(layer_name: str) -> bool:
    """Return True when the layer must match cell values exactly.

    Central switch for category / percentage-cover rules. Neighbourhood
    CEH landcover buffers remain exact until architecture review changes this.
    """
    name = tidy_variable_name(layer_name)
    if any(token in name for token in EXACT_MATCH_NAME_CONTAINS):
        return True
    if any(name.startswith(prefix) for prefix in EXACT_MATCH_NAME_PREFIXES):
        return True
    if any(infix in name for infix in EXACT_MATCH_NAME_INFIXES):
        return True
    return False


def classify_value_kind(layer_name: str) -> LayerValueKind:
    """Category / percentage-cover → exact; distances, heights, climate → continuous."""
    if is_exact_match_layer(layer_name):
        return LayerValueKind.EXACT
    return LayerValueKind.CONTINUOUS


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
        # Path.match is relative to the pattern; prefer glob/rglob semantics.
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
    """Index named layers under a folder (single-band files and multi-band stacks)."""
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
                found.setdefault(
                    name, LayerRef(name=name, path=path, band_index=band_index)
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


def _read_band(ref: LayerRef) -> tuple[np.ndarray, dict]:
    with rasterio.open(ref.path) as src:
        data = src.read(ref.band_index)
        profile = src.profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": data.dtype,
                "nodata": src.nodata,
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
            }
        )
        description = None
        if src.descriptions and src.descriptions[ref.band_index - 1]:
            description = src.descriptions[ref.band_index - 1]
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "description": description,
            "name": ref.name,
            "res": src.res,
            "bounds": src.bounds,
        }
    return data, {**profile, "_meta": meta}


def check_grid(
    current_meta: Mapping,
    rebuilt_meta: Mapping,
    *,
    expected_name: str,
) -> GridCheck:
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
        nodata_cells_match=False,
    )


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    mask = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        try:
            mask &= ~np.isclose(data.astype(np.float64), float(nodata), equal_nan=True)
        except (TypeError, ValueError):
            mask &= data != nodata
    mask &= ~np.isnan(data.astype(np.float64, copy=False))
    return mask


def cells_within_tolerance(
    current: np.ndarray,
    rebuilt: np.ndarray,
    *,
    nodata,
    kind: LayerValueKind,
) -> tuple[np.ndarray, float, float]:
    """Return (within_mask, share, max_abs_diff) over jointly valid cells."""
    cur_valid = _valid_mask(current, nodata)
    reb_valid = _valid_mask(rebuilt, nodata)
    jointly_valid = cur_valid & reb_valid
    n_valid = int(jointly_valid.sum())
    if n_valid == 0:
        return jointly_valid, 1.0, 0.0

    cur_v = current.astype(np.float64, copy=False)
    reb_v = rebuilt.astype(np.float64, copy=False)
    abs_diff = np.zeros(current.shape, dtype=np.float64)
    abs_diff[jointly_valid] = np.abs(cur_v[jointly_valid] - reb_v[jointly_valid])
    max_diff = float(abs_diff[jointly_valid].max()) if n_valid else 0.0

    if kind is LayerValueKind.EXACT:
        within = np.zeros(current.shape, dtype=bool)
        within[jointly_valid] = abs_diff[jointly_valid] == 0.0
    elif kind is LayerValueKind.CONTINUOUS:
        cur_range = float(cur_v[jointly_valid].max() - cur_v[jointly_valid].min())
        tol = CONTINUOUS_RANGE_FRACTION * cur_range if cur_range > 0 else 0.0
        within = np.zeros(current.shape, dtype=bool)
        within[jointly_valid] = abs_diff[jointly_valid] <= tol
    else:
        raise TypeError(f"Unhandled layer value kind: {kind!r}")

    share = float(within[jointly_valid].sum() / n_valid)
    return within, share, max_diff


def analyze_mismatch_patches(
    out_of_tolerance: np.ndarray,
    jointly_valid: np.ndarray,
) -> tuple[bool, list[PatchInfo], int]:
    """Return (differences_local, patches, n_components).

    Fails the local-patch check when there are too many components, or when any
    component is sparse inside its bounding box (low fill ratio — typical of
    elongated / road-network-shaped differences).
    """
    mismatch = out_of_tolerance & jointly_valid
    n_mismatch = int(mismatch.sum())
    if n_mismatch == 0:
        return True, [], 0

    labeled, n_components = ndimage.label(mismatch)
    patches: list[PatchInfo] = []
    sparse_component = False

    for component_id in range(1, n_components + 1):
        coords = np.argwhere(labeled == component_id)
        cell_count = int(coords.shape[0])
        row_min, col_min = (int(v) for v in coords.min(axis=0))
        row_max, col_max = (int(v) for v in coords.max(axis=0))
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
        # Single-cell patches are compact by definition.
        if cell_count > 1 and fill_ratio < MIN_BBOX_FILL_RATIO:
            sparse_component = True

    too_many = n_components > MAX_COMPONENTS_FOR_COMPACT
    differences_local = not too_many and not sparse_component
    return differences_local, patches, int(n_components)


def write_difference_map(
    current: np.ndarray,
    rebuilt: np.ndarray,
    *,
    profile: dict,
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


def decide_verdict(
    *,
    grid_ok: bool,
    share_within: Optional[float],
    differences_local: Optional[bool],
    source_changed: bool,
    both_present: bool,
) -> LayerVerdict:
    if not both_present:
        return LayerVerdict.MISSING
    if not grid_ok:
        return LayerVerdict.FAIL
    if share_within is None:
        return LayerVerdict.FAIL
    # Source version drift is always flagged when both versions are recorded,
    # even if values and locality would otherwise pass.
    if source_changed:
        return LayerVerdict.FLAGGED_SOURCE_CHANGE
    values_ok = share_within >= SHARE_WITHIN_TOLERANCE_PASS
    local_ok = differences_local is True or differences_local is None
    if values_ok and local_ok:
        return LayerVerdict.PASS
    return LayerVerdict.FAIL


def compare_layer_pair(
    layer_name: str,
    current_ref: Optional[LayerRef],
    rebuilt_ref: Optional[LayerRef],
    *,
    diff_dir: Path,
    source_version_current: str = "",
    source_version_rebuilt: str = "",
    reason: str = "",
) -> LayerCompareResult:
    kind = classify_value_kind(layer_name)
    source_changed = bool(
        source_version_current
        and source_version_rebuilt
        and source_version_current != source_version_rebuilt
    )

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
            max_difference=None,
            difference_map_path="",
            differences_local=None,
            n_mismatch_components=None,
            source_version_current=source_version_current,
            source_version_rebuilt=source_version_rebuilt,
            reason=reason,
            verdict=LayerVerdict.MISSING,
            notes=f"Layer absent from: {', '.join(missing_side)}",
        )

    current_data, current_pack = _read_band(current_ref)
    rebuilt_data, rebuilt_pack = _read_band(rebuilt_ref)
    current_meta = current_pack.pop("_meta")
    rebuilt_meta = rebuilt_pack.pop("_meta")

    grid = check_grid(current_meta, rebuilt_meta, expected_name=layer_name)
    nodata = current_meta["nodata"]
    if grid.extent_match and current_data.shape == rebuilt_data.shape:
        cur_nodata_mask = ~_valid_mask(current_data, nodata)
        reb_nodata_mask = ~_valid_mask(rebuilt_data, rebuilt_meta["nodata"])
        grid.nodata_cells_match = bool(np.array_equal(cur_nodata_mask, reb_nodata_mask))
    else:
        grid.nodata_cells_match = False

    diff_path = ""
    patch_table_path = ""
    patches: list[PatchInfo] = []
    share: Optional[float] = None
    max_diff: Optional[float] = None
    local: Optional[bool] = None
    n_components: Optional[int] = None
    notes = ""

    if current_data.shape == rebuilt_data.shape:
        within, share, max_diff = cells_within_tolerance(
            current_data,
            rebuilt_data,
            nodata=nodata,
            kind=kind,
        )
        jointly = _valid_mask(current_data, nodata) & _valid_mask(
            rebuilt_data, rebuilt_meta["nodata"]
        )
        out_of_tol = jointly & ~within
        local, patches, n_components = analyze_mismatch_patches(out_of_tol, jointly)
        out_file = Path(diff_dir) / f"{layer_name}_difference.tif"
        write_difference_map(
            current_data,
            rebuilt_data,
            profile=current_pack,
            nodata=nodata,
            out_path=out_file,
        )
        diff_path = str(out_file)
        patch_file = Path(diff_dir) / f"{layer_name}_patches.csv"
        _write_patch_table(patch_file, patches)
        patch_table_path = str(patch_file)

    if source_changed:
        notes = "source versions differ"

    verdict = decide_verdict(
        grid_ok=grid.ok,
        share_within=share,
        differences_local=local,
        source_changed=source_changed,
        both_present=True,
    )

    return LayerCompareResult(
        layer_name=layer_name,
        value_kind=kind,
        grid_check=grid.summary(),
        grid_ok=grid.ok,
        share_within_tolerance=share,
        max_difference=max_diff,
        difference_map_path=diff_path,
        differences_local=local,
        n_mismatch_components=n_components,
        patches=patches,
        patch_table_path=patch_table_path,
        source_version_current=source_version_current,
        source_version_rebuilt=source_version_rebuilt,
        reason=reason,
        verdict=verdict,
        notes=notes,
    )


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
        for patch in patches:
            writer.writerow(
                {
                    "component_id": patch.component_id,
                    "cell_count": patch.cell_count,
                    "row_min": patch.row_min,
                    "row_max": patch.row_max,
                    "col_min": patch.col_min,
                    "col_max": patch.col_max,
                    "bbox_fill_ratio": patch.bbox_fill_ratio,
                }
            )


def model_compare_hooks(model_ids: Sequence[str] | None = None) -> list[ModelCompareHookResult]:
    """Document later model checks without running a retrain."""
    ids = list(model_ids) if model_ids else ["<release-model-id>"]
    return [ModelCompareHookResult(model_id=mid) for mid in ids]


def result_to_row(result: LayerCompareResult) -> dict:
    return {
        "layer_name": result.layer_name,
        "value_kind": result.value_kind.value,
        "grid_check": result.grid_check,
        "grid_ok": result.grid_ok,
        "share_within_tolerance": result.share_within_tolerance,
        "max_difference": result.max_difference,
        "difference_map_path": result.difference_map_path,
        "differences_local": result.differences_local,
        "n_mismatch_components": result.n_mismatch_components,
        "patch_table_path": result.patch_table_path,
        "source_version_current": result.source_version_current,
        "source_version_rebuilt": result.source_version_rebuilt,
        "reason": result.reason,
        "verdict": result.verdict.value,
        "notes": result.notes,
    }
