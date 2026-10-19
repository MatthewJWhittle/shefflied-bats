"""Build ``evs-to-model.tif`` from a merged EV stack (notebook feature-engineering)."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rioxarray as rxr
import xarray as xr

from sdm.raster.processing import calculate_multiscale_variables
from sdm.utils.text_utils import tidy_variable_name

# Defaults match notebooks/feature-engineering.ipynb and the architecture review.
DEFAULT_SELECTED_VARIABLES_CSV = Path("data/ev_clusters_selected.csv")
DEFAULT_MULTISCALE_GROUPS: tuple[str, ...] = (
    "ceh_landcover",
    "os_cover_water",
    "vom_vegetation_height",
)
DEFAULT_MULTISCALE_WINDOWS_CELLS: tuple[int, ...] = (10, 20, 30)
DEFAULT_RESOLUTION_M = 100

# Layer-level clipping: settled ON; limits live in config / rebuild_metrics.
DEFAULT_LAYER_CLIPPING: dict[str, dict[str, float]] = {
    "bgs_coast_distance_to_coast": {"min": 0.0, "max": 7000.0},
    "os_distance_distance_to_buildings": {"min": 0.0, "max": 7000.0},
    "os_distance_distance_to_major_roads": {"min": 0.0, "max": 7000.0},
    "terrain_stats_curvature": {"quantile_min": 0.05, "quantile_max": 0.95},
}

_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("landcover_", "ceh_landcover_"),
    ("coastal_", "bgs_coast_"),
    ("climate_bio_", "climate_bioclim_"),
)

_VOM_STAT_RE = re.compile(
    r"^(?:vom_vegetation_height_)?"
    r"(?:vom_)?"
    r"(?:vegetation_height_)?"
    r"(?:vom_height_)?"
    r"(mean|min|max|std)"
    r"(?:_\d+m)?$"
)
_VOM_NESTED_RE = re.compile(
    r"^vom_vegetation_height_vom_height_(mean|min|max|std)(?:_\d+m)?$"
)
_BIO_NUM_RE = re.compile(r"^climate_bioclim_(?:band_)?(?:bio_?)?(\d+)$")


def default_build_settings() -> dict[str, Any]:
    """Build-settings payload for config mirrors and ``rebuild_metrics.json``."""
    return {
        "selected_variables_csv": str(DEFAULT_SELECTED_VARIABLES_CSV),
        "multiscale": {
            "groups": list(DEFAULT_MULTISCALE_GROUPS),
            "windows_cells": list(DEFAULT_MULTISCALE_WINDOWS_CELLS),
            "resolution_m": DEFAULT_RESOLUTION_M,
        },
        "layer_clipping": {
            "enabled": True,
            "layers": {
                name: dict(spec) for name, spec in DEFAULT_LAYER_CLIPPING.items()
            },
        },
    }


def canonicalize_band_name(name: str) -> str:
    """Map merge / long_name band labels onto the live ``evs-to-model`` names."""
    n = tidy_variable_name(str(name))

    if n == "bgs_coast":
        return "bgs_coast_distance_to_coast"

    for old, new in _PREFIX_ALIASES:
        if n.startswith(old) and not n.startswith(new):
            n = new + n[len(old) :]
            break

    nested = _VOM_NESTED_RE.match(n)
    if nested:
        return f"vom_vegetation_height_{nested.group(1)}"

    # Already-canonical VOM bands (including multiscale suffixes) stay put.
    if n.startswith("vom_vegetation_height_"):
        return n

    vom = _VOM_STAT_RE.match(n)
    if vom and ("height" in n or n.startswith("vom_")):
        return f"vom_vegetation_height_{vom.group(1)}"

    bio = _BIO_NUM_RE.match(n)
    if bio:
        return f"climate_bioclim_bio_{int(bio.group(1))}"

    return n


def rename_ev_bands(dataset: xr.Dataset) -> xr.Dataset:
    """Rename bands via ``long_name`` when present, then canonicalize."""
    rename_from_long: dict[str, str] = {}
    for var in list(dataset.data_vars):
        long_name = dataset[var].attrs.get("long_name")
        if long_name is not None and str(long_name).strip():
            rename_from_long[str(var)] = str(long_name)
    if rename_from_long:
        dataset = dataset.rename(rename_from_long)

    final_map: dict[str, str] = {}
    used: set[str] = set()
    for var in list(dataset.data_vars):
        canonical = canonicalize_band_name(str(var))
        if canonical in used:
            raise ValueError(
                f"Canonical band name collision for {var!r} → {canonical!r}"
            )
        used.add(canonical)
        if canonical != str(var):
            final_map[str(var)] = canonical
    if final_map:
        dataset = dataset.rename(final_map)
    for var in dataset.data_vars:
        dataset[var].attrs["long_name"] = str(var)
    return dataset


def load_selected_variable_names(csv_path: Path | str) -> list[str]:
    """Return ``column_name`` values with ``include == 1`` from the clusters CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Selected variables CSV not found: {path}. "
            "Expected data/ev_clusters_selected.csv from the feature-engineering notebook."
        )
    keep: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "column_name" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must have a column_name column")
        for row in reader:
            flag = str(row.get("include", "")).strip()
            if flag in {"1", "1.0", "true", "True"}:
                keep.append(str(row["column_name"]).strip())
    if not keep:
        raise ValueError(f"No included variables found in {path}")
    return keep


def apply_layer_clipping(
    dataset: xr.Dataset,
    *,
    clipping: Mapping[str, Mapping[str, float]] | None = None,
    enabled: bool = True,
) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    """Apply notebook layer-level clipping (distances and curvature quantiles).

    Clipping is settled ON by the architecture review. ``enabled`` exists so the
    option is explicit in config/metrics; turning it off is out of scope for #38.

    Returns
    -------
    dataset, applied
        ``applied`` maps each clipped layer to the concrete min/max used. For
        quantile clips (curvature), this includes the resolved 5%/95% values so
        ``rebuild_metrics.json`` can record them, not only the log line.
    """
    if not enabled:
        logging.warning(
            "Layer clipping disabled — out of scope for #38; output will not "
            "match today's evs-to-model.tif like-for-like"
        )
        return dataset, {}

    specs = clipping if clipping is not None else DEFAULT_LAYER_CLIPPING
    out = dataset.copy()
    applied: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        if name not in out.data_vars:
            logging.info("Skipping clip for missing layer %s", name)
            continue
        if "quantile_min" in spec or "quantile_max" in spec:
            q_lo = float(spec.get("quantile_min", 0.0))
            q_hi = float(spec.get("quantile_max", 1.0))
            lo = float(out[name].quantile(q_lo).item())
            hi = float(out[name].quantile(q_hi).item())
            out[name] = out[name].clip(min=lo, max=hi)
            applied[name] = {
                "quantile_min": q_lo,
                "quantile_max": q_hi,
                "min": lo,
                "max": hi,
            }
            logging.info(
                "Clipped %s to quantiles %.2f–%.2f (values %.6g–%.6g)",
                name,
                q_lo,
                q_hi,
                lo,
                hi,
            )
        else:
            lo = float(spec["min"])
            hi = float(spec["max"])
            out[name] = out[name].clip(min=lo, max=hi)
            applied[name] = {"min": lo, "max": hi}
            logging.info("Clipped %s to %.0f–%.0f", name, lo, hi)
    return out, applied


def subset_selected_variables(
    dataset: xr.Dataset, selected: Sequence[str]
) -> xr.Dataset:
    missing = [name for name in selected if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            "Merged stack is missing selected variables: "
            + ", ".join(missing)
        )
    return dataset[list(selected)]


def _capture_nodata(dataset: xr.Dataset) -> dict[str, Any]:
    """Snapshot each band's rio nodata (notebook preserves source tags on write)."""
    return {str(var): dataset[var].rio.nodata for var in dataset.data_vars}


def _restore_nodata(dataset: xr.Dataset, nodata_by_var: Mapping[str, Any]) -> xr.Dataset:
    """Re-apply source nodata after ops (clip/astype) that drop encoding.

    Does not invent NaN when every source band lacked a nodata tag. When some
    source bands had a tag, neighbourhood / clip survivors inherit that source
    value so the Dataset write stays consistent.
    """
    out = dataset
    fallback = next(
        (value for value in nodata_by_var.values() if value is not None),
        None,
    )
    for var in out.data_vars:
        name = str(var)
        nodata = nodata_by_var.get(name)
        if nodata is None:
            # Neighbourhood layers inherit the base band's nodata.
            base = re.sub(r"_(?:500|1000|1500)m$", "", name)
            nodata = nodata_by_var.get(base)
        if nodata is None:
            nodata = fallback
        if nodata is not None:
            out[var].rio.write_nodata(nodata, inplace=True)
    return out


def add_multiscale_layers(
    dataset: xr.Dataset,
    *,
    groups: Sequence[str] = DEFAULT_MULTISCALE_GROUPS,
    windows_cells: Sequence[int] = DEFAULT_MULTISCALE_WINDOWS_CELLS,
) -> xr.Dataset:
    """Add 500/1000/1500 m neighbourhood means for configured variable groups."""
    multiscale_vars = [
        var
        for var in dataset.data_vars
        if any(str(var).startswith(group) for group in groups)
    ]
    if not multiscale_vars:
        logging.warning("No multiscale source variables matched groups %s", groups)
        return dataset

    pieces = [
        calculate_multiscale_variables(dataset[multiscale_vars], window)
        for window in windows_cells
    ]
    multiscale_ds = xr.merge(pieces)
    for var in multiscale_ds.data_vars:
        multiscale_ds[var].attrs["long_name"] = str(var)
    return xr.merge([dataset, multiscale_ds])


def build_settings_from_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge optional ``ev_stack`` config with defaults (clipping stays ON)."""
    settings = default_build_settings()
    if not raw:
        return settings

    if "selected_variables_csv" in raw:
        settings["selected_variables_csv"] = str(raw["selected_variables_csv"])

    multi = raw.get("multiscale") or {}
    if "groups" in multi:
        settings["multiscale"]["groups"] = list(multi["groups"])
    if "windows_cells" in multi:
        settings["multiscale"]["windows_cells"] = [int(w) for w in multi["windows_cells"]]
    if "resolution_m" in multi:
        settings["multiscale"]["resolution_m"] = int(multi["resolution_m"])

    clip = raw.get("layer_clipping") or {}
    # Architecture review: default ON; disabling is out of scope for #38.
    settings["layer_clipping"]["enabled"] = bool(clip.get("enabled", True))
    if "layers" in clip and clip["layers"]:
        settings["layer_clipping"]["layers"] = {
            str(name): {str(k): float(v) for k, v in spec.items()}
            for name, spec in clip["layers"].items()
        }
    return settings


def build_model_ev_stack(
    input_path: Path | str,
    output_path: Path | str,
    *,
    build_settings: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Reproduce notebook steps: rename → clip → subset → multiscale → write.

    Order matches ``notebooks/feature-engineering.ipynb`` (clipping runs on the
    full renamed stack before the selected subset and neighbourhood layers).

    Parameters
    ----------
    input_path:
        Merged stack (``all-evs.tif``).
    output_path:
        Destination ``evs-to-model.tif``.
    build_settings:
        Explicit build settings (from config). Defaults reproduce today's stack.

    Returns
    -------
    (output_path, settings_used)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = build_settings_from_config(build_settings)
    input_path = Path(input_path)
    output_path = Path(output_path)

    logging.info("Loading merged EV stack from %s", input_path)
    dataset = rxr.open_rasterio(input_path, masked=True, band_as_variable=True)
    dataset = rename_ev_bands(dataset)
    # Capture before clip/astype — those ops drop rio nodata encoding.
    source_nodata = _capture_nodata(dataset)

    clip_cfg = settings["layer_clipping"]
    dataset, applied_clips = apply_layer_clipping(
        dataset,
        clipping=clip_cfg["layers"],
        enabled=bool(clip_cfg["enabled"]),
    )
    # Persist concrete clip bounds (including curvature 5%/95% values) for metrics.
    settings["layer_clipping"]["applied"] = applied_clips

    selected = load_selected_variable_names(settings["selected_variables_csv"])
    dataset = subset_selected_variables(dataset, selected)

    multi = settings["multiscale"]
    dataset = add_multiscale_layers(
        dataset,
        groups=multi["groups"],
        windows_cells=multi["windows_cells"],
    )

    # float32 + long_name for GeoTIFF band descriptions. Restore source nodata
    # rather than inventing NaN when the source tag was unset (notebook write
    # path: ``rio.to_raster`` only, no forced NaN fill).
    # Note: load uses masked=True (same as the notebook), so a tagged source
    # nodata such as -9999 is already NaN in ``source_nodata`` before write —
    # that is why today's evs-to-model.tif carries NaN nodata.
    for var in dataset.data_vars:
        dataset[var] = dataset[var].astype(np.float32)
        dataset[var].attrs["long_name"] = str(var)
    dataset = _restore_nodata(dataset, source_nodata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing model EV stack (%d bands) to %s", len(dataset.data_vars), output_path)
    dataset.rio.to_raster(output_path)
    return output_path, settings


__all__ = [
    "DEFAULT_LAYER_CLIPPING",
    "DEFAULT_MULTISCALE_GROUPS",
    "DEFAULT_MULTISCALE_WINDOWS_CELLS",
    "DEFAULT_SELECTED_VARIABLES_CSV",
    "add_multiscale_layers",
    "apply_layer_clipping",
    "build_model_ev_stack",
    "build_settings_from_config",
    "canonicalize_band_name",
    "default_build_settings",
    "load_selected_variable_names",
    "rename_ev_bands",
    "subset_selected_variables",
]
