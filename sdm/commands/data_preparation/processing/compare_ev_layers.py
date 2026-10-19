"""Compare a rebuilt environmental layer folder against the current set.

Writes one CSV row per layer (shape check, share within tolerance, max
difference, difference-map path, reason, verdict). Layer order: names from
the Plecotus auritus roost variables config first, then any other layers
discovered in either folder.

Model AUC / cut-off hooks are written as stub rows only — no retrain.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, Sequence

import yaml

from sdm.raster.layer_compare import (
    CompareRunSummary,
    compare_layer_pair,
    discover_layers,
    model_compare_hooks,
    ordered_layer_names,
    result_to_row,
)
from sdm.utils.io import load_variables_config
from sdm.utils.logging_utils import setup_logging
from sdm.utils.text_utils import tidy_variable_name

DEFAULT_PRIORITY_VARIABLES = Path(
    "data/sdm_config/Plecotus_auritus_Roost/variables_config.yml"
)
LAYER_CSV_COLUMNS = [
    "layer_name",
    "value_kind",
    "grid_check",
    "grid_ok",
    "share_within_tolerance",
    "max_difference",
    "difference_map_path",
    "differences_local",
    "n_mismatch_components",
    "source_version_current",
    "source_version_rebuilt",
    "reason",
    "verdict",
    "notes",
]


def load_priority_layer_names(variables_config_path: Path) -> list[str]:
    cfg = load_variables_config(variables_config_path)
    return list(cfg.variables)


def load_source_versions(path: Optional[Path]) -> dict[str, dict[str, str]]:
    """Optional YAML: layer_name → {current: …, rebuilt: …}."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source versions file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Source versions YAML must be a mapping of layer name → versions")
    out: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        name = tidy_variable_name(str(key))
        if isinstance(value, dict):
            out[name] = {
                "current": str(value.get("current", "") or ""),
                "rebuilt": str(value.get("rebuilt", "") or ""),
            }
        else:
            raise ValueError(
                f"Source versions entry for {key!r} must be a mapping with "
                "'current' and/or 'rebuilt' keys"
            )
    return out


def compare_ev_layers(
    rebuilt_dir: Path,
    current_dir: Path,
    output_csv: Path,
    *,
    variables_config_path: Path = DEFAULT_PRIORITY_VARIABLES,
    difference_maps_dir: Optional[Path] = None,
    source_versions_path: Optional[Path] = None,
    model_ids: Optional[Sequence[str]] = None,
    write_model_hooks_csv: Optional[Path] = None,
    verbose: bool = False,
) -> CompareRunSummary:
    """Compare all layers found under rebuilt_dir vs current_dir."""
    setup_logging(verbose=verbose)
    rebuilt_dir = Path(rebuilt_dir)
    current_dir = Path(current_dir)
    output_csv = Path(output_csv)

    if difference_maps_dir is None:
        difference_maps_dir = output_csv.parent / "difference_maps"
    difference_maps_dir = Path(difference_maps_dir)
    difference_maps_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    priority = load_priority_layer_names(variables_config_path)
    current_layers = discover_layers(current_dir)
    rebuilt_layers = discover_layers(rebuilt_dir)
    all_names = ordered_layer_names(
        priority,
        set(current_layers) | set(rebuilt_layers),
    )
    source_versions = load_source_versions(source_versions_path)

    logging.info(
        "Comparing %s layers (priority list %s, discovered current=%s rebuilt=%s)",
        len(all_names),
        len(priority),
        len(current_layers),
        len(rebuilt_layers),
    )

    summary = CompareRunSummary()
    for name in all_names:
        versions = source_versions.get(name, {})
        result = compare_layer_pair(
            name,
            current_layers.get(name),
            rebuilt_layers.get(name),
            diff_dir=difference_maps_dir,
            source_version_current=versions.get("current", ""),
            source_version_rebuilt=versions.get("rebuilt", ""),
        )
        summary.layers.append(result)
        logging.info(
            "%s → %s (grid=%s share=%s)",
            name,
            result.verdict.value,
            result.grid_check,
            result.share_within_tolerance,
        )

    _write_layer_csv(output_csv, summary.layers)

    summary.model_hooks = model_compare_hooks(model_ids)
    if write_model_hooks_csv is not None:
        _write_model_hooks_csv(Path(write_model_hooks_csv), summary.model_hooks)

    return summary


def _write_layer_csv(path: Path, results: Sequence) -> None:
    rows = [result_to_row(r) for r in results]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAYER_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info("Wrote layer comparison table: %s", path)


def _write_model_hooks_csv(path: Path, hooks: Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "auc_current",
        "auc_rebuilt",
        "share_cutoff_switch",
        "verdict",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for hook in hooks:
            writer.writerow(
                {
                    "model_id": hook.model_id,
                    "auc_current": hook.auc_current,
                    "auc_rebuilt": hook.auc_rebuilt,
                    "share_cutoff_switch": hook.share_cutoff_switch,
                    "verdict": hook.verdict,
                    "notes": hook.notes,
                }
            )
    logging.info("Wrote model comparison stub table: %s", path)
