"""Compare a rebuilt environmental layer folder against the current set.

Writes one CSV row per layer (shape check, share within tolerance, max
difference, difference-map path, patch table path, reason, verdict). Layer
order: union of variables across release-model configs (repeatable
``--priority-config``), then any other layers discovered in either folder.

Comparison artifact writes reuse the EV output guards from #40/#44 so results
never land under ``data/evs`` or ``inputs/``.

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
from sdm.utils.ev_output_guards import (
    ForbiddenEvOutputError,
    validate_ev_output_dir,
)
from sdm.utils.io import load_variables_config
from sdm.utils.logging_utils import setup_logging
from sdm.utils.text_utils import tidy_variable_name

DEFAULT_SDM_CONFIG_DIR = Path("data/sdm_config")
DEFAULT_OUTPUT_CSV = Path("layer_comparison/layer_comparison.csv")
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
    "patch_table_path",
    "source_version_current",
    "source_version_rebuilt",
    "reason",
    "verdict",
    "notes",
]


def default_priority_config_paths(
    config_dir: Path = DEFAULT_SDM_CONFIG_DIR,
) -> list[Path]:
    """All ``variables_config.yml`` files under the release-model config tree."""
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        return []
    return sorted(config_dir.glob("*/variables_config.yml"))


def load_priority_layer_names(variables_config_paths: Sequence[Path]) -> list[str]:
    """Union of variables across configs, preserving first-seen order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for path in variables_config_paths:
        cfg = load_variables_config(path)
        for name in cfg.variables:
            tidied = tidy_variable_name(name)
            if tidied not in seen:
                ordered.append(tidied)
                seen.add(tidied)
    return ordered


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


def validate_comparison_write_path(path: Path, *, label: str) -> Path:
    """Refuse comparison artifacts under live EV / inputs locations (#44 guard)."""
    path = Path(path)
    # Directory to protect: parent of a file path, or the path itself if dir-like.
    target_dir = path if path.suffix == "" else path.parent
    if str(target_dir) in ("", "."):
        target_dir = Path(".")
    try:
        validate_ev_output_dir(target_dir)
    except ForbiddenEvOutputError as exc:
        raise ForbiddenEvOutputError(
            f"Refusing to write {label} under a protected path ({path}): {exc}"
        ) from exc
    return path


def compare_ev_layers(
    rebuilt_dir: Path,
    current_dir: Path,
    output_csv: Path,
    *,
    priority_config_paths: Optional[Sequence[Path]] = None,
    difference_maps_dir: Optional[Path] = None,
    source_versions_path: Optional[Path] = None,
    model_ids: Optional[Sequence[str]] = None,
    write_model_hooks_csv: Optional[Path] = None,
    stack_only: bool = False,
    file_glob: Optional[str] = None,
    verbose: bool = False,
) -> CompareRunSummary:
    """Compare all layers found under rebuilt_dir vs current_dir."""
    setup_logging(verbose=verbose)
    rebuilt_dir = Path(rebuilt_dir)
    current_dir = Path(current_dir)
    output_csv = validate_comparison_write_path(Path(output_csv), label="comparison CSV")

    if difference_maps_dir is None:
        difference_maps_dir = output_csv.parent / "difference_maps"
    difference_maps_dir = validate_comparison_write_path(
        Path(difference_maps_dir), label="difference maps"
    )
    if write_model_hooks_csv is not None:
        write_model_hooks_csv = validate_comparison_write_path(
            Path(write_model_hooks_csv), label="model hooks CSV"
        )

    difference_maps_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if priority_config_paths is None:
        priority_config_paths = default_priority_config_paths()
    priority_config_paths = [Path(p) for p in priority_config_paths]
    if not priority_config_paths:
        logging.warning(
            "No priority variables configs found; ordering will use discovered names only"
        )
    priority = load_priority_layer_names(priority_config_paths)

    current_layers = discover_layers(
        current_dir, stack_only=stack_only, file_glob=file_glob
    )
    rebuilt_layers = discover_layers(
        rebuilt_dir, stack_only=stack_only, file_glob=file_glob
    )
    all_names = ordered_layer_names(
        priority,
        set(current_layers) | set(rebuilt_layers),
    )
    source_versions = load_source_versions(source_versions_path)

    logging.info(
        "Comparing %s layers (priority union %s from %s configs; "
        "discovered current=%s rebuilt=%s; stack_only=%s file_glob=%s)",
        len(all_names),
        len(priority),
        len(priority_config_paths),
        len(current_layers),
        len(rebuilt_layers),
        stack_only,
        file_glob,
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
            "%s → %s (grid=%s share=%s local=%s)",
            name,
            result.verdict.value,
            result.grid_check,
            result.share_within_tolerance,
            result.differences_local,
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
