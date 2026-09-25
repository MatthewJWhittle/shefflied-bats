"""Compute suitability thresholds from held-out CV validation scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sdm.commands.modelling.utils import get_model_id
from sdm.models.core.threshold import (
    compute_threshold_from_validation_scores,
    enrich_threshold_package_dict,
    threshold_result_to_package_dict,
)
from sdm.utils.io import load_model_config
from sdm.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

VALIDATION_SCORES_FILENAME = "validation_scores.parquet"


def _load_package(package_path: Path) -> dict:
    with open(package_path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_package(package_path: Path, package: dict) -> None:
    with open(package_path, "w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2, allow_nan=False)


def compute_threshold_for_package(
    package_dir: Path,
    *,
    model_config_path: Path,
) -> Optional[float]:
    """Compute and persist threshold metadata for one model package directory."""
    package_path = package_dir / "package.json"
    scores_path = package_dir / VALIDATION_SCORES_FILENAME

    if not package_path.exists():
        logger.warning("Skipping %s: missing package.json", package_dir)
        return None
    if not scores_path.exists():
        logger.warning("Skipping %s: missing %s", package_dir, VALIDATION_SCORES_FILENAME)
        return None

    model_config = load_model_config(model_config_path)
    validation_scores = pd.read_parquet(scores_path)
    result = compute_threshold_from_validation_scores(
        validation_scores,
        model_config.threshold,
    )

    package = _load_package(package_path)
    threshold_dict = enrich_threshold_package_dict(
        threshold_result_to_package_dict(result),
        model_config.threshold,
    )
    package["threshold"] = threshold_dict
    _write_package(package_path, package)

    logger.info(
        "Threshold for %s: %.6f (%s, n_presence=%d, bootstrap [%.6f, %.6f])",
        package.get("identifier", package_dir.name),
        result.value,
        result.rule,
        result.n_presence_records,
        result.bootstrap_low if result.bootstrap_low is not None else float("nan"),
        result.bootstrap_high if result.bootstrap_high is not None else float("nan"),
    )
    return result.value


def compute_sdm_thresholds(
    models_dir: Path,
    model_config_path: Path,
    species: Optional[List[str]] = None,
    activity_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """Apply threshold rules to all model packages under ``models_dir``."""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    logger.info("Computing suitability thresholds from held-out CV scores in %s", models_dir)

    package_dirs = sorted(
        path.parent for path in models_dir.glob(f"*/{VALIDATION_SCORES_FILENAME}")
    )
    if not package_dirs:
        logger.warning("No validation_scores.parquet files found under %s", models_dir)
        return

    for package_dir in package_dirs:
        package_path = package_dir / "package.json"
        if not package_path.exists():
            continue
        package = _load_package(package_path)
        latin_name = package.get("latin_name")
        activity_type = package.get("activity_type")

        if species is not None and latin_name not in species:
            continue
        if activity_types is not None and activity_type not in activity_types:
            continue

        compute_threshold_for_package(
            package_dir,
            model_config_path=model_config_path,
        )

    logger.info("Threshold computation complete")


def resolve_package_dir(
    models_dir: Path,
    latin_name: str,
    activity_type: str,
) -> Path:
    """Resolve a model package directory from species and activity labels."""
    model_id = get_model_id([latin_name, activity_type])
    return models_dir / model_id
