"""Compute suitability thresholds from held-out CV validation scores."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from sdm.models.core.threshold import (
    P5_MEDIAN_TRANSFER_RULE,
    THRESHOLD_RULES,
    plot_score_spreads,
    summarise_thresholds_for_scores,
    write_run_thresholds,
)
from sdm.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

VALIDATION_SCORES_FILENAME = "validation_scores.parquet"
SUMMARY_CSV = "threshold_summary.csv"
SUMMARY_JSON = "threshold_summary.json"
TRAINING_DATA_FILENAME = "training_data.parquet"
MODEL_PICKLE_FILENAME = "model.pkl"


def _load_package(package_path: Path) -> dict:
    with open(package_path, encoding="utf-8") as handle:
        return json.load(handle)


def discover_package_dirs(models_dir: Path) -> List[Path]:
    """Return package directories that contain validation scores."""
    return sorted(
        path.parent for path in models_dir.glob(f"*/{VALIDATION_SCORES_FILENAME}")
    )


def resolve_rule_params(
    *,
    presence_percentile: float = 10.0,
    presence_capture: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Global rule parameters applied identically to every model."""
    return {
        "presence_percentile": {"percentile": presence_percentile},
        "presence_capture": {"capture": presence_capture},
        "max_sens_spec": {},
    }


def compute_thresholds_for_package(
    package_dir: Path,
    *,
    rules: Sequence[str],
    rule_params: dict[str, dict[str, float]],
    plot: bool = False,
    plot_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute threshold summary rows for one model package."""
    package_path = package_dir / "package.json"
    scores_path = package_dir / VALIDATION_SCORES_FILENAME
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {VALIDATION_SCORES_FILENAME} in {package_dir}")

    validation_scores = pd.read_parquet(scores_path)
    summary = summarise_thresholds_for_scores(
        validation_scores,
        rules=rules,
        rule_params=rule_params,
    )

    model_id = package_dir.name
    latin_name = None
    activity_type = None
    if package_path.exists():
        package = _load_package(package_path)
        model_id = package.get("model_id") or model_id
        latin_name = package.get("latin_name")
        activity_type = package.get("activity_type")

    summary.insert(0, "model_id", model_id)
    summary.insert(1, "latin_name", latin_name)
    summary.insert(2, "activity_type", activity_type)
    summary.insert(3, "package_dir", str(package_dir))

    if plot:
        target_dir = plot_dir or package_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        plot_path = target_dir / f"{model_id}_score_spreads.png"
        plot_score_spreads(
            validation_scores,
            plot_path,
            title=f"{model_id}: presence vs background scores",
        )
        logger.info("Wrote score spread plot to %s", plot_path)

    return summary


def compute_sdm_thresholds(
    models_dir: Path,
    output_dir: Path,
    *,
    rules: Optional[Sequence[str]] = None,
    presence_percentile: float = 10.0,
    presence_capture: float = 0.95,
    model_ids: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    activity_types: Optional[Sequence[str]] = None,
    plot: bool = False,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """Apply threshold rules to one or all model packages; write a tidy table.

    Does not modify ``package.json`` or any published threshold artefacts.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    selected_rules = list(rules) if rules is not None else list(THRESHOLD_RULES)
    unknown = [name for name in selected_rules if name not in THRESHOLD_RULES]
    if unknown:
        raise KeyError(
            f"unknown threshold rule(s): {unknown}; "
            f"known: {sorted(THRESHOLD_RULES)}"
        )

    rule_params = resolve_rule_params(
        presence_percentile=presence_percentile,
        presence_capture=presence_capture,
    )

    package_dirs = discover_package_dirs(models_dir)
    if model_ids is not None:
        wanted = {mid.lower() for mid in model_ids}
        package_dirs = [d for d in package_dirs if d.name.lower() in wanted]

    if not package_dirs:
        logger.warning("No validation_scores.parquet files found under %s", models_dir)
        return None

    frames: list[pd.DataFrame] = []
    for package_dir in package_dirs:
        package_path = package_dir / "package.json"
        latin_name = None
        activity_type = None
        if package_path.exists():
            package = _load_package(package_path)
            latin_name = package.get("latin_name")
            activity_type = package.get("activity_type")
            if species is not None and latin_name not in species:
                continue
            if activity_types is not None and activity_type not in activity_types:
                continue

        logger.info("Computing thresholds for %s", package_dir.name)
        frames.append(
            compute_thresholds_for_package(
                package_dir,
                rules=selected_rules,
                rule_params=rule_params,
                plot=plot,
                plot_dir=output_dir / "plots",
            )
        )

    if not frames:
        logger.warning("No model packages matched the requested filters")
        return None

    result = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / SUMMARY_CSV
    json_path = output_dir / SUMMARY_JSON
    result.to_csv(csv_path, index=False)
    result.to_json(json_path, orient="records", indent=2)
    logger.info("Wrote threshold summary to %s and %s", csv_path, json_path)
    return result


def _load_feature_names(package_dir: Path) -> List[str]:
    package_path = package_dir / "package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"Missing package.json in {package_dir}")
    package = _load_package(package_path)
    features = package.get("feature_names")
    if not features:
        raise ValueError(f"package.json in {package_dir} has no feature_names")
    return list(features)


def _score_final_presence_from_package(
    package_dir: Path,
    training_frame: pd.DataFrame,
) -> np.ndarray:
    """Score presence rows for one package using ``model.pkl`` and run training data."""
    package = _load_package(package_dir / "package.json")
    identifier = package.get("identifier")
    if not identifier:
        raise ValueError(f"package.json in {package_dir} has no identifier")
    model_path = package_dir / MODEL_PICKLE_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {MODEL_PICKLE_FILENAME} in {package_dir}")

    subset = training_frame[training_frame["identifier"] == identifier]
    presence = subset[subset["class"] == 1]
    if len(presence) == 0:
        raise ValueError(f"No presence rows in training data for {identifier}")

    features = _load_feature_names(package_dir)
    missing = [name for name in features if name not in presence.columns]
    if missing:
        raise ValueError(
            f"Training data for {identifier} missing feature columns: {missing}"
        )

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return np.asarray(model.predict_proba(presence[features])[:, 1], dtype=float)


def publish_run_thresholds(
    models_dir: Path,
    *,
    model_ids: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Write METHOD.md ``p5_median_transfer`` cut-offs into a training run folder.

    Requires ``validation_scores.parquet`` and ``model.pkl`` per package, plus a
    run-level ``training_data.parquet`` so final-model presence scores can be
    transferred. Writes ``thresholds.json`` / ``thresholds.csv`` next to the
    packages (and a per-package ``thresholds.json``). Does not modify
    ``package.json``.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    models_dir = Path(models_dir)
    training_path = models_dir / TRAINING_DATA_FILENAME
    if not training_path.exists():
        raise FileNotFoundError(
            f"Publish into-run thresholds requires {TRAINING_DATA_FILENAME} under "
            f"{models_dir}"
        )
    training_frame = pd.read_parquet(training_path)
    if "identifier" not in training_frame.columns or "class" not in training_frame.columns:
        raise ValueError(
            f"{TRAINING_DATA_FILENAME} must include identifier and class columns"
        )

    package_dirs = discover_package_dirs(models_dir)
    if model_ids is not None:
        wanted = {mid.lower() for mid in model_ids}
        package_dirs = [d for d in package_dirs if d.name.lower() in wanted]

    if not package_dirs:
        logger.warning("No validation_scores.parquet files found under %s", models_dir)
        return None

    final_by_model: Dict[str, np.ndarray] = {}
    for package_dir in package_dirs:
        model_path = package_dir / MODEL_PICKLE_FILENAME
        if not model_path.exists():
            logger.warning("Skipping %s (no %s)", package_dir.name, MODEL_PICKLE_FILENAME)
            continue
        logger.info("Scoring final presence for %s", package_dir.name)
        final_by_model[package_dir.name] = _score_final_presence_from_package(
            package_dir,
            training_frame,
        )

    if not final_by_model:
        logger.warning("No packages had model.pkl + training presence to score")
        return None

    rows = write_run_thresholds(
        models_dir,
        final_presence_by_model=final_by_model,
        model_ids=list(final_by_model),
    )
    logger.info(
        "Published %s thresholds for %d model(s) under %s",
        P5_MEDIAN_TRANSFER_RULE,
        len(rows),
        models_dir,
    )
    return rows
