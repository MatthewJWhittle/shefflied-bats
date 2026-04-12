"""Helpers for HSM Visualiser API payloads from local training packages."""

from __future__ import annotations

import json
from typing import Any, Dict


def model_metadata_from_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """Map training ``package.json`` (parsed dict) to API ``ModelMetadata``."""
    latin = pkg.get("latin_name", "")
    activity = pkg.get("activity_type", "")
    metrics = pkg.get("metrics") or {}
    mean_auc = metrics.get("mean_cv_auc")

    return {
        "analysis": {
            "feature_band_names": list(pkg.get("feature_names") or []),
        },
        "card": {
            "title": f"{latin} — {activity}".strip(" —"),
            "summary": "MaxEnt (Sheffield Bats training package)",
            "version": str(pkg.get("schema_version", 1)),
            "primary_metric_type": "mean_cv_auc",
            "primary_metric_value": "" if mean_auc is None else str(mean_auc),
        },
        "extras": {
            "training_model_id": str(pkg.get("model_id", "")),
            "training_metrics_json": json.dumps(metrics, separators=(",", ":")),
            "training_maxent_config_json": json.dumps(
                pkg.get("maxent_config") or {}, separators=(",", ":")
            ),
            "training_package_json": json.dumps(pkg, separators=(",", ":")),
        },
    }
