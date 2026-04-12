#!/usr/bin/env python3
"""Build HSM Visualiser ``ModelMetadata`` JSON from a training ``package.json``.

Usage:
  python scripts/build_model_metadata_from_package.py path/to/package.json

Prints a single-line JSON suitable for ``curl --form-string metadata=...``.
``analysis.feature_band_names`` is taken from ``feature_names`` in the package.
``card`` is filled from ``metrics`` and species/activity; ``extras`` holds string
blobs for ``maxent_config`` and full ``training_package`` for traceability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_json",
        type=Path,
        help="Path to package.json next to model.pkl",
    )
    args = parser.parse_args()
    pkg = json.loads(args.package_json.read_text(encoding="utf-8"))

    latin = pkg.get("latin_name", "")
    activity = pkg.get("activity_type", "")
    metrics = pkg.get("metrics") or {}
    mean_auc = metrics.get("mean_cv_auc")
    std_auc = metrics.get("std_cv_auc")

    meta: dict = {
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

    json.dump(meta, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
