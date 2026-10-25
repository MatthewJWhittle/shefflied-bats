#!/usr/bin/env python3
"""Partial-dependence / response curves for a DeepMaxent run (real data; not CI).

## Implementation notes

Loads one packaged DeepMaxent head from a run folder and writes PD CSV files
per feature (mean predicted suitability while sweeping that feature). Requires
``uv sync --extra dl``. Does not publish maps.

Example::

    uv run python scripts/deepmaxent_partial_dependence.py \\
        --run data/sdm_models/runs/<deepmaxent-run-id> \\
        --model-id myotis_brandtii_roost \\
        --out-dir deepmaxent_pd/myotis_brandtii_roost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdm.utils.io import load_pickled_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=51)
    args = parser.parse_args()

    pkg_dir = args.run / args.model_id
    package = json.loads((pkg_dir / "package.json").read_text())
    feature_names = list(package["feature_names"])
    model = load_pickled_model(pkg_dir / "model.pkl")

    # Background for PD: mean feature vector from training_data if present.
    train_path = args.run / "training_data.parquet"
    if train_path.exists():
        train = pd.read_parquet(train_path)
        subset = train[train["identifier"] == package["identifier"]]
        if subset.empty:
            subset = train
        center = subset[feature_names].mean(numeric_only=True)
        spans = subset[feature_names]
    else:
        center = pd.Series(0.0, index=feature_names)
        spans = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for feat in feature_names:
        base = np.tile(center.to_numpy(dtype=float), (args.grid_size, 1))
        frame = pd.DataFrame(base, columns=feature_names)
        if spans is not None and feat in spans.columns:
            lo = float(spans[feat].quantile(0.05))
            hi = float(spans[feat].quantile(0.95))
        else:
            lo, hi = -2.0, 2.0
        grid = np.linspace(lo, hi, args.grid_size)
        frame[feat] = grid
        pred = model.predict_proba(frame)[:, 1]
        out = pd.DataFrame({feat: grid, "suitability": pred})
        out.to_csv(args.out_dir / f"pd__{feat}.csv", index=False)
    print(f"Wrote PD curves for {len(feature_names)} features under {args.out_dir}")


if __name__ == "__main__":
    main()
