#!/usr/bin/env python3
"""Partial-dependence / response curves for a DeepMaxent run (real data; not CI).

## Implementation notes

Thin CLI over :mod:`sdm.models.deepmaxent.partial_dependence`. Prefer letting
``sdm train`` write PD CSV+PNG under each model package automatically. This
script re-runs PD for one model when needed. Requires ``uv sync --extra dl``.
Does not publish maps; outputs stay under the chosen ``--out-dir`` (never commit
licensed site data).

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

import pandas as pd

from sdm.models.deepmaxent.partial_dependence import write_model_partial_dependence
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

    train_path = args.run / "training_data.parquet"
    if train_path.exists():
        train = pd.read_parquet(train_path)
        subset = train[train["identifier"] == package["identifier"]]
        if subset.empty:
            subset = train
        background = subset[feature_names]
    else:
        background = pd.DataFrame(
            0.0, index=range(8), columns=feature_names
        )

    paths = write_model_partial_dependence(
        model,
        background,
        feature_names,
        args.out_dir,
        grid_resolution=args.grid_size,
    )
    n_feat = len(feature_names)
    print(
        f"Wrote PD curves for {n_feat} features "
        f"({len(paths)} files) under {args.out_dir}"
    )


if __name__ == "__main__":
    main()
