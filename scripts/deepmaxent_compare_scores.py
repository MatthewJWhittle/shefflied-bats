#!/usr/bin/env python3
"""Compare DeepMaxent vs MaxEnt held-out scores (real runs; not for CI).

## Implementation notes

Reads two completed run folders (MaxEnt baseline and DeepMaxent) and writes a
CSV of per-model mean CV AUC side by side. Flags the five models with the
fewest presence counts in the DeepMaxent run.

Example (after training both; do not run publish from this script)::

    uv sync --extra dl
    uv run python scripts/deepmaxent_compare_scores.py \\
        --maxent-run data/sdm_models/runs/<maxent-run-id> \\
        --deepmaxent-run data/sdm_models/runs/<deepmaxent-run-id> \\
        --out deepmaxent_vs_maxent_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_results(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "model_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model_results.csv under {run_dir}")
    df = pd.read_csv(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxent-run", type=Path, required=True)
    parser.add_argument("--deepmaxent-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("deepmaxent_vs_maxent_scores.csv"))
    parser.add_argument("--rarest", type=int, default=5)
    args = parser.parse_args()

    mx = _load_results(args.maxent_run).rename(
        columns={
            "mean_cv_score": "maxent_mean_cv_auc",
            "std_cv_score": "maxent_std_cv_auc",
            "n_presence": "n_presence",
        }
    )
    dm = _load_results(args.deepmaxent_run).rename(
        columns={
            "mean_cv_score": "deepmaxent_mean_cv_auc",
            "std_cv_score": "deepmaxent_std_cv_auc",
        }
    )
    keep_dm = [
        "identifier",
        "latin_name",
        "activity_type",
        "deepmaxent_mean_cv_auc",
        "deepmaxent_std_cv_auc",
    ]
    merged = mx.merge(dm[keep_dm], on=["identifier", "latin_name", "activity_type"], how="outer")
    merged["auc_diff_dm_minus_maxent"] = (
        merged["deepmaxent_mean_cv_auc"] - merged["maxent_mean_cv_auc"]
    )
    merged = merged.sort_values("n_presence", ascending=True)
    merged["rarest_flag"] = False
    rarest_n = min(int(args.rarest), len(merged))
    if rarest_n:
        merged.iloc[:rarest_n, merged.columns.get_loc("rarest_flag")] = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} models; rarest_flag on {rarest_n})")
    print(merged.loc[merged["rarest_flag"], ["identifier", "n_presence", "auc_diff_dm_minus_maxent"]])


if __name__ == "__main__":
    main()
