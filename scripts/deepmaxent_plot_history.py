#!/usr/bin/env python3
"""Plot DeepMaxent per-epoch loss and AUC from a run history CSV.

## Implementation notes

Reads ``deepmaxent_training_history.csv`` (aggregated from Lightning CSVLogger
metrics during ``sdm train``) and saves one figure (loss + mean/per-output AUC
vs epoch, panels per fit).

Example::

    uv sync --extra dl
    uv run python scripts/deepmaxent_plot_history.py \\
        --run data/sdm_models/runs/<deepmaxent-run-id> \\
        --out deepmaxent_history.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sdm.models.deepmaxent.history import HISTORY_CSV_NAME, plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run folder containing deepmaxent_training_history.csv",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Explicit history CSV (default: <run>/deepmaxent_training_history.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <run>/deepmaxent_training_history.png)",
    )
    args = parser.parse_args()

    history_path = args.history or (args.run / HISTORY_CSV_NAME)
    if not history_path.exists():
        raise SystemExit(f"History CSV not found: {history_path}")
    out_path = args.out or (args.run / "deepmaxent_training_history.png")
    history = pd.read_csv(history_path)
    path = plot_training_history(history, out_path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
