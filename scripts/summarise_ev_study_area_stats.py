#!/usr/bin/env python3
"""Summarise study-area aggregates for a multi-band environmental stack.

Writes only study-area aggregate CSVs (no coordinates, no presence points,
no training parquet). Intended for regenerating the tables under
``docs/data_quality/`` when the rebuild-5 (or later) stack path is available.

Usage
-----
    uv run python scripts/summarise_ev_study_area_stats.py \\
        --stack /path/to/evs-to-model.tif \\
        --output-dir docs/data_quality
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PERCENTILES = [0, 5, 25, 50, 75, 95, 100]
PERCENTILE_NAMES = ["min", "p5", "p25", "median", "p75", "p95", "max"]
LANDCOVER_RE = re.compile(r"^ceh_landcover_(.*?)(?:_(\d+)m)?$")


def study_area_stats(stack_path: Path) -> pd.DataFrame:
    """Per-band min/percentiles/mean/sd and missing share over finite cells."""
    with rasterio.open(stack_path) as ds:
        names = list(ds.descriptions)
        arr = ds.read()

    finite = np.isfinite(arr)
    area = finite.any(axis=0)
    rows: list[dict] = []
    for i, name in enumerate(names):
        values = arr[i][area]
        ok = np.isfinite(values)
        valid = values[ok]
        row: dict = {
            "band": i + 1,
            "variable": name,
            "n_valid": int(ok.sum()),
            "pct_missing": round(100.0 * (1.0 - float(ok.mean())), 3),
        }
        row.update(dict(zip(PERCENTILE_NAMES, np.percentile(valid, PERCENTILES))))
        row["mean"] = float(valid.mean())
        row["sd"] = float(valid.std())
        rows.append(row)
    return pd.DataFrame(rows)


def landcover_shares(stack_path: Path) -> pd.DataFrame:
    """Mean/median cover and non-zero shares for CEH land-cover bands."""
    with rasterio.open(stack_path) as ds:
        names = list(ds.descriptions)
        arr = ds.read()

    finite = np.isfinite(arr)
    area = finite.any(axis=0)
    rows: list[dict] = []
    for name in names:
        match = LANDCOVER_RE.match(name or "")
        if match is None:
            continue
        values = arr[names.index(name)][area]
        values = values[np.isfinite(values)]
        scale = f"{match.group(2)}m" if match.group(2) else "cell_100m"
        mean_cover = float(values.mean())
        rows.append(
            {
                "variable": name,
                "landcover_class": match.group(1),
                "scale": scale,
                "mean_cover_pct": mean_cover / 100.0,
                "mean_cover": mean_cover,
                "median_cover": float(np.median(values)),
                "pct_cells_cover_gt0": 100.0 * float((values > 0).mean()),
                "pct_cells_cover_ge0_5": 100.0 * float((values >= 0.5).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write study-area aggregate CSVs for an environmental GeoTIFF stack. "
            "Does not read training data or write presence statistics."
        )
    )
    parser.add_argument(
        "--stack",
        type=Path,
        required=True,
        help="Path to multi-band evs-to-model.tif (or equivalent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for study_area_stats.csv and landcover_shares.csv.",
    )
    args = parser.parse_args()

    if not args.stack.is_file():
        raise SystemExit(f"Stack not found: {args.stack}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = study_area_stats(args.stack)
    stats.to_csv(
        args.output_dir / "study_area_stats.csv",
        index=False,
        float_format="%.6g",
    )
    shares = landcover_shares(args.stack)
    shares.to_csv(
        args.output_dir / "landcover_shares.csv",
        index=False,
        float_format="%.4g",
    )
    cell = shares[shares["scale"] == "cell_100m"]
    print(
        f"Wrote {len(stats)} band rows and {len(shares)} land-cover rows to "
        f"{args.output_dir}"
    )
    if not cell.empty:
        print(
            "Cell-scale land-cover mean_cover sum:",
            float(cell["mean_cover"].sum()),
        )


if __name__ == "__main__":
    main()
