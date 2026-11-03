#!/usr/bin/env python3
"""Per-layer data-quality pack for an environmental GeoTIFF stack.

Study-area aggregates only. Writes ``layer_stats.csv`` (and optionally
log-scale histograms). Does not read training data. Do not commit histogram
or map image outputs — they are for local inspection.

Usage
-----
    uv run python scripts/make_layer_qa.py \\
        --stack /path/to/evs-to-model.tif \\
        --output-dir /path/to/layer_qa_out

    uv run python scripts/make_layer_qa.py \\
        --stack /path/to/evs-to-model.tif \\
        --output-dir /tmp/layer_qa \\
        --write-histograms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

PERCENTILES = [0, 1, 5, 50, 95, 99, 100]
PERCENTILE_NAMES = ["min", "p1", "p5", "median", "p95", "p99", "max"]
DISTANCE_CAP_M = 7000.0


def verdict(name: str, stats: dict, values: np.ndarray) -> tuple[str, str]:
    """Return (verdict, reason) for one band's study-area values."""
    if name == "terrain_dtm" and stats["min"] < 0:
        n_neg = int((values < 0).sum())
        return (
            "flag",
            f"{n_neg} cells below 0 m (min {stats['min']:.1f} m); "
            "check sea/estuary buffer and inland artefacts",
        )
    if name.startswith("vom_vegetation_height_max") and stats["max"] > 40:
        n_tall = int((values > 40).sum())
        return (
            "flag",
            f"{n_tall} cells >40 m (max {stats['max']:.1f} m); "
            "tallest UK trees ~40-50 m, likely structures in the vegetation model",
        )
    if name.startswith("vom_") and stats["max"] > 40:
        return "flag", f"max {stats['max']:.1f} m above plausible tree height"
    if name.startswith("os_cover_water"):
        level = "flag" if stats["mean"] > 5 else "ok"
        pct_full = 100.0 * float((values >= 99.99).mean())
        return (
            level,
            f"mean {stats['mean']:.1f}% cover, {pct_full:.1f}% of cells at 100%; "
            "inland UK open water is typically ~1-3% of land area",
        )
    if name == "bgs_coast_distance_to_coast":
        pct_cap = 100.0 * float((values >= DISTANCE_CAP_M).mean())
        n_info = int((values < DISTANCE_CAP_M).sum())
        return (
            "flag",
            f"{pct_cap:.1f}% of cells at the 7000 m cap; "
            f"layer is near-constant inland, only {n_info} cells carry information",
        )
    if name.startswith("os_distance") and np.isclose(stats["max"], DISTANCE_CAP_M):
        pct = 100.0 * stats["n_at_max"] / len(values)
        return (
            "flag",
            f"capped at 7000 m: {stats['n_at_max']} cells ({pct:.2f}%) sit at the cap",
        )
    if name == "terrain_stats_slope":
        return (
            "ok",
            f"range {stats['min']:.2f}-{stats['max']:.2f}; "
            f"reads as radians (max ~{np.degrees(stats['max']):.0f} deg), "
            "plausible at 100 m; confirm unit in code",
        )
    if name == "terrain_stats_twi":
        level = "check" if stats["max"] > 25 else "ok"
        return (
            level,
            f"range {stats['min']:.1f}-{stats['max']:.1f}; typical 5-20, "
            "a few high values on flats/channels are expected",
        )
    if name.startswith("ceh_landcover"):
        level = "ok" if stats["max"] <= 10000 else "flag"
        return level, "cover in 0-10000 units (percent x100); in range"
    if stats["pct_missing"] > 1:
        return "check", f"{stats['pct_missing']:.2f}% missing inside study area"
    if (
        stats["n_at_max"] > 0.01 * len(values)
        and stats["max"] != stats["min"]
        and not name.startswith(("os_cover", "ceh_"))
    ):
        return "check", f"{stats['n_at_max']} cells at max - possible cap"
    return "ok", "within expected range"


def layer_stats(stack_path: Path) -> pd.DataFrame:
    """Per-band aggregates and verdicts over the study-area mask."""
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
        row["n_at_max"] = int((valid == valid.max()).sum())
        row["verdict"], row["reason"] = verdict(name, row, valid)
        rows.append(row)
    return pd.DataFrame(rows)


def write_histograms(stack_path: Path, hist_dir: Path, stats: pd.DataFrame) -> None:
    """Write one log-scale histogram PNG per band (local use only; do not commit)."""
    hist_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack_path) as ds:
        names = list(ds.descriptions)
        arr = ds.read()
    finite = np.isfinite(arr)
    area = finite.any(axis=0)
    for _, row in stats.iterrows():
        i = int(row["band"]) - 1
        name = names[i]
        values = arr[i][area]
        valid = values[np.isfinite(values)]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(valid, bins=80, color="#4a7")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=8)
        ax.set_ylabel("cells (log)")
        ax.tick_params(labelsize=7)
        for q in ("p1", "p99"):
            ax.axvline(row[q], ls=":", c="k", lw=0.7)
        fig.tight_layout()
        fig.savefig(hist_dir / f"{i + 1:02d}_{name}.png", dpi=90)
        plt.close(fig)


def print_coast_followup(stack_path: Path) -> None:
    """Print aggregate coast/sea follow-up stats (no coordinates)."""
    with rasterio.open(stack_path) as ds:
        names = list(ds.descriptions)
        arr = ds.read()
    finite = np.isfinite(arr)
    area = finite.any(axis=0)
    idx = {n: i for i, n in enumerate(names)}
    water = arr[idx["os_cover_water"]]
    coast = arr[idx["bgs_coast_distance_to_coast"]]
    elev = arr[idx["terrain_dtm"]]
    veg = arr[idx["vom_vegetation_height_max"]]
    valid = area & np.isfinite(water) & np.isfinite(coast)
    sea = valid & (coast == 0)
    print(
        "cells coast_dist==0:",
        int(sea.sum()),
        "share of area %.1f%%" % (100.0 * sea.sum() / valid.sum()),
    )
    print(
        "water mean, all %.2f | excluding coast_dist==0 %.2f | "
        "excluding <1km of coast %.2f"
        % (
            water[valid].mean(),
            water[valid & ~sea].mean(),
            water[valid & (coast >= 1000)].mean(),
        )
    )
    water100 = valid & (water >= 99.99)
    print(
        "100%% water cells: %d, of which coast_dist==0: %.1f%%"
        % (water100.sum(), 100.0 * (water100 & sea).sum() / water100.sum())
    )
    neg = area & (elev < 0) & np.isfinite(coast)
    inland_neg = neg & (coast >= 2000)
    print(
        "elev<0: %d, coast_dist==0: %.1f%%, within 2km of coast: %.1f%%, "
        "min elev where coast>2km: %.2f"
        % (
            neg.sum(),
            100.0 * (neg & (coast == 0)).sum() / neg.sum(),
            100.0 * (neg & (coast < 2000)).sum() / neg.sum(),
            float(elev[inland_neg].min()) if inland_neg.any() else float("nan"),
        )
    )
    print("elev<0 inland (>=2km) cells:", int(inland_neg.sum()))
    print("veg>40 cells:", int((area & (veg > 40)).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write layer_stats.csv for an environmental stack. "
            "Optional histograms for local inspection — do not commit image outputs."
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
        help="Directory for layer_stats.csv (and optional hist/).",
    )
    parser.add_argument(
        "--write-histograms",
        action="store_true",
        help="Write log-scale histogram PNGs under <output-dir>/hist/ (do not commit).",
    )
    parser.add_argument(
        "--print-coast-followup",
        action="store_true",
        help="Print aggregate coast/sea follow-up numbers to stdout.",
    )
    args = parser.parse_args()

    if not args.stack.is_file():
        raise SystemExit(f"Stack not found: {args.stack}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = layer_stats(args.stack)
    out_csv = args.output_dir / "layer_stats.csv"
    stats.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"Wrote {len(stats)} rows to {out_csv}")
    flagged = stats[stats["verdict"] != "ok"]
    if not flagged.empty:
        print(flagged[["variable", "min", "p99", "max", "mean", "n_at_max", "verdict", "reason"]].to_string())

    if args.write_histograms:
        write_histograms(args.stack, args.output_dir / "hist", stats)
        print(f"Wrote histograms under {args.output_dir / 'hist'} (do not commit)")

    if args.print_coast_followup:
        print_coast_followup(args.stack)


if __name__ == "__main__":
    main()
