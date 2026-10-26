"""DeepMaxent training history from Lightning CSVLogger + plot helper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_CSV_NAME = "deepmaxent_training_history.csv"


def empty_history_frame(output_names: Sequence[str]) -> pd.DataFrame:
    """Schema for one row per (fit, seed, epoch)."""
    cols = [
        "fit_id",
        "seed",
        "epoch",
        "train_loss",
        "val_loss",
        "val_auc_mean",
    ]
    cols.extend(f"val_auc__{name}" for name in output_names)
    return pd.DataFrame(columns=cols)


def concat_histories(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and len(f) > 0]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def metrics_csv_to_history(
    metrics_csv: Path,
    *,
    fit_id: str,
    seed: int,
    output_names: Sequence[str],
) -> pd.DataFrame:
    """Convert a Lightning CSVLogger ``metrics.csv`` into our history schema."""
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return empty_history_frame(output_names)
    raw = pd.read_csv(metrics_csv)
    if raw.empty or "epoch" not in raw.columns:
        return empty_history_frame(output_names)

    # Lightning writes one row per logged call; collapse to one row per epoch.
    rows: List[Dict[str, Any]] = []
    for epoch, group in raw.groupby("epoch", sort=True):
        # Prefer non-null values within the epoch group.
        def _first(col: str) -> float:
            if col not in group.columns:
                return float("nan")
            series = group[col].dropna()
            if series.empty:
                return float("nan")
            return float(series.iloc[-1])

        auc_cols = {
            name: _first(f"val_auc__{name}") for name in output_names
        }
        finite = [v for v in auc_cols.values() if np.isfinite(v)]
        mean_auc = _first("val_auc_mean")
        if not np.isfinite(mean_auc) and finite:
            mean_auc = float(np.mean(finite))
        row: Dict[str, Any] = {
            "fit_id": fit_id,
            "seed": int(seed),
            # Lightning epochs are 0-indexed; store 1-based for readability.
            "epoch": int(epoch) + 1,
            "train_loss": _first("train_loss"),
            "val_loss": _first("val_loss"),
            "val_auc_mean": mean_auc,
        }
        for name, auc in auc_cols.items():
            row[f"val_auc__{name}"] = auc
        rows.append(row)
    return pd.DataFrame(rows) if rows else empty_history_frame(output_names)


def find_lightning_metrics_csv(log_dir: Path) -> Optional[Path]:
    """Locate the newest ``metrics.csv`` under a Lightning CSVLogger directory."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_training_history(history: pd.DataFrame, output_dir: Path) -> Path:
    """Write ``deepmaxent_training_history.csv`` under the run folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / HISTORY_CSV_NAME
    history.to_csv(path, index=False)
    logger.info("Wrote DeepMaxent training history (%d rows) to %s", len(history), path)
    return path


def plot_training_history(
    history: pd.DataFrame,
    output_path: Path,
    *,
    max_outputs_plotted: int = 8,
) -> Path:
    """One figure: train/val loss and mean / per-output AUC vs epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if history is None or len(history) == 0:
        raise ValueError("history is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    auc_cols = [c for c in history.columns if c.startswith("val_auc__")]
    fit_ids = list(history["fit_id"].unique())

    n_fit = len(fit_ids)
    fig, axes = plt.subplots(
        n_fit,
        2,
        figsize=(10, max(3.0, 2.8 * n_fit)),
        squeeze=False,
        sharex=False,
    )

    for row_i, fit_id in enumerate(fit_ids):
        sub = history[history["fit_id"] == fit_id]
        ax_loss, ax_auc = axes[row_i][0], axes[row_i][1]
        for seed, g in sub.groupby("seed"):
            g = g.sort_values("epoch")
            if g["train_loss"].notna().any():
                ax_loss.plot(
                    g["epoch"], g["train_loss"], label=f"train seed={seed}", alpha=0.85
                )
            if g["val_loss"].notna().any():
                ax_loss.plot(
                    g["epoch"],
                    g["val_loss"],
                    linestyle="--",
                    label=f"val seed={seed}",
                    alpha=0.85,
                )
            if g["val_auc_mean"].notna().any():
                ax_auc.plot(
                    g["epoch"],
                    g["val_auc_mean"],
                    label=f"mean AUC seed={seed}",
                    linewidth=2,
                )
            for col in auc_cols[:max_outputs_plotted]:
                if g[col].notna().any():
                    ax_auc.plot(
                        g["epoch"],
                        g[col],
                        alpha=0.35,
                        linewidth=1,
                        label=(
                            col.replace("val_auc__", "")
                            if seed == sub["seed"].min()
                            else None
                        ),
                    )
        ax_loss.set_title(f"{fit_id}: loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.legend(fontsize=7, loc="best")
        ax_auc.set_title(f"{fit_id}: AUC")
        ax_auc.set_xlabel("epoch")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_ylim(0.0, 1.05)
        ax_auc.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
