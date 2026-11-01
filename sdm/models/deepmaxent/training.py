"""Joint DeepMaxent training via PyTorch Lightning, CV, and package assembly."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sdm.commands.modelling.utils import get_model_id
from sdm.models.deepmaxent._torch_util import require_torch
from sdm.models.deepmaxent.config import DeepMaxentParams
from sdm.models.deepmaxent.data import (
    JointDeepMaxentTable,
    build_joint_presence_background_table,
)
from sdm.models.deepmaxent.device import seed_everything
from sdm.models.deepmaxent.estimator import (
    DeepMaxentEnsemble,
    wrap_output_pipeline,
)
from sdm.models.deepmaxent.history import (
    concat_histories,
    find_lightning_metrics_csv,
    metrics_csv_to_history,
)
from sdm.models.deepmaxent.lightning_module import (
    build_deepmaxent_module,
    lightning_accelerator,
    require_lightning,
    resolved_trainer_device,
)
from sdm.models.deepmaxent.network import average_state_dicts, clone_state_dict
from sdm.models.maxent.folds import FoldMethod, FoldParams, iter_fold_splits
from sdm.types import TrainingData, TrainingResults

logger = logging.getLogger(__name__)


@dataclass
class DeepMaxentTrainResult:
    """Joint training artefacts plus per-output TrainingResults for save/predict."""

    ensemble: DeepMaxentEnsemble
    results: List[TrainingResults]
    device: str
    params: DeepMaxentParams
    feature_names: List[str]
    output_names: List[str]
    joint_table: JointDeepMaxentTable
    validation_by_output: Dict[str, pd.DataFrame]
    seed_states: List[Dict[str, Any]]
    training_history: pd.DataFrame


def _species_weights_from_targets(targets: np.ndarray) -> Optional[np.ndarray]:
    """Inverse-frequency weights (optional; off by default in config)."""
    totals = targets.sum(axis=0).astype(float)
    totals = np.maximum(totals, 1.0)
    w = totals.sum() / (totals * totals.size)
    return w.astype(np.float64)


def _numpy_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    balanced_batches: bool = False,
    presences_per_output: int = 4,
):
    """TensorDataset + DataLoader (torch only; no Lightning import needed)."""
    torch, _nn = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from sdm.models.deepmaxent.sampling import build_train_dataloader

    if shuffle:
        return build_train_dataloader(
            X,
            Y,
            batch_size=batch_size,
            seed=seed,
            balanced_batches=balanced_batches,
            presences_per_output=presences_per_output,
        )

    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(Y, dtype=torch.float32),
    )
    n = X.shape[0]
    bs = max(2, min(int(batch_size), max(n, 2)))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def _build_loggers(
    *,
    log_dir: Path,
    fit_id: str,
    seed: int,
    enable_mlflow: bool,
) -> list:
    """CSVLogger always; MLFlowLogger when MLflow is active / requested."""
    L = require_lightning()
    from lightning.pytorch.loggers import CSVLogger

    loggers: list = [
        CSVLogger(
            save_dir=str(log_dir),
            name=f"{fit_id}_seed{seed}",
            version=0,
        )
    ]
    if enable_mlflow:
        try:
            import mlflow
            from lightning.pytorch.loggers import MLFlowLogger

            if mlflow.active_run() is not None or enable_mlflow:
                loggers.append(
                    MLFlowLogger(
                        experiment_name="deepmaxent",
                        run_name=f"deepmaxent_{fit_id}_seed{seed}",
                        tracking_uri=mlflow.get_tracking_uri(),
                    )
                )
        except Exception as exc:
            logger.debug("MLFlowLogger not attached: %s", exc)
    return loggers


def _fit_one_seed_lightning(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    params: DeepMaxentParams,
    seed: int,
    output_names: Sequence[str],
    fit_id: str,
    species_weights: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    log_root: Optional[Path] = None,
    enable_mlflow: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Train one seed with Lightning Trainer; return state, history, device."""
    L = require_lightning()
    seed_everything(seed, device="cpu")

    module = build_deepmaxent_module(
        n_features=X.shape[1],
        n_outputs=Y.shape[1],
        params=params,
        output_names=output_names,
        species_weights=species_weights,
    )
    train_loader = _numpy_dataloader(
        X,
        Y,
        batch_size=params.batch_size,
        shuffle=True,
        seed=seed,
        balanced_batches=params.balanced_batches,
        presences_per_output=params.presences_per_output,
    )
    val_loader = None
    if X_val is not None and Y_val is not None and len(X_val) >= 2:
        val_loader = _numpy_dataloader(
            X_val,
            Y_val,
            batch_size=params.batch_size,
            shuffle=False,
            seed=seed,
        )

    if log_root is None:
        tmp = tempfile.TemporaryDirectory(prefix="deepmaxent_logs_")
        log_dir = Path(tmp.name)
        cleanup_tmp = tmp
    else:
        log_dir = Path(log_root)
        log_dir.mkdir(parents=True, exist_ok=True)
        cleanup_tmp = None

    try:
        loggers = _build_loggers(
            log_dir=log_dir,
            fit_id=fit_id,
            seed=seed,
            enable_mlflow=enable_mlflow,
        )
        callbacks = []
        if params.early_stopping_patience is not None and params.early_stopping_patience > 0:
            from lightning.pytorch.callbacks import EarlyStopping

            monitor = "val_loss" if val_loader is not None else "train_loss"
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    patience=int(params.early_stopping_patience),
                    mode="min",
                )
            )

        accelerator = lightning_accelerator(params.device)
        trainer = L.Trainer(
            max_epochs=int(params.epochs),
            accelerator=accelerator,
            devices=1,
            logger=loggers,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            deterministic=accelerator == "cpu",
            num_sanity_val_steps=0,
        )
        if val_loader is not None:
            trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(module, train_dataloaders=train_loader)

        device = resolved_trainer_device(trainer)
        state = clone_state_dict(module.net)
        metrics_path = find_lightning_metrics_csv(log_dir / f"{fit_id}_seed{seed}")
        if metrics_path is None:
            history = metrics_csv_to_history(
                Path("/nonexistent"),
                fit_id=fit_id,
                seed=seed,
                output_names=output_names,
            )
        else:
            history = metrics_csv_to_history(
                metrics_path,
                fit_id=fit_id,
                seed=seed,
                output_names=output_names,
            )
        return state, history, device
    finally:
        if cleanup_tmp is not None:
            cleanup_tmp.cleanup()


def fit_ensemble(
    table: JointDeepMaxentTable,
    params: DeepMaxentParams,
    *,
    row_index: Optional[np.ndarray] = None,
    val_row_index: Optional[np.ndarray] = None,
    n_seeds: Optional[int] = None,
    fit_id: str = "final",
    log_root: Optional[Path] = None,
    enable_mlflow: bool = False,
) -> Tuple[DeepMaxentEnsemble, List[Dict[str, Any]], str, pd.DataFrame]:
    """Fit StandardScaler + Lightning residual MLP ensemble on joint-table rows."""
    idx = (
        np.arange(len(table.features))
        if row_index is None
        else np.asarray(row_index, dtype=int)
    )
    X_df = table.features.iloc[idx]
    Y_df = table.targets.iloc[idx]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.to_numpy(dtype=float))
    Y = Y_df.to_numpy(dtype=float)

    X_val = Y_val = None
    if val_row_index is not None and len(val_row_index) >= 2:
        val_idx = np.asarray(val_row_index, dtype=int)
        X_val = scaler.transform(table.features.iloc[val_idx].to_numpy(dtype=float))
        Y_val = table.targets.iloc[val_idx].to_numpy(dtype=float)

    weights = (
        _species_weights_from_targets(Y) if params.use_species_weights else None
    )
    seeds_n = params.n_seeds if n_seeds is None else int(n_seeds)
    seed_list = [params.seed + i for i in range(seeds_n)]
    states: List[Dict[str, Any]] = []
    histories: List[pd.DataFrame] = []
    device = "cpu"
    for s in seed_list:
        state, hist, device = _fit_one_seed_lightning(
            X,
            Y,
            params=params,
            seed=s,
            output_names=table.output_names,
            fit_id=fit_id,
            species_weights=weights,
            X_val=X_val,
            Y_val=Y_val,
            log_root=log_root,
            enable_mlflow=enable_mlflow,
        )
        states.append(state)
        histories.append(hist)

    mean_state = average_state_dicts(states)
    ensemble = DeepMaxentEnsemble(
        feature_names=table.feature_names,
        output_names=table.output_names,
        state_dicts=[mean_state],
        hidden_size=params.hidden_size,
        n_hidden_layers=params.n_hidden_layers,
        dropout=params.dropout,
        input_dropout=params.input_dropout,
        transform=params.transform,
        device="cpu" if str(device).startswith("cpu") else str(device).split(":")[0],
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
    )
    # Prefer a plain device label for metadata / sklearn predict.
    device_label = ensemble.device
    return ensemble, states, device_label, concat_histories(histories)


def _points_gdf_for_folds(table: JointDeepMaxentTable) -> gpd.GeoDataFrame:
    any_presence = (table.targets.to_numpy() > 0).any(axis=1).astype(int)
    return gpd.GeoDataFrame(
        {"class": any_presence},
        geometry=table.geometry,
        crs=table.geometry.crs,
    )


def cross_validate_joint(
    table: JointDeepMaxentTable,
    params: DeepMaxentParams,
    fold_params: FoldParams,
    *,
    log_root: Optional[Path] = None,
    enable_mlflow: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame], FoldParams, pd.DataFrame]:
    """Shared spatial-block folds; per-output held-out AUC and Lightning history."""
    if fold_params.method is not FoldMethod.BLOCKS:
        logger.warning(
            "DeepMaxent joint CV expected spatial blocks; got %s",
            fold_params.method.value,
        )
    points = _points_gdf_for_folds(table)
    splits = list(iter_fold_splits(points, fold_params))
    scores: Dict[str, List[float]] = {name: [] for name in table.output_names}
    val_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in table.output_names}
    histories: List[pd.DataFrame] = []

    cv_params = DeepMaxentParams(
        **{**params.to_metadata(), "n_seeds": 1, "epochs": params.epochs}
    )

    for fold_id, train_idx, test_idx in splits:
        if len(train_idx) < 4 or len(test_idx) < 2:
            for name in table.output_names:
                scores[name].append(np.nan)
            continue
        ensemble, _states, _device, hist = fit_ensemble(
            table,
            cv_params,
            row_index=train_idx,
            val_row_index=test_idx,
            n_seeds=1,
            fit_id=f"fold_{fold_id}",
            log_root=log_root,
            enable_mlflow=enable_mlflow,
        )
        histories.append(hist)
        proba = ensemble.predict_proba_multi(table.features.iloc[test_idx])
        for j, name in enumerate(table.output_names):
            y = (table.targets.iloc[test_idx][name].to_numpy() > 0).astype(int)
            p = proba[:, j]
            if len(np.unique(y)) < 2:
                scores[name].append(np.nan)
            else:
                try:
                    scores[name].append(float(roc_auc_score(y, p)))
                except ValueError:
                    scores[name].append(np.nan)
            for local_i, point_idx in enumerate(test_idx):
                val_rows[name].append(
                    {
                        "point_index": int(point_idx),
                        "class": int(y[local_i]),
                        "fold": int(fold_id),
                        "held_out_score": float(p[local_i]),
                    }
                )

    score_arrays = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    val_frames = {
        k: pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["point_index", "class", "fold", "held_out_score"])
        for k, rows in val_rows.items()
    }
    return score_arrays, val_frames, fold_params, concat_histories(histories)


def _latin_activity_from_id(
    model_id: str, training_data: Sequence[TrainingData]
) -> Tuple[str, str]:
    for data in training_data:
        if get_model_id([data.latin_name, data.activity_type]) == model_id:
            return data.latin_name, data.activity_type
    raise KeyError(f"No TrainingData matching model_id={model_id!r}")


def train_deepmaxent_joint(
    training_data: Sequence[TrainingData],
    params: DeepMaxentParams,
    fold_params: FoldParams,
    *,
    log_root: Optional[Path] = None,
    enable_mlflow: bool = False,
) -> DeepMaxentTrainResult:
    """Train one shared multi-output DeepMaxent model (Lightning Trainer)."""
    table = build_joint_presence_background_table(training_data)
    logger.info(
        "Training DeepMaxent joint model (%d outputs) via Lightning "
        "(device preference=%s)",
        len(table.output_names),
        params.device,
    )
    cv_scores, val_by_out, used_folds, cv_history = cross_validate_joint(
        table,
        params,
        fold_params,
        log_root=log_root,
        enable_mlflow=enable_mlflow,
    )
    ensemble, seed_states, device, final_history = fit_ensemble(
        table,
        params,
        fit_id="final",
        log_root=log_root,
        enable_mlflow=enable_mlflow,
    )
    ensemble.state_dicts = [average_state_dicts(seed_states)]
    training_history = concat_histories([cv_history, final_history])

    results: List[TrainingResults] = []
    for name in table.output_names:
        latin, activity = _latin_activity_from_id(name, training_data)
        pipe = wrap_output_pipeline(ensemble, name)
        results.append(
            TrainingResults(
                latin_name=latin,
                activity_type=activity,
                final_model=pipe,
                cv_models=None,
                cv_scores=cv_scores.get(name),
                validation_scores=val_by_out.get(name),
                cv_n_folds=used_folds.n_folds,
                fold_method=used_folds.method.value,
                fold_params=used_folds.to_metadata(),
                fold_presence_counts=None,
                success=True,
                error=None,
            )
        )

    return DeepMaxentTrainResult(
        ensemble=ensemble,
        results=results,
        device=device,
        params=params,
        feature_names=table.feature_names,
        output_names=table.output_names,
        joint_table=table,
        validation_by_output=val_by_out,
        seed_states=seed_states,
        training_history=training_history,
    )
