"""Joint DeepMaxent training, spatial-block CV, and package assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from sdm.models.deepmaxent.device import resolve_torch_device, seed_everything
from sdm.models.deepmaxent.estimator import (
    DeepMaxentEnsemble,
    wrap_output_pipeline,
)
from sdm.models.deepmaxent.loss import deepmaxent_loss
from sdm.models.deepmaxent.network import (
    average_state_dicts,
    build_residual_mlp,
    clone_state_dict,
)
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


def _species_weights_from_targets(targets: np.ndarray) -> Optional[np.ndarray]:
    """Inverse-frequency weights (optional; off by default in config)."""
    totals = targets.sum(axis=0).astype(float)
    totals = np.maximum(totals, 1.0)
    w = totals.sum() / (totals * totals.size)
    return w.astype(np.float64)


def _fit_one_seed(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    params: DeepMaxentParams,
    device: str,
    seed: int,
    species_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    torch, _nn = require_torch()
    seed_everything(seed, device=device)
    n_features = X.shape[1]
    n_outputs = Y.shape[1]
    model = build_residual_mlp(
        n_features,
        n_outputs,
        hidden_size=params.hidden_size,
        n_hidden_layers=params.n_hidden_layers,
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    n = X.shape[0]
    batch_size = max(2, min(params.batch_size, n))
    rng = np.random.default_rng(seed)
    best_state = clone_state_dict(model)
    best_loss = float("inf")
    patience = params.early_stopping_patience
    wait = 0

    x_all = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_all = torch.as_tensor(Y, dtype=torch.float32, device=device)
    sw = None
    if species_weights is not None:
        sw = torch.as_tensor(species_weights, dtype=torch.float32, device=device)

    for epoch in range(params.epochs):
        model.train()
        order = rng.permutation(n)
        epoch_losses: List[float] = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if idx.size < 2:
                continue
            logits = model(x_all[idx])
            loss = deepmaxent_loss(
                logits,
                y_all[idx],
                epsilon=params.epsilon,
                species_weights=sw,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        if mean_loss < best_loss - 1e-8:
            best_loss = mean_loss
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if patience is not None and patience > 0 and wait >= patience:
                logger.info(
                    "DeepMaxent early stop at epoch %d (seed=%d, loss=%.5f)",
                    epoch + 1,
                    seed,
                    best_loss,
                )
                break
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.debug(
                "DeepMaxent seed=%d epoch=%d/%d loss=%.5f",
                seed,
                epoch + 1,
                params.epochs,
                mean_loss,
            )

    return best_state


def fit_ensemble(
    table: JointDeepMaxentTable,
    params: DeepMaxentParams,
    *,
    row_index: Optional[np.ndarray] = None,
    n_seeds: Optional[int] = None,
) -> Tuple[DeepMaxentEnsemble, List[Dict[str, Any]], str]:
    """Fit StandardScaler + residual MLP ensemble on selected joint-table rows."""
    device = resolve_torch_device(params.device)
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

    weights = (
        _species_weights_from_targets(Y) if params.use_species_weights else None
    )
    seeds_n = params.n_seeds if n_seeds is None else int(n_seeds)
    seed_list = [params.seed + i for i in range(seeds_n)]
    states: List[Dict[str, Any]] = []
    for s in seed_list:
        states.append(
            _fit_one_seed(
                X,
                Y,
                params=params,
                device=device,
                seed=s,
                species_weights=weights,
            )
        )

    # Store mean weights for a compact default predictor; keep all seeds for spread.
    mean_state = average_state_dicts(states)
    ensemble = DeepMaxentEnsemble(
        feature_names=table.feature_names,
        output_names=table.output_names,
        state_dicts=[mean_state],
        hidden_size=params.hidden_size,
        n_hidden_layers=params.n_hidden_layers,
        transform=params.transform,
        device=device,
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
    )
    return ensemble, states, device


def _points_gdf_for_folds(table: JointDeepMaxentTable) -> gpd.GeoDataFrame:
    """GeoDataFrame with class=1 on any presence so block CV sees all sites."""
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame], FoldParams]:
    """Shared 10 km block folds; per-output held-out AUC and score tables."""
    if fold_params.method is not FoldMethod.BLOCKS:
        logger.warning(
            "DeepMaxent joint CV expected spatial blocks; got %s",
            fold_params.method.value,
        )
    points = _points_gdf_for_folds(table)
    splits = list(iter_fold_splits(points, fold_params))
    scores: Dict[str, List[float]] = {name: [] for name in table.output_names}
    val_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in table.output_names}

    # One seed during CV for speed; final fit uses full n_seeds.
    cv_params = DeepMaxentParams(**{**params.to_metadata(), "n_seeds": 1, "epochs": params.epochs})

    for fold_id, train_idx, test_idx in splits:
        if len(train_idx) < 4 or len(test_idx) < 2:
            for name in table.output_names:
                scores[name].append(np.nan)
            continue
        ensemble, _states, _device = fit_ensemble(
            table, cv_params, row_index=train_idx, n_seeds=1
        )
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
        k: pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["point_index", "class", "fold", "held_out_score"]
        )
        for k, rows in val_rows.items()
    }
    return score_arrays, val_frames, fold_params


def _latin_activity_from_id(model_id: str, training_data: Sequence[TrainingData]) -> Tuple[str, str]:
    for data in training_data:
        if get_model_id([data.latin_name, data.activity_type]) == model_id:
            return data.latin_name, data.activity_type
    # Fallback parse: last token roost/in_flight style is unreliable; raise.
    raise KeyError(f"No TrainingData matching model_id={model_id!r}")


def train_deepmaxent_joint(
    training_data: Sequence[TrainingData],
    params: DeepMaxentParams,
    fold_params: FoldParams,
) -> DeepMaxentTrainResult:
    """Train one shared multi-output DeepMaxent model for all TrainingData rows."""
    table = build_joint_presence_background_table(training_data)
    logger.info(
        "Training DeepMaxent joint model (%d outputs) on device preference=%s",
        len(table.output_names),
        params.device,
    )
    cv_scores, val_by_out, used_folds = cross_validate_joint(table, params, fold_params)
    ensemble, seed_states, device = fit_ensemble(table, params)
    # Attach all seed states for optional spread analysis; predict uses mean.
    ensemble.state_dicts = [average_state_dicts(seed_states)]

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
    )
