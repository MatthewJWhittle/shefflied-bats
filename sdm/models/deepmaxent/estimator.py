"""Sklearn-compatible wrappers around a fitted DeepMaxent network."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from sdm.models.deepmaxent._torch_util import require_torch
from sdm.models.deepmaxent.network import (
    build_residual_mlp,
    logits_to_suitability,
)


class DeepMaxentEnsemble(BaseEstimator, ClassifierMixin):
    """Multi-output DeepMaxent model (optionally averaged over seeds).

    ``predict_proba`` returns suitability in ``[0, 1]`` for every output head
    (shape ``[n_samples, n_outputs]``). Use :class:`DeepMaxentOutputEstimator`
    for per-model packages expected by ``sdm predict``.
    """

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        output_names: Sequence[str],
        state_dicts: Sequence[Dict[str, Any]],
        hidden_size: int = 250,
        n_hidden_layers: int = 2,
        dropout: float = 0.0,
        transform: str = "cloglog",
        device: str = "cpu",
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.output_names = list(output_names)
        self.state_dicts = list(state_dicts)
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = float(dropout)
        self.transform = transform
        self.device = device
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale


    def _scaler(self) -> StandardScaler:
        scaler = StandardScaler()
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("DeepMaxentEnsemble is missing fitted scaler stats")
        scaler.mean_ = np.asarray(self.scaler_mean, dtype=float)
        scaler.scale_ = np.asarray(self.scaler_scale, dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(self.feature_names)
        return scaler

    def _matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing features for DeepMaxent predict: {missing}")
            arr = X[self.feature_names].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected X with {len(self.feature_names)} columns, got shape {arr.shape}"
                )
        return self._scaler().transform(arr)

    def _predict_logits(self, X) -> np.ndarray:
        torch, _nn = require_torch()
        if not self.state_dicts:
            raise RuntimeError("DeepMaxentEnsemble has no fitted state dicts")
        x = self._matrix(X)
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        logits_sum = None
        for state in self.state_dicts:
            model = build_residual_mlp(
                len(self.feature_names),
                len(self.output_names),
                hidden_size=self.hidden_size,
                n_hidden_layers=self.n_hidden_layers,
                dropout=self.dropout,
            )
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        assert logits_sum is not None
        mean_logits = logits_sum / float(len(self.state_dicts))
        return mean_logits.detach().cpu().numpy()

    def predict_proba_multi(self, X) -> np.ndarray:
        torch, _nn = require_torch()
        logits = self._predict_logits(X)
        t = torch.as_tensor(logits, dtype=torch.float32)
        suit = logits_to_suitability(t, transform=self.transform)
        return suit.numpy()

    def predict_proba(self, X) -> np.ndarray:
        """Sklearn binary API for the first output (prefer OutputEstimator)."""
        multi = self.predict_proba_multi(X)
        p1 = multi[:, 0]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba_multi(X) >= 0.5).astype(int)

    def output_index(self, output_name: str) -> int:
        try:
            return self.output_names.index(output_name)
        except ValueError as exc:
            raise KeyError(
                f"Unknown DeepMaxent output {output_name!r}; "
                f"have {self.output_names}"
            ) from exc


class DeepMaxentOutputEstimator(BaseEstimator, ClassifierMixin):
    """Single-output view of a joint DeepMaxent ensemble for model packages."""

    def __init__(
        self,
        ensemble: DeepMaxentEnsemble,
        output_name: str,
    ) -> None:
        self.ensemble = ensemble
        self.output_name = output_name

    @property
    def feature_names(self) -> List[str]:
        return list(self.ensemble.feature_names)

    def predict_proba(self, X) -> np.ndarray:
        idx = self.ensemble.output_index(self.output_name)
        multi = self.ensemble.predict_proba_multi(X)
        p1 = multi[:, idx]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_clone__(self):
        # Ensemble weights are shared; cloning for CV uses fresh trainers instead.
        return DeepMaxentOutputEstimator(self.ensemble, self.output_name)


def wrap_output_pipeline(
    ensemble: DeepMaxentEnsemble,
    output_name: str,
):
    """ColumnTransformer + single-output estimator matching MaxEnt package shape.

    Predict paths call ``predict_proba`` on the fitted object; packages store this
    pipeline so ``sdm predict --run`` stays unchanged.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    cols = list(ensemble.feature_names)
    column_selector = ColumnTransformer(
        [("features", "passthrough", cols)],
        remainder="drop",
        sparse_threshold=0,
    )
    # Numpy output (default): DeepMaxentEnsemble._matrix accepts arrays by width.
    # Fit on a one-row dummy frame so transform works at predict time.
    dummy = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    column_selector.fit(dummy)

    head = DeepMaxentOutputEstimator(ensemble, output_name)
    head.classes_ = np.array([0, 1])
    pipe = Pipeline(
        [
            ("feature_selection", column_selector),
            ("deepmaxent", head),
        ]
    )
    # Pipeline considers itself fitted when all steps expose ``__sklearn_is_fitted__``
    # or have been through ``fit``. Selector was fitted above; head is predict-only.
    pipe._sklearn_is_fitted_ = True  # type: ignore[attr-defined]
    return pipe
