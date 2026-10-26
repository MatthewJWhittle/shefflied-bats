"""Reference hand-rolled DeepMaxent fit (tests only: Lightning equivalence).

Production training uses PyTorch Lightning (:mod:`sdm.models.deepmaxent.training`).
This module keeps the previous Adam loop so tests can show the Lightning path
matches on a small synthetic case (same loss, optimiser, and seed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from sdm.models.deepmaxent._torch_util import require_torch
from sdm.models.deepmaxent.config import DeepMaxentParams
from sdm.models.deepmaxent.device import seed_everything
from sdm.models.deepmaxent.loss import deepmaxent_loss
from sdm.models.deepmaxent.network import build_residual_mlp, clone_state_dict


def fit_one_seed_manual(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    params: DeepMaxentParams,
    device: str,
    seed: int,
    species_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train one residual MLP with the pre-Lightning loop; return state dict."""
    torch, _nn = require_torch()
    seed_everything(seed, device=device)
    model = build_residual_mlp(
        X.shape[1],
        Y.shape[1],
        hidden_size=params.hidden_size,
        n_hidden_layers=params.n_hidden_layers,
        dropout=params.dropout,
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    n = X.shape[0]
    batch_size = max(2, min(params.batch_size, n))
    rng = np.random.default_rng(seed)
    x_all = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_all = torch.as_tensor(Y, dtype=torch.float32, device=device)
    sw = None
    if species_weights is not None:
        sw = torch.as_tensor(species_weights, dtype=torch.float32, device=device)

    best_state = clone_state_dict(model)
    best_loss = float("inf")
    for _epoch in range(params.epochs):
        model.train()
        order = rng.permutation(n)
        epoch_losses = []
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
    return best_state


def predict_logits_from_state(
    state: Dict[str, Any],
    X: np.ndarray,
    *,
    n_outputs: int,
    params: DeepMaxentParams,
    device: str = "cpu",
) -> np.ndarray:
    """Forward pass with a saved state dict (eval mode)."""
    torch, _nn = require_torch()
    model = build_residual_mlp(
        X.shape[1],
        n_outputs,
        hidden_size=params.hidden_size,
        n_hidden_layers=params.n_hidden_layers,
        dropout=params.dropout,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    return logits.detach().cpu().numpy()
