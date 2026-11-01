"""PyTorch Lightning module wrapping DeepMaxent network + count-weighted loss."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sdm.models.deepmaxent._torch_util import require_torch
from sdm.models.deepmaxent.config import DeepMaxentParams
from sdm.models.deepmaxent.loss import deepmaxent_loss
from sdm.models.deepmaxent.network import build_residual_mlp, logits_to_suitability


def require_lightning():
    """Import lightning or raise a clear optional-extra error."""
    try:
        import lightning as L
    except ImportError as exc:
        raise ImportError(
            "PyTorch Lightning is required for DeepMaxent. "
            "Install with: uv sync --extra dl"
        ) from exc
    return L


def build_deepmaxent_module(
    *,
    n_features: int,
    n_outputs: int,
    params: DeepMaxentParams,
    output_names: Sequence[str],
    species_weights: Optional[np.ndarray] = None,
):
    """Construct a :class:`DeepMaxentLightningModule` (lazy Lightning import)."""
    L = require_lightning()
    torch, _nn = require_torch()

    class DeepMaxentLightningModule(L.LightningModule):
        """Shared residual MLP + DeepMaxent loss with per-epoch val AUC."""

        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(
                {
                    "hidden_size": params.hidden_size,
                    "n_hidden_layers": params.n_hidden_layers,
                    "dropout": params.dropout,
                    "input_dropout": params.input_dropout,
                    "learning_rate": params.learning_rate,
                    "weight_decay": params.weight_decay,
                    "epsilon": params.epsilon,
                    "transform": params.transform,
                    "use_species_weights": params.use_species_weights,
                    "n_features": n_features,
                    "n_outputs": n_outputs,
                }
            )
            self.params = params
            self.output_names = list(output_names)
            self.net = build_residual_mlp(
                n_features,
                n_outputs,
                hidden_size=params.hidden_size,
                n_hidden_layers=params.n_hidden_layers,
                dropout=params.dropout,
                input_dropout=params.input_dropout,
            )
            self._species_weights = None
            if species_weights is not None:
                self.register_buffer(
                    "_species_weights_buf",
                    torch.as_tensor(species_weights, dtype=torch.float32),
                )
            else:
                self._species_weights_buf = None
            self._val_logits: List[Any] = []
            self._val_targets: List[Any] = []

        def forward(self, x):
            return self.net(x)

        def _sw(self):
            return self._species_weights_buf

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.net(x)
            loss = deepmaxent_loss(
                logits,
                y,
                epsilon=self.params.epsilon,
                species_weights=self._sw(),
            )
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.shape[0],
            )
            return loss

        def on_validation_epoch_start(self) -> None:
            self._val_logits = []
            self._val_targets = []

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self.net(x)
            self._val_logits.append(logits.detach())
            self._val_targets.append(y.detach())
            loss = deepmaxent_loss(
                logits,
                y,
                epsilon=self.params.epsilon,
                species_weights=self._sw(),
            )
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=x.shape[0],
            )
            return loss

        def on_validation_epoch_end(self) -> None:
            if not self._val_logits:
                return
            from sklearn.metrics import roc_auc_score

            logits = torch.cat(self._val_logits, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            suit = logits_to_suitability(logits, transform=self.params.transform)
            suit_np = suit.detach().cpu().numpy()
            y_np = targets.detach().cpu().numpy()
            aucs: List[float] = []
            for j, name in enumerate(self.output_names):
                y_bin = (y_np[:, j] > 0).astype(int)
                if len(np.unique(y_bin)) < 2:
                    auc = float("nan")
                else:
                    try:
                        auc = float(roc_auc_score(y_bin, suit_np[:, j]))
                    except ValueError:
                        auc = float("nan")
                if np.isfinite(auc):
                    aucs.append(auc)
                    self.log(f"val_auc__{name}", auc, on_epoch=True, prog_bar=False)
            if aucs:
                self.log("val_auc_mean", float(np.mean(aucs)), on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.Adam(
                self.parameters(),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
            )

    return DeepMaxentLightningModule()


def lightning_accelerator(device_preference: str = "auto") -> str:
    """Map config device preference to a Lightning ``accelerator`` string."""
    pref = (device_preference or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "gpu"
    if pref == "mps":
        return "mps"
    return "auto"


def resolved_trainer_device(trainer) -> str:
    """Best-effort device label for run metadata."""
    try:
        dev = trainer.strategy.root_device
        return str(dev)
    except Exception:
        pass
    try:
        return str(trainer.accelerator.__class__.__name__)
    except Exception:
        return "auto"
