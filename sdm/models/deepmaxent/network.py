"""Residual MLP feature extractor + multi-species linear heads (DeepMaxent)."""

from __future__ import annotations

from typing import List

from sdm.models.deepmaxent._torch_util import require_torch


def build_residual_mlp(
    n_features: int,
    n_outputs: int,
    *,
    hidden_size: int = 250,
    n_hidden_layers: int = 2,
    dropout: float = 0.0,
    input_dropout: float = 0.0,
):
    """Shared trunk: BatchNorm over sites, ReLU hidden layers with residual skips,
    optional dropout on hidden activations, then one logit per output.

    Residual skips apply between consecutive hidden layers of equal width
    (works for 2 or 4+ layers). ``dropout=0`` (default) disables hidden dropout.
    ``input_dropout`` (default 0) drops input feature channels in ``train()``
    mode only (``nn.Dropout``; inactive in ``eval()`` / predict).
    """
    torch, nn = require_torch()
    rate = float(dropout)
    if rate < 0.0 or rate >= 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
    in_rate = float(input_dropout)
    if in_rate < 0.0 or in_rate >= 1.0:
        raise ValueError(f"input_dropout must be in [0, 1), got {input_dropout!r}")

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if n_hidden_layers < 1:
                raise ValueError("n_hidden_layers must be >= 1")
            if hidden_size < 1:
                raise ValueError("hidden_size must be >= 1")
            self.input_bn = nn.BatchNorm1d(n_features)
            self.input_dropout = nn.Dropout(p=in_rate)
            self.fc_in = nn.Linear(n_features, hidden_size)
            self.hidden = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden_layers)]
            )
            self.dropout = nn.Dropout(p=rate)
            self.fc_out = nn.Linear(hidden_size, n_outputs)
            self.dropout_rate = rate
            self.input_dropout_rate = in_rate

        def forward(self, x):
            x = self.input_bn(x)
            x = self.input_dropout(x)
            x = self.dropout(self.fc_in(x).relu())
            for layer in self.hidden:
                # Residual skip between hidden layers (equal width).
                x = self.dropout(layer(x).relu()) + x
            return self.fc_out(x)

    return ResidualMLP()


def logits_to_suitability(logits, transform: str = "cloglog"):
    """Map log-intensity logits to [0, 1] suitability scores."""
    torch, _nn = require_torch()
    name = (transform or "cloglog").lower()
    if name == "cloglog":
        # Complementary log-log of intensity λ = exp(logit): 1 - exp(-λ).
        return 1.0 - torch.exp(-torch.exp(logits.clamp(max=20.0)))
    if name == "logistic":
        return torch.sigmoid(logits)
    if name in {"raw", "intensity"}:
        return torch.exp(logits.clamp(max=20.0))
    raise ValueError(f"Unknown DeepMaxent transform {transform!r}")


def clone_state_dict(model) -> dict:
    """CPU-cloned state dict suitable for pickling ensembles."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_into(model, state: dict, device: str):
    model.load_state_dict(state)
    model.to(device)
    return model


def average_state_dicts(states: List[dict]) -> dict:
    """Element-wise mean of state dicts (same keys / shapes)."""
    if not states:
        raise ValueError("No state dicts to average")
    keys = states[0].keys()
    out = {}
    for key in keys:
        stacked = [s[key].float() for s in states]
        out[key] = sum(stacked) / float(len(stacked))
    return out
