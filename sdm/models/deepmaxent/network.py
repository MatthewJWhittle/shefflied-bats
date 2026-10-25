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
):
    """Shared trunk: BatchNorm over sites, two (or more) ReLU hidden layers with
    residual skips between consecutive hidden layers, then one logit per output.

    Matches Ryckewaert et al. 2024 (DeepMaxent) residual MLP; BatchNorm1d is
    applied on the site batch so covariates are standardised per mini-batch.
    """
    torch, nn = require_torch()

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if n_hidden_layers < 1:
                raise ValueError("n_hidden_layers must be >= 1")
            self.input_bn = nn.BatchNorm1d(n_features)
            self.fc_in = nn.Linear(n_features, hidden_size)
            self.hidden = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden_layers)]
            )
            self.fc_out = nn.Linear(hidden_size, n_outputs)

        def forward(self, x):
            x = self.input_bn(x)
            x = self.fc_in(x).relu()
            for layer in self.hidden:
                x = layer(x).relu() + x
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
