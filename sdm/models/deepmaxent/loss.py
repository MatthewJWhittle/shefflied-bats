"""DeepMaxent batched MaxEnt / Poisson loss (Ryckewaert et al. 2024, eq. 4)."""

from __future__ import annotations

from typing import Optional

from sdm.models.deepmaxent._torch_util import require_torch


def deepmaxent_loss(
    logits,
    targets,
    *,
    epsilon: float = 1e-6,
    species_weights: Optional[object] = None,
):
    """Mini-batch DeepMaxent loss (paper equation 4).

    Softmax is over **sites in the mini-batch**. With optional balanced
    sampling (``balanced_batches``), oversampled presence rows for one output
    also sit in every other output's partition as non-presences; eq. 4 already
    treats them as background in those denominators — no further correction.

    Parameters
    ----------
    logits :
        ``[batch, n_species]`` log-intensities.
    targets :
        ``[batch, n_species]`` non-negative occurrence counts (or 0/1).
    epsilon :
        Added to per-species batch count so species absent from the batch
        contribute zero (their normalised weights are zero).
    species_weights :
        Optional ``[n_species]`` multipliers (imbalance reweighting). Off by
        default in config; when provided they scale each species term before
        the mean. Known risk: count weighting can pull rare species toward
        common patterns — leave disabled unless explicitly enabling.
    """
    torch, _nn = require_torch()
    import torch.nn.functional as F

    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError("logits and targets must be rank-2 [batch, n_species]")
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}"
        )

    batch_size = logits.shape[0]
    n_species = logits.shape[1]
    if batch_size == 0 or n_species == 0:
        return logits.sum() * 0.0

    # Softmax over sites in the mini-batch (partition function approximation).
    log_probs = F.log_softmax(logits, dim=0)
    count_per_species = targets.sum(dim=0)  # [N]
    # y_ij / (sum_i y_ij + eps)
    weights = targets / (count_per_species.unsqueeze(0) + float(epsilon))
    # sum_i weight_ij * log_prob_ij  → [N]
    per_species = (weights * log_probs).sum(dim=0)
    if species_weights is not None:
        sw = torch.as_tensor(species_weights, device=logits.device, dtype=logits.dtype)
        per_species = per_species * sw
    # −1 / (|B| N) Σ_i Σ_j …
    return -(per_species.mean()) / float(batch_size)
