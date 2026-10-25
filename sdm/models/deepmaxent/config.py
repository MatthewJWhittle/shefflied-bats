"""DeepMaxent hyperparameters resolved from ``model_config.yml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from sdm.types import DeepMaxentConfigModel, ModelConfig


@dataclass(frozen=True)
class DeepMaxentParams:
    """Resolved training / architecture knobs for the joint DeepMaxent model."""

    device: str = "auto"
    hidden_size: int = 250
    n_hidden_layers: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 250
    weight_decay: float = 3e-4
    epochs: int = 100
    n_seeds: int = 10
    epsilon: float = 1e-6
    use_species_weights: bool = False
    transform: str = "cloglog"
    early_stopping_patience: Optional[int] = None
    seed: int = 42

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_deepmaxent_params(
    model_config: ModelConfig,
    *,
    seed_override: Optional[int] = None,
) -> DeepMaxentParams:
    """Build ``DeepMaxentParams`` from a ``ModelConfig`` (defaults fill gaps)."""
    section: Optional[DeepMaxentConfigModel] = getattr(model_config, "deepmaxent", None)
    seed = seed_override if seed_override is not None else int(model_config.seed)
    if section is None:
        return DeepMaxentParams(seed=seed)
    return DeepMaxentParams(
        device=section.device,
        hidden_size=section.hidden_size,
        n_hidden_layers=section.n_hidden_layers,
        learning_rate=section.learning_rate,
        batch_size=section.batch_size,
        weight_decay=section.weight_decay,
        epochs=section.epochs,
        n_seeds=section.n_seeds,
        epsilon=section.epsilon,
        use_species_weights=section.use_species_weights,
        transform=section.transform,
        early_stopping_patience=section.early_stopping_patience,
        seed=seed,
    )


def default_output_names_from_training_ids(model_ids: List[str]) -> List[str]:
    """Stable output-head order for a joint model (sorted filesystem ids)."""
    return sorted(model_ids)
