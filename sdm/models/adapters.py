"""Interoperable model adapters for ``sdm train`` / ``predict`` / ``tune``.

Shared CLI orchestration resolves a ``ModelAdapter`` by ``model.type`` (train/tune)
or package/run ``trainer`` (predict). MaxEnt is the only registered backend here;
DeepMaxent and later methods add a class + registry entry (see
``docs/modelling/adapters.md``). Do not introduce façade pickles for new trainers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from sklearn.metrics import roc_auc_score

from sdm.models.maxent.maxent_model import (
    create_maxent_pipeline,
    cross_validate_maxent_model,
)
from sdm.models.maxent.training import summarize_cv_scores, train_models_parallel
from sdm.types import TrainingData, TrainingResults
from sdm.utils.io import load_pickled_model
from sdm.utils.rng import apply_training_seed

DEFAULT_TRAINER = "maxent"


def normalize_trainer_id(value: Optional[str]) -> str:
    """Lowercase / strip a trainer id; empty or None → maxent."""
    if value is None:
        return DEFAULT_TRAINER
    key = str(value).strip().lower()
    return key or DEFAULT_TRAINER


@dataclass
class TrainBundle:
    """Adapter train output consumed by shared run-folder writers."""

    results: List[TrainingResults]
    training_data: List[TrainingData]


class Predictor(Protocol):
    """Minimal scoring surface used after package load (sklearn Pipeline today)."""

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities; MaxEnt raster path uses column ``[:, 1]``."""


class ModelAdapter(Protocol):
    """Train / load surface for one modelling method."""

    name: str

    def train(
        self,
        training_data: Sequence[TrainingData],
        *,
        seed: int,
        fold_params: Any = None,
        max_threads_per_model: int = 1,
        n_jobs: Optional[int] = None,
    ) -> TrainBundle:
        """Fit models and return a bundle for shared I/O."""

    def load_predictor(self, model_path: Path) -> Predictor:
        """Load a scorer from a package pickle path (or equivalent)."""


class Tunable(Protocol):
    """Optuna suggest / eval surface; adapters implement this when tune is ready."""

    def suggest_hyperparameters(self, trial: Any) -> Dict[str, Any]:
        """Suggest method-specific hyperparameters for one Optuna trial."""

    def eval_trial(
        self,
        data: TrainingData,
        *,
        seed: int,
        max_threads_per_model: int = 1,
        n_cv_folds: int = 3,
    ) -> Tuple[float, float]:
        """Evaluate one trial; return ``(mean_cv_score, std_cv_score)``."""


class MaxEntAdapter:
    """elapid / sklearn Pipeline MaxEnt — current production trainer."""

    name = DEFAULT_TRAINER

    def train(
        self,
        training_data: Sequence[TrainingData],
        *,
        seed: int,
        fold_params: Any = None,
        max_threads_per_model: int = 1,
        n_jobs: Optional[int] = None,
    ) -> TrainBundle:
        results = train_models_parallel(
            list(training_data),
            max_threads_per_model=max_threads_per_model,
            n_jobs=n_jobs,
            seed=seed,
            fold_params=fold_params,
        )
        return TrainBundle(
            results=results,
            training_data=list(training_data),
        )

    def load_predictor(self, model_path: Path) -> Predictor:
        return load_pickled_model(model_path)

    def suggest_hyperparameters(self, trial: Any) -> Dict[str, Any]:
        """Suggest MaxEnt hyperparameters from an Optuna trial.

        Fixed for map consistency: ``clamp=True``, ``tau=0.5``, ``transform="cloglog"``.
        Tuned: feature types, betas, hinge feature count.
        """
        available_feature_types = ["linear", "hinge", "product", "quadratic"]
        selected_feature_types: List[str] = []

        for feature_type in available_feature_types:
            include = trial.suggest_categorical(
                f"feature_type_{feature_type}",
                [True, False],
            )
            if include:
                selected_feature_types.append(feature_type)

        if "linear" not in selected_feature_types:
            selected_feature_types.append("linear")

        feature_types = sorted(selected_feature_types)

        beta_multiplier = trial.suggest_float("beta_multiplier", 1, 5.0, step=0.1)
        beta_lqp = trial.suggest_float("beta_lqp", 0.5, 3.0, step=0.1)
        beta_hinge = trial.suggest_float("beta_hinge", 1, 4.0, step=0.1)
        beta_threshold = trial.suggest_float("beta_threshold", 1, 3.0, step=0.1)
        beta_categorical = trial.suggest_float("beta_categorical", 0.5, 3.0, step=0.1)

        has_hinge = "hinge" in selected_feature_types
        if has_hinge:
            n_hinge_features = trial.suggest_int("n_hinge_features", 10, 15)
        else:
            n_hinge_features = 0

        return {
            "feature_types": feature_types,
            "beta_multiplier": beta_multiplier,
            "beta_lqp": beta_lqp,
            "beta_hinge": beta_hinge,
            "beta_threshold": beta_threshold,
            "beta_categorical": beta_categorical,
            "n_hinge_features": n_hinge_features,
            "n_threshold_features": 0,
            "clamp": True,
            "tau": 0.5,
            "transform": "cloglog",
        }

    def eval_trial(
        self,
        data: TrainingData,
        *,
        seed: int,
        max_threads_per_model: int = 1,
        n_cv_folds: int = 3,
    ) -> Tuple[float, float]:
        apply_training_seed(seed)
        model = create_maxent_pipeline(
            feature_names=data.model_features,
            maxent_n_jobs=max_threads_per_model,
            model_config=data.maxent_config,
        )

        missing_features = list(set(data.model_features) - set(data.occurrence.columns))
        if missing_features:
            raise ValueError(
                f"Missing features {missing_features} for "
                f"{data.latin_name} - {data.activity_type}"
            )

        if data.occurrence["class"].isna().any():
            raise ValueError(
                f"NaN values in class column for "
                f"{data.latin_name} - {data.activity_type}"
            )

        _cv_models, cv_scores, _ = cross_validate_maxent_model(
            model=model,
            occurrence_gdf=data.occurrence,
            n_folds=n_cv_folds,
            metric_fn=roc_auc_score,
            feature_columns=data.model_features,
            random_state=seed,
        )
        cv_mean, cv_std, _, _ = summarize_cv_scores(cv_scores)
        return cv_mean, cv_std


ADAPTERS: Dict[str, ModelAdapter] = {
    DEFAULT_TRAINER: MaxEntAdapter(),
}


def get_adapter(model_type: Optional[str]) -> ModelAdapter:
    """Return the registered adapter for ``model_type`` (default: maxent)."""
    key = normalize_trainer_id(model_type)
    adapter = ADAPTERS.get(key)
    if adapter is None:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(
            f"Unknown model.type {model_type!r}. Known adapters: {known}. "
            "Register a new ModelAdapter in sdm.models.adapters.ADAPTERS to add a method."
        )
    return adapter


def resolve_trainer(
    *,
    package: Optional[Mapping[str, Any]] = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
    model_type: Optional[str] = None,
) -> str:
    """Resolve trainer id; missing metadata defaults to maxent (pre-adapter runs)."""
    for source in (package, run_metadata):
        if source is None:
            continue
        raw = source.get("trainer")
        if raw is not None and str(raw).strip():
            return normalize_trainer_id(str(raw))
    if model_type is not None and str(model_type).strip():
        return normalize_trainer_id(str(model_type))
    return DEFAULT_TRAINER


def resolve_run_trainer(models_dir: Path) -> str:
    """Read trainer from ``run_metadata.json`` or a package; default maxent."""
    models_dir = Path(models_dir)
    meta_path = models_dir / "run_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
        if isinstance(meta, dict) and meta.get("trainer"):
            return resolve_trainer(run_metadata=meta)

    for pkg_path in sorted(models_dir.glob("*/package.json")):
        try:
            package = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(package, dict):
            return resolve_trainer(package=package)
    return DEFAULT_TRAINER


def _is_tunable(adapter: Any) -> bool:
    return callable(getattr(adapter, "suggest_hyperparameters", None)) and callable(
        getattr(adapter, "eval_trial", None)
    )


def require_tune_support(adapter: ModelAdapter) -> Tunable:
    """Return ``adapter`` as Tunable, or raise when Optuna surfaces are missing."""
    if not _is_tunable(adapter):
        raise ValueError(
            f"Hyperparameter tuning is not supported for model.type={adapter.name!r} "
            "yet. Use model.type: maxent, or implement suggest_hyperparameters / "
            "eval_trial on that adapter."
        )
    return adapter  # type: ignore[return-value]
