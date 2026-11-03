"""MaxEnt train helpers (parallel fit + CV summary).

Kept outside the Typer command module so adapters can import at top level
without a command ↔ model cycle. Behaviour matches the previous helpers that
lived in ``train_sdm_models``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sdm.models.maxent.folds import (
    FoldMethod,
    FoldParams,
    buffered_loo_presence_warning,
    presence_counts_per_fold,
    resolve_fold_params,
)
from sdm.models.maxent.maxent_model import (
    ActivityType,
    create_maxent_pipeline,
    evaluate_and_train_maxent_model,
)
from sdm.types import CvConfig, TrainingData, TrainingResults
from sdm.utils.rng import apply_training_seed

logger = logging.getLogger(__name__)


def summarize_cv_scores(
    cv_scores: Optional[np.ndarray],
) -> tuple[float, float, int, int]:
    """Return (mean, std, n_valid, n_total) for a CV score array, safely handling NaNs."""
    if cv_scores is None or len(cv_scores) == 0:
        return np.nan, np.nan, 0, 0

    valid_scores = cv_scores[~np.isnan(cv_scores)]
    n_total = len(cv_scores)
    n_valid = len(valid_scores)

    if n_valid == 0:
        return np.nan, np.nan, 0, n_total

    return float(valid_scores.mean()), float(valid_scores.std()), n_valid, n_total


# Private alias kept for call sites / tests that still import the underscored name.
_summarize_cv_scores = summarize_cv_scores


def train_single_model(
    data: TrainingData,
    max_threads_per_model: int,
    n_cv_folds: int = 3,
    min_presence: int = 15,
    *,
    seed: int,
    fold_params: Optional[FoldParams] = None,
) -> TrainingResults:
    """Train a single MaxEnt model for a given set of training data."""
    try:
        apply_training_seed(seed)
        activity_type = ActivityType(data.activity_type)
        latin_name = data.latin_name
        logger.debug(f"Training model for {latin_name} ({activity_type.value})...")

        resolved_folds = fold_params or resolve_fold_params(CvConfig(), model_seed=seed)
        model_config = data.maxent_config
        model_features = data.model_features

        if len(model_features) == 0:
            raise ValueError(
                f"model_features list is empty for {latin_name} - {activity_type}. "
                f"At least one feature must be provided."
            )

        missing_features = list(set(model_features) - set(data.occurrence.columns))
        if missing_features:
            raise ValueError(
                f"Missing features {missing_features} in occurrence data for "
                f"{latin_name} - {activity_type}. "
                f"Available columns: {list(data.occurrence.columns)}"
            )

        model = create_maxent_pipeline(
            feature_names=model_features,
            maxent_n_jobs=max_threads_per_model,
            model_config=model_config,
        )
        logger.info(f"Model features: {model_features}")
        n_presence = len(data.occurrence[data.occurrence["class"] == 1])
        if resolved_folds.method is FoldMethod.BUFFERED_LOO:
            loo_warning = buffered_loo_presence_warning(
                n_presence,
                resolved_folds.buffered_loo_warn_above,
                latin_name=latin_name,
                activity_type=activity_type.value,
            )
            if loo_warning:
                logger.warning(loo_warning)
        if resolved_folds.method is FoldMethod.KMEANS:
            effective_n_folds = (
                resolved_folds.n_folds if fold_params is not None else n_cv_folds
            )
            if effective_n_folds > 0 and (n_presence // effective_n_folds) < min_presence:
                effective_n_folds = 2
                logger.info(
                    "Using 2 folds for %s - %s because n_presence < %s",
                    latin_name,
                    activity_type.value,
                    min_presence,
                )
        elif resolved_folds.method is FoldMethod.BLOCKS:
            effective_n_folds = resolved_folds.n_folds
        elif resolved_folds.method is FoldMethod.BUFFERED_LOO:
            effective_n_folds = int(n_presence)
        else:
            raise ValueError(f"unknown fold method: {resolved_folds.method!r}")

        effective_params = FoldParams(
            method=resolved_folds.method,
            n_folds=effective_n_folds,
            random_state=resolved_folds.random_state,
            block_size_m=resolved_folds.block_size_m,
            block_n_tries=resolved_folds.block_n_tries,
            buffer_m=resolved_folds.buffer_m,
            buffered_loo_warn_above=resolved_folds.buffered_loo_warn_above,
        )
        fold_presence = presence_counts_per_fold(data.occurrence, effective_params)

        final_model, cv_models, cv_scores, validation_scores = evaluate_and_train_maxent_model(
            model=model,
            occurrence_gdf=data.occurrence,
            n_cv_folds=effective_n_folds,
            metric_fn=roc_auc_score,
            feature_columns=model_features,
            collect_validation_scores=True,
            random_state=effective_params.random_state,
            fold_method=effective_params.method,
            block_size_m=effective_params.block_size_m,
            block_n_tries=effective_params.block_n_tries,
            buffer_m=effective_params.buffer_m,
        )

        if final_model is None:
            raise ValueError("Model training failed - final_model is None")

        cv_mean, cv_std, n_valid, n_total = summarize_cv_scores(cv_scores)
        if n_valid > 0:
            logger.debug(
                "✓ %s - %s: CV AUC = %.4f ± %.4f (%d/%d folds valid)",
                latin_name,
                activity_type.value,
                cv_mean,
                cv_std,
                n_valid,
                n_total,
            )
        else:
            logger.debug(
                "✗ %s - %s: No valid CV scores", latin_name, activity_type.value
            )
        logger.info(
            "%s - %s: fold_method=%s scored %d/%d folds; presence per fold=%s",
            latin_name,
            activity_type.value,
            effective_params.method.value,
            n_valid,
            n_total,
            fold_presence,
        )

        return TrainingResults(
            latin_name=latin_name,
            activity_type=activity_type.value,
            final_model=final_model,
            cv_models=cv_models if cv_models is not None else None,
            cv_scores=cv_scores,
            validation_scores=validation_scores,
            cv_n_folds=effective_n_folds if effective_n_folds else n_total,
            fold_method=effective_params.method.value,
            fold_params=effective_params.to_metadata(),
            fold_presence_counts=fold_presence,
            success=True,
            error=None,
        )
    except Exception as e:
        logger.error(
            f"Error training model for {data.latin_name} - {data.activity_type}: {e}"
        )
        return TrainingResults(
            latin_name=data.latin_name,
            activity_type=data.activity_type,
            final_model=None,
            cv_models=None,
            cv_scores=None,
            success=False,
            error=str(e),
        )


def train_models_parallel(
    training_data: List[TrainingData],
    max_threads_per_model: int = 1,
    n_jobs: Optional[int] = None,
    *,
    seed: int,
    fold_params: Optional[FoldParams] = None,
) -> List[TrainingResults]:
    """Train MaxEnt models in parallel for each set of training data.

    Args:
        training_data: List of TrainingData objects, one per species-activity combination.
            Each TrainingData must include maxent_config and model_features.
        max_threads_per_model: Maximum threads per model (default 1).
        n_jobs: Number of parallel jobs (None = auto).

    Returns:
        List of successful TrainingResults.
    """
    if n_jobs is None:
        total_cpus = os.cpu_count()
        if total_cpus is None:
            total_cpus = 1.0
        n_jobs = max(1, int(total_cpus * 0.8) // max_threads_per_model)

    logger.debug(
        f"Training with {n_jobs} parallel jobs, {max_threads_per_model} threads per model"
    )

    results: List[TrainingResults] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for data in training_data:
            futures.append(
                executor.submit(
                    train_single_model,
                    data,
                    max_threads_per_model,
                    seed=seed,
                    fold_params=fold_params,
                )
            )

        for future in tqdm(futures, total=len(futures), desc="Training models"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel training: {e}")
                results.append(
                    TrainingResults(
                        latin_name="unknown",
                        activity_type="unknown",
                        final_model=None,
                        cv_models=None,
                        cv_scores=None,
                        success=False,
                        error=str(e),
                    )
                )

    successful_results = [r for r in results if r.success]

    failed_models = [
        f"{r.latin_name} - {r.activity_type}: {r.error}"
        for r in results
        if not r.success
    ]
    if failed_models:
        logger.warning("Failed models:")
        for failure in failed_models:
            logger.warning(f"  {failure}")

    logger.info(
        f"Successfully trained {len(successful_results)} models out of "
        f"{len(training_data)} attempts"
    )

    if successful_results:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        for result in successful_results:
            cv_mean, cv_std, n_valid, n_total = summarize_cv_scores(result.cv_scores)
            if n_valid > 0:
                logger.info(
                    "  %s - %s: AUC = %.4f ± %.4f (%d/%d folds valid)",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                    cv_mean,
                    cv_std,
                    n_valid,
                    n_total,
                )
            else:
                logger.info(
                    "  %s - %s: No valid scores",
                    f"{result.latin_name:30s}",
                    f"{result.activity_type:15s}",
                )
        logger.info("=" * 80 + "\n")

    return successful_results
