"""Helpers to read configured feature columns from SDM sklearn pipelines."""

from __future__ import annotations

from typing import Any, List

from sklearn.compose import ColumnTransformer

from sdm.models.core.feature_subsetter import FeatureSubsetter


def selection_step_feature_names(step: Any) -> List[str]:
    """Feature column list configured on a single selection transformer step."""
    if isinstance(step, FeatureSubsetter):
        return list(step.feature_names)

    if isinstance(step, ColumnTransformer):
        names: List[str] = []
        for name, _trans, cols in step.transformers:
            if name == "remainder":
                continue
            if isinstance(cols, str):
                raise ValueError(
                    f"Unexpected ColumnTransformer column spec {cols!r} on transformer {name!r}"
                )
            names.extend(list(cols))
        return names

    raise TypeError(
        f"Unsupported feature_selection type: {type(step).__name__}. "
        "Expected FeatureSubsetter or ColumnTransformer."
    )


def pipeline_selected_feature_names(model: Any) -> List[str]:
    """Return ordered feature names from the ``feature_selection`` pipeline step."""
    try:
        step = model["feature_selection"]
    except (KeyError, TypeError) as e:
        raise ValueError("Model has no feature_selection step") from e
    return selection_step_feature_names(step)
