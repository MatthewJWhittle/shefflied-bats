"""Load and validate predictor-layer docs from ``docs/predictors/``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from sdm.predictors.schema import LayerDoc, PredictorCatalog

_REPO_ROOT_MARKERS = ("pyproject.toml", "docs", "sdm")


def _find_repo_root(start: Optional[Path] = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _REPO_ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root (expected pyproject.toml, docs/, sdm/)."
    )


def default_docs_root(*, repo_root: Optional[Path] = None) -> Path:
    """Return ``docs/predictors`` under the repository root."""
    root = repo_root if repo_root is not None else _find_repo_root()
    return root / "docs" / "predictors"


def load_catalog(*, docs_root: Optional[Path] = None) -> PredictorCatalog:
    """Load ``catalog.yml`` and validate it."""
    root = docs_root if docs_root is not None else default_docs_root()
    path = root / "catalog.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Predictor catalog not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return PredictorCatalog.model_validate(payload)


def load_layer(
    layer_id: str,
    *,
    docs_root: Optional[Path] = None,
) -> LayerDoc:
    """Load and validate one layer YAML by id."""
    root = docs_root if docs_root is not None else default_docs_root()
    path = root / "layers" / f"{layer_id}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Predictor layer doc not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    doc = LayerDoc.model_validate(payload)
    if doc.id != layer_id:
        raise ValueError(
            f"Layer YAML id {doc.id!r} does not match filename stem {layer_id!r}"
        )
    return doc


def iter_layer_ids(*, docs_root: Optional[Path] = None) -> Iterable[str]:
    catalog = load_catalog(docs_root=docs_root)
    return list(catalog.layers)


def resolve_layer_markdown(
    doc: LayerDoc,
    *,
    docs_root: Optional[Path] = None,
) -> Path:
    """Return the human Markdown path for a layer doc."""
    root = docs_root if docs_root is not None else default_docs_root()
    name = doc.markdown or f"{doc.id}.md"
    path = root / "layers" / name
    if not path.is_file():
        raise FileNotFoundError(f"Predictor layer markdown not found: {path}")
    return path
