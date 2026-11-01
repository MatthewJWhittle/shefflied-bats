"""Predictor-layer documentation catalog and raster association helpers."""

from sdm.predictors.associate import (
    DOC_ATTR_KEYS,
    attach_predictor_docs,
    layer_raster_attrs,
    read_predictor_sidecar,
    write_predictor_sidecar,
)
from sdm.predictors.catalog import (
    default_docs_root,
    iter_layer_ids,
    iter_source_ids,
    load_catalog,
    load_layer,
    load_source,
    resolve_layer_markdown,
    resolve_source_markdown,
)
from sdm.predictors.schema import (
    LayerDoc,
    LayerKind,
    PredictorCatalog,
    SourceDoc,
    SourceInfo,
)

__all__ = [
    "DOC_ATTR_KEYS",
    "LayerDoc",
    "LayerKind",
    "PredictorCatalog",
    "SourceDoc",
    "SourceInfo",
    "attach_predictor_docs",
    "default_docs_root",
    "iter_layer_ids",
    "iter_source_ids",
    "layer_raster_attrs",
    "load_catalog",
    "load_layer",
    "load_source",
    "read_predictor_sidecar",
    "resolve_layer_markdown",
    "resolve_source_markdown",
    "write_predictor_sidecar",
]
