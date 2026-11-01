"""Associate predictor docs with rasters via attrs and a YAML sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import yaml

from sdm.predictors.catalog import default_docs_root, load_layer
from sdm.predictors.schema import LayerDoc

# Compact keys written onto xarray / GeoTIFF metadata so docs travel with bands.
DOC_ATTR_KEYS = (
    "sdm_layer_id",
    "sdm_layer_doc",
    "sdm_layer_kind",
    "sdm_layer_family",
)


def layer_raster_attrs(
    doc: LayerDoc,
    *,
    docs_root: Optional[Path] = None,
    repo_relative: bool = True,
) -> dict[str, str]:
    """Attrs to stamp on a band so agents/tools can resolve its doc."""
    root = docs_root if docs_root is not None else default_docs_root()
    md_name = doc.markdown or f"{doc.id}.md"
    doc_path = root / "layers" / md_name
    if repo_relative:
        try:
            doc_rel = str(doc_path.resolve().relative_to(root.parent.parent.resolve()))
        except ValueError:
            doc_rel = str(doc_path)
    else:
        doc_rel = str(doc_path)
    return {
        "sdm_layer_id": doc.id,
        "sdm_layer_doc": doc_rel,
        "sdm_layer_kind": doc.kind.value,
        "sdm_layer_family": doc.family,
    }


def write_predictor_sidecar(
    raster_path: Path | str,
    band_ids: Sequence[str],
    *,
    docs_root: Optional[Path] = None,
    only_documented: bool = True,
) -> Path:
    """Write ``{raster}.predictors.yml`` mapping bands → doc paths / lineage.

    Undocumented bands are listed with ``documented: false`` unless
    ``only_documented`` is true, in which case they are omitted.
    """
    raster_path = Path(raster_path)
    root = docs_root if docs_root is not None else default_docs_root()
    bands: list[dict[str, Any]] = []
    for band_id in band_ids:
        try:
            doc = load_layer(band_id, docs_root=root)
        except FileNotFoundError:
            if only_documented:
                continue
            bands.append({"id": band_id, "documented": False})
            continue
        md_name = doc.markdown or f"{doc.id}.md"
        bands.append(
            {
                "id": doc.id,
                "documented": True,
                "kind": doc.kind.value,
                "family": doc.family,
                "derived_from": list(doc.derived_from),
                "doc_yaml": f"docs/predictors/layers/{doc.id}.yml",
                "doc_markdown": f"docs/predictors/layers/{md_name}",
            }
        )
    payload = {
        "schema_version": 1,
        "raster": raster_path.name,
        "bands": bands,
    }
    sidecar = Path(f"{raster_path}.predictors.yml")
    with sidecar.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return sidecar


def read_predictor_sidecar(raster_path: Path | str) -> dict[str, Any]:
    """Load the predictor sidecar next to a raster."""
    sidecar = Path(f"{Path(raster_path)}.predictors.yml")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Predictor sidecar not found: {sidecar}")
    with sidecar.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def attach_predictor_docs(
    dataset: Any,
    *,
    docs_root: Optional[Path] = None,
    band_names: Optional[Sequence[str]] = None,
) -> list[str]:
    """Stamp known layer attrs onto an xarray Dataset's data variables.

    Returns the list of band ids that received documentation attrs. Missing
    docs are skipped (first-batch coverage is partial by design).
    """
    root = docs_root if docs_root is not None else default_docs_root()
    names = list(band_names) if band_names is not None else list(dataset.data_vars)
    stamped: list[str] = []
    for name in names:
        if name not in dataset.data_vars:
            continue
        try:
            doc = load_layer(str(name), docs_root=root)
        except FileNotFoundError:
            continue
        attrs: MutableMapping[str, Any] = dataset[name].attrs
        attrs.update(layer_raster_attrs(doc, docs_root=root))
        stamped.append(doc.id)
    return stamped


def documented_band_map(
    band_ids: Sequence[str],
    *,
    docs_root: Optional[Path] = None,
) -> Mapping[str, LayerDoc]:
    """Return ``{band_id: LayerDoc}`` for bands that have docs."""
    root = docs_root if docs_root is not None else default_docs_root()
    found: dict[str, LayerDoc] = {}
    for band_id in band_ids:
        try:
            found[band_id] = load_layer(band_id, docs_root=root)
        except FileNotFoundError:
            continue
    return found
