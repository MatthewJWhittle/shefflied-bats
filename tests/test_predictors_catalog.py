"""Predictor-layer documentation catalog and raster association."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from sdm.predictors import (
    LayerDoc,
    LayerKind,
    attach_predictor_docs,
    default_docs_root,
    layer_raster_attrs,
    load_catalog,
    load_layer,
    load_source,
    read_predictor_sidecar,
    resolve_layer_markdown,
    resolve_source_markdown,
    write_predictor_sidecar,
)


def test_catalog_loads_and_matches_layer_files() -> None:
    """Every catalog id has YAML + Markdown that validate."""
    root = default_docs_root()
    catalog = load_catalog(docs_root=root)
    assert catalog.schema_version == 1
    assert len(catalog.layers) >= 1
    for layer_id in catalog.layers:
        doc = load_layer(layer_id, docs_root=root)
        assert doc.id == layer_id
        md = resolve_layer_markdown(doc, docs_root=root)
        assert md.is_file()
        assert md.read_text(encoding="utf-8").strip()


def test_catalog_loads_and_matches_source_files() -> None:
    """Every catalog source id has YAML + Markdown that validate."""
    root = default_docs_root()
    catalog = load_catalog(docs_root=root)
    assert len(catalog.sources) >= 1
    for source_id in catalog.sources:
        doc = load_source(source_id, docs_root=root)
        assert doc.id == source_id
        assert doc.licence
        assert doc.provider
        assert doc.product
        md = resolve_source_markdown(doc, docs_root=root)
        assert md.is_file()
        assert md.read_text(encoding="utf-8").strip()


def test_layer_source_ids_resolve() -> None:
    """Layer source_id values must be catalogued sources."""
    root = default_docs_root()
    catalog = load_catalog(docs_root=root)
    known = set(catalog.sources)
    for layer_id in catalog.layers:
        doc = load_layer(layer_id, docs_root=root)
        if doc.source_id is None:
            continue
        assert doc.source_id in known, f"{layer_id} -> missing source {doc.source_id}"
        load_source(doc.source_id, docs_root=root)


def test_all_model_feature_layers_are_documented() -> None:
    """Every variable in data/sdm_config/*/variables_config.yml has a layer doc."""
    needed: set[str] = set()
    for path in Path("data/sdm_config").glob("*/variables_config.yml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        needed.update(payload.get("variables") or [])
    documented = set(load_catalog().layers)
    missing = sorted(needed - documented)
    assert not missing, f"Undocumented model layers: {missing}"


def test_derived_layers_reference_existing_parents() -> None:
    root = default_docs_root()
    catalog = load_catalog(docs_root=root)
    for layer_id in catalog.layers:
        doc = load_layer(layer_id, docs_root=root)
        if doc.kind is LayerKind.DERIVED:
            assert doc.derived_from
            for parent in doc.derived_from:
                parent_doc = load_layer(parent, docs_root=root)
                assert parent_doc.id == parent


def test_source_layer_rejects_derived_from() -> None:
    with pytest.raises(ValidationError):
        LayerDoc(
            id="example_source",
            title="Example",
            kind=LayerKind.SOURCE,
            family="terrain",
            summary="x",
            derived_from=["terrain_dtm"],
        )


def test_vom_docs_include_verified_source_limits() -> None:
    """Ecology / EA VOM limits must appear on vegetation height docs."""
    mean = load_layer("vom_vegetation_height_mean")
    text = " ".join(mean.caveats).lower()
    assert "2.5" in text
    assert "canopy" in text
    assert "leaf-off" in text or "leaf off" in text
    md = resolve_layer_markdown(mean).read_text(encoding="utf-8").lower()
    assert "2.5" in md
    assert "canopy" in md


def test_coast_distance_documented_as_not_model_feature() -> None:
    doc = load_layer("bgs_coast_distance_to_coast")
    blob = " ".join([doc.summary, *doc.caveats, *doc.processing]).lower()
    assert "include=0" in blob or "not used" in blob
    assert doc.source_id == "os_boundary_line"
    md = resolve_layer_markdown(doc).read_text(encoding="utf-8").lower()
    assert "not used" in md or "include=0" in md


def test_layer_docs_avoid_git_policy_wording() -> None:
    """Layer caveats/notes document sources, not repo commit policy."""
    banned = (
        "must not enter git",
        "do not commit",
        "stays out of git",
        "not in git",
        "gitignored",
    )
    root = default_docs_root()
    for layer_id in load_catalog(docs_root=root).layers:
        doc = load_layer(layer_id, docs_root=root)
        blob = " ".join([*doc.processing, *doc.caveats]).lower()
        for phrase in banned:
            assert phrase not in blob, f"{layer_id} still has {phrase!r}"


def test_attach_and_sidecar_round_trip(tmp_path: Path) -> None:
    """Documented bands get attrs; sidecar lists them next to the raster path."""
    ds = xr.Dataset(
        {
            "terrain_dtm": (("y", "x"), np.zeros((2, 2), dtype=np.float32)),
            "undocumented_band": (("y", "x"), np.ones((2, 2), dtype=np.float32)),
        },
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    stamped = attach_predictor_docs(ds)
    assert "terrain_dtm" in stamped
    assert "undocumented_band" not in stamped
    attrs = ds["terrain_dtm"].attrs
    for key in ("sdm_layer_id", "sdm_layer_doc", "sdm_layer_kind", "sdm_layer_family"):
        assert key in attrs
    assert attrs["sdm_layer_id"] == "terrain_dtm"

    raster = tmp_path / "evs-to-model.tif"
    raster.write_bytes(b"fake")
    sidecar = write_predictor_sidecar(
        raster,
        ["terrain_dtm", "undocumented_band"],
        only_documented=True,
    )
    assert sidecar == Path(f"{raster}.predictors.yml")
    payload = read_predictor_sidecar(raster)
    ids = [row["id"] for row in payload["bands"]]
    assert ids == ["terrain_dtm"]
    assert payload["bands"][0]["derived_from"] == []


def test_layer_raster_attrs_point_at_repo_doc() -> None:
    doc = load_layer("alan_viirs_radiance")
    attrs = layer_raster_attrs(doc)
    assert attrs["sdm_layer_id"] == "alan_viirs_radiance"
    assert "docs/predictors/layers/" in attrs["sdm_layer_doc"]
