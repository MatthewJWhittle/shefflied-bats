"""Provenance block written into rebuild_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from sdm.utils.rebuild_provenance import (
    DEFAULT_CEH_PATH,
    WORLDCLIM_PRODUCT,
    collect_rebuild_provenance,
    git_commit_sha,
)
from sdm.utils.step_metrics import StepMetricsRecorder


def test_collect_rebuild_provenance_includes_git_and_sources(tmp_path: Path) -> None:
    ceh = tmp_path / "gblcm2023_10m.tif"
    ceh.write_bytes(b"fake-ceh")

    payload = collect_rebuild_provenance(ceh_data_path=ceh)
    assert "recorded_at_utc" in payload
    assert payload["git_commit"] == git_commit_sha() or payload["git_commit"] is None
    sources = payload["sources"]
    assert sources["worldclim"]["product"] == WORLDCLIM_PRODUCT
    assert sources["worldclim"]["version"] == "2.1"
    assert sources["os_vector_map_district"]["product_id"] == "VectorMapDistrict"
    assert sources["os_boundary_line"]["product_id"] == "BoundaryLine"
    assert sources["ceh_land_cover"]["present"] is True
    assert sources["ceh_land_cover"]["mtime_utc"]
    assert sources["ceh_land_cover"]["size_bytes"] == 8
    # Public-safe: no study-area labels in provenance.
    blob = json.dumps(payload)
    assert "Skipton" not in blob
    assert "Yorkshire" not in blob


def test_recorder_write_embeds_provenance(tmp_path: Path) -> None:
    recorder = StepMetricsRecorder(sample_interval_seconds=0.02)
    recorder.run("stub", lambda: None)
    provenance = collect_rebuild_provenance(ceh_data_path=DEFAULT_CEH_PATH)
    paths = recorder.write(tmp_path, extra={"provenance": provenance})
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["provenance"]["sources"]["worldclim"]["version"] == "2.1"
    assert "steps" in payload


def test_rebuild_metrics_include_provenance_on_orchestrator(
    tmp_path: Path, monkeypatch
) -> None:
    """``rebuild_environmental_layers`` writes provenance even when steps are stubbed."""
    from sdm.commands.data_preparation import rebuild_ev_layers as mod

    boundary = tmp_path / "boundary.geojson"
    boundary.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    out = tmp_path / "evs-rebuild"

    def _ok(*_args, **_kwargs):
        return out

    monkeypatch.setattr(mod, "validate_ev_output_dir", lambda *a, **k: Path(out))
    monkeypatch.setattr(mod, "generate_terrain_data", _ok)
    monkeypatch.setattr(mod, "generate_terrain_stats", _ok)
    monkeypatch.setattr(mod, "generate_climate_data", _ok)
    monkeypatch.setattr(mod, "generate_ceh_lc_data", _ok)
    monkeypatch.setattr(mod, "generate_vom_data", _ok)
    monkeypatch.setattr(mod, "generate_coastal_distance", _ok)
    monkeypatch.setattr(mod, "process_os_data", _ok)
    monkeypatch.setattr(mod, "merge_ev_layers", _ok)
    monkeypatch.setattr(
        mod,
        "build_ev_dataset_inputs",
        lambda *a, **k: [],
    )

    with patch.object(mod, "collect_rebuild_provenance", wraps=collect_rebuild_provenance):
        mod.rebuild_environmental_layers(
            out,
            boundary_path=boundary,
            resolution=100,
        )

    metrics = json.loads((out / "rebuild_metrics.json").read_text(encoding="utf-8"))
    assert "provenance" in metrics
    assert metrics["provenance"]["sources"]["ceh_land_cover"]["product"]
    assert "git_commit" in metrics["provenance"]


def test_gitignore_covers_default_rebuild_output() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/evs-rebuild/" in text
