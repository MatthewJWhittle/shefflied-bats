"""Provenance block written into rebuild_metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from sdm.utils.rebuild_provenance import (
    DEFAULT_CEH_PATH,
    WORLDCLIM_PRODUCT,
    collect_rebuild_provenance,
    fetch_os_product_release,
    git_commit_sha,
)
from sdm.utils.step_metrics import StepMetricsRecorder


def _mock_os_session() -> MagicMock:
    """Session whose GET /products/{id} returns release versions."""
    session = MagicMock()

    def _get(url, timeout=None):  # type: ignore[no-untyped-def]
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url.endswith("/products/VectorMapDistrict"):
            response.json.return_value = {
                "id": "VectorMapDistrict",
                "name": "OS VectorMap District",
                "version": "2026-05",
            }
        elif url.endswith("/products/BoundaryLine"):
            response.json.return_value = {
                "id": "BoundaryLine",
                "name": "Boundary-Line",
                "version": "2026-10",
            }
        else:
            response.json.return_value = {}
        return response

    session.get.side_effect = _get
    return session


def test_fetch_os_product_release_reads_version() -> None:
    session = _mock_os_session()
    meta = fetch_os_product_release("VectorMapDistrict", session=session)
    assert meta["product_id"] == "VectorMapDistrict"
    assert meta["version"] == "2026-05"
    assert meta["release"] == "2026-05"
    assert meta["name"] == "OS VectorMap District"


def test_collect_rebuild_provenance_includes_git_and_sources(tmp_path: Path) -> None:
    ceh = tmp_path / "gblcm2023_10m.tif"
    ceh.write_bytes(b"fake-ceh")
    session = _mock_os_session()
    fetched = "2026-10-19T10:00:00+00:00"

    payload = collect_rebuild_provenance(
        ceh_data_path=ceh,
        session=session,
        fetched_at_utc=fetched,
    )
    assert payload["recorded_at_utc"] == fetched
    assert payload["git_commit"] == git_commit_sha() or payload["git_commit"] is None
    sources = payload["sources"]
    assert sources["worldclim"]["product"] == WORLDCLIM_PRODUCT
    assert sources["worldclim"]["version"] == "2.1"
    assert sources["os_vector_map_district"]["product_id"] == "VectorMapDistrict"
    assert sources["os_vector_map_district"]["version"] == "2026-05"
    assert sources["os_vector_map_district"]["release"] == "2026-05"
    assert sources["os_boundary_line"]["product_id"] == "BoundaryLine"
    assert sources["os_boundary_line"]["version"] == "2026-10"
    assert sources["ea_lidar_dtm"]["dataset_name"]
    assert sources["ea_lidar_dtm"]["fetched_at_utc"] == fetched
    assert sources["ea_lidar_dsm"]["fetched_at_utc"] == fetched
    assert sources["ea_vom"]["dataset_name"]
    assert sources["ea_vom"]["fetched_at_utc"] == fetched
    assert sources["ceh_land_cover"]["present"] is True
    assert sources["ceh_land_cover"]["mtime_utc"]
    assert sources["ceh_land_cover"]["size_bytes"] == 8
    assert sources["alan_viirs"]["band_name"] == "alan_viirs_radiance"
    assert sources["alan_viirs"]["units"] == "nW/cm^2/sr"
    assert sources["alan_viirs"]["resampling"] == "bilinear"
    # Public-safe: no study-area labels in provenance.
    blob = json.dumps(payload)
    assert "Skipton" not in blob
    assert "Yorkshire" not in blob


def test_recorder_write_embeds_provenance(tmp_path: Path) -> None:
    recorder = StepMetricsRecorder(sample_interval_seconds=0.02)
    recorder.run("stub", lambda: None)
    provenance = collect_rebuild_provenance(
        ceh_data_path=DEFAULT_CEH_PATH,
        session=_mock_os_session(),
    )
    paths = recorder.write(tmp_path, extra={"provenance": provenance})
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["provenance"]["sources"]["worldclim"]["version"] == "2.1"
    assert payload["provenance"]["sources"]["os_boundary_line"]["version"] == "2026-10"
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
    monkeypatch.setattr(mod, "generate_alan_data", _ok)
    monkeypatch.setattr(mod, "merge_ev_layers", _ok)
    monkeypatch.setattr(
        mod,
        "build_ev_dataset_inputs",
        lambda *a, **k: [],
    )
    stub_settings = {
        "layer_clipping": {"enabled": True, "layers": {}, "applied": {}},
    }
    monkeypatch.setattr(
        mod,
        "build_model_ev_stack",
        lambda *_a, **_k: (out / "evs-to-model.tif", stub_settings),
    )
    monkeypatch.setattr(
        mod, "build_settings_from_config", lambda *_a, **_k: stub_settings
    )
    monkeypatch.setattr(mod, "load_config", lambda: {})

    def _collect(**kwargs):
        return collect_rebuild_provenance(
            session=_mock_os_session(),
            **kwargs,
        )

    with patch.object(mod, "collect_rebuild_provenance", side_effect=_collect):
        mod.rebuild_environmental_layers(
            out,
            boundary_path=boundary,
            resolution=100,
            alan_source_path=tmp_path / "no-viirs.tif",
        )

    metrics = json.loads((out / "rebuild_metrics.json").read_text(encoding="utf-8"))
    assert "provenance" in metrics
    assert metrics["provenance"]["sources"]["ceh_land_cover"]["product"]
    assert "git_commit" in metrics["provenance"]
    assert metrics["provenance"]["sources"]["os_vector_map_district"]["version"]
    assert metrics["provenance"]["sources"]["ea_lidar_dtm"]["fetched_at_utc"]
    assert "build_settings" in metrics


def test_gitignore_covers_default_rebuild_output() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/evs-rebuild/" in text
