"""Versioned prediction publishing: local/fake FS only (no real bucket)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate
from types import SimpleNamespace

import sdm.commands.modelling.predict_sdm_models as pred_mod
from sdm.models.core.prediction import (
    save_prediction_raster,
    write_threshold_binary_raster,
)
from sdm.models.core.threshold import POOLED_OOF_PERCENTILE_RULE
from sdm.utils.prediction_publish import (
    build_version_folder_name,
    publish_model_prediction_version,
    publish_predictions_from_output_dir,
    resolve_prediction_publish_root,
    stable_config_hash,
    threshold_raster_filename,
    version_folder_exists,
)


def _write_tiny_prediction(path: Path, *, values: np.ndarray | None = None) -> Path:
    h, w = 8, 6
    data = (
        values
        if values is not None
        else np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    )
    transform = from_bounds(400_000, 500_000, 400_000 + w * 50, 500_000 + h * 50, w, h)
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:27700",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_stable_config_hash_is_deterministic() -> None:
    payload = {"model_id": "demo_model", "seed": 42, "nested": {"a": 1}}
    assert stable_config_hash(payload) == stable_config_hash(payload)
    assert len(stable_config_hash(payload)) == 8
    assert stable_config_hash(payload) != stable_config_hash({**payload, "seed": 43})


def test_build_version_folder_name_format() -> None:
    name = build_version_folder_name(
        created_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
        git_sha="abcdef0123456789",
        config_hash="deadbeef",
    )
    assert name == "20261025_abcdef0_cfgdeadbeef"


def test_threshold_raster_filename_uses_rule() -> None:
    assert (
        threshold_raster_filename(POOLED_OOF_PERCENTILE_RULE)
        == "threshold_pooled_oof_percentile.tif"
    )


def test_write_threshold_binary_raster_known_answer(tmp_path: Path) -> None:
    data = np.array(
        [
            [0.05, 0.20, -9999.0],
            [0.10, 0.50, 0.09],
        ],
        dtype=np.float32,
    )
    pred = _write_tiny_prediction(
        tmp_path / "pred.tif",
        values=np.pad(data, ((0, 6), (0, 3)), constant_values=-9999.0),
    )
    # Rewrite with exact 2x3 for clarity of assertion window
    transform = from_bounds(0, 0, 3, 2, 3, 2)
    with rasterio.open(
        pred,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    out = tmp_path / "threshold_pooled_oof_percentile.tif"
    write_threshold_binary_raster(pred, out, cutoff=0.10, as_cog=True)

    ok, _, errs = cog_validate(out, quiet=True)
    assert ok, errs
    with rasterio.open(out) as ds:
        binary = ds.read(1)
        assert ds.nodata == 255
        assert binary.dtype == np.uint8
        # >= 0.10 suitable (1); < 0.10 unsuitable (0); nodata 255
        assert binary[0, 0] == 0
        assert binary[0, 1] == 1
        assert binary[0, 2] == 255
        assert binary[1, 0] == 1
        assert binary[1, 1] == 1
        assert binary[1, 2] == 0


def test_save_prediction_raster_optional_cog(tmp_path: Path) -> None:
    data = np.ones((4, 4), dtype=np.float32) * 0.4
    transform = from_bounds(0, 0, 4, 4, 4, 4)
    path = tmp_path / "prediction.tif"
    save_prediction_raster(
        data,
        path,
        transform=transform,
        crs="EPSG:27700",
        as_cog=True,
    )
    ok, _, errs = cog_validate(path, quiet=True)
    assert ok, errs


def test_publish_model_prediction_version_local_tree(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_id = "demo_species_roost"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    package = {
        "schema_version": 1,
        "model_id": model_id,
        "feature_names": ["terrain_dtm", "ceh_broadleaved_500m"],
        "metrics": {"mean_cv_auc": 0.81, "n_presence": 40},
        "model_limited": False,
    }
    (pkg / "package.json").write_text(json.dumps(package), encoding="utf-8")
    (pkg / "thresholds.json").write_text(
        json.dumps(
            {
                "rule": POOLED_OOF_PERCENTILE_RULE,
                "model_id": model_id,
                "threshold": 0.25,
                "percentile": 10.0,
                "model_limited": False,
                "no_cutoff_reason": None,
            }
        ),
        encoding="utf-8",
    )

    prediction = _write_tiny_prediction(tmp_path / "local" / f"prediction_{model_id}.tif")
    publish_root = tmp_path / "predictions"
    created = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)

    uri = publish_model_prediction_version(
        model_id=model_id,
        prediction_raster=prediction,
        publish_root=str(publish_root),
        models_dir=models_dir,
        config={"prediction_crs": "EPSG:27700", "write_cog": True},
        label="unit-test",
        created_at=created,
        git_sha="abc1234ffff",
        git_dirty=False,
        write_cog=True,
    )

    version_dir = Path(uri)
    assert version_dir.is_dir()
    assert (version_dir / "prediction.tif").is_file()
    assert (version_dir / "threshold_pooled_oof_percentile.tif").is_file()
    run = json.loads((version_dir / "run.json").read_text(encoding="utf-8"))
    assert run["model_id"] == model_id
    assert run["git_sha"] == "abc1234ffff"
    assert run["git_dirty"] is False
    assert run["label"] == "unit-test"
    assert run["cutoff"]["rule"] == POOLED_OOF_PERCENTILE_RULE
    assert run["cutoff"]["value"] == pytest.approx(0.25)
    assert run["cv_metrics"]["mean_cv_auc"] == pytest.approx(0.81)
    assert run["config_hash"]
    assert "terrain_dtm" in run["environmental_layer_versions"]
    assert run["environmental_layer_versions"]["terrain_dtm"] is None
    assert version_dir.name.startswith("20261025_abc1234_cfg")

    ok_p, _, e_p = cog_validate(version_dir / "prediction.tif", quiet=True)
    assert ok_p, e_p
    ok_t, _, e_t = cog_validate(
        version_dir / "threshold_pooled_oof_percentile.tif", quiet=True
    )
    assert ok_t, e_t

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_model_prediction_version(
            model_id=model_id,
            prediction_raster=prediction,
            publish_root=str(publish_root),
            models_dir=models_dir,
            config={"prediction_crs": "EPSG:27700", "write_cog": True},
            label="unit-test",
            created_at=created,
            git_sha="abc1234ffff",
            git_dirty=False,
            write_cog=True,
        )


def test_publish_skips_threshold_when_model_limited(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_id = "limited_model_in_flight"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "feature_names": ["terrain_dtm"],
                "metrics": {},
                "model_limited": True,
            }
        ),
        encoding="utf-8",
    )
    (pkg / "thresholds.json").write_text(
        json.dumps(
            {
                "rule": POOLED_OOF_PERCENTILE_RULE,
                "threshold": None,
                "model_limited": True,
                "no_cutoff_reason": "model_limited",
            }
        ),
        encoding="utf-8",
    )
    prediction = _write_tiny_prediction(tmp_path / f"prediction_{model_id}.tif")
    uri = publish_model_prediction_version(
        model_id=model_id,
        prediction_raster=prediction,
        publish_root=str(tmp_path / "predictions"),
        models_dir=models_dir,
        config={},
        created_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
        git_sha="1234567",
        git_dirty=False,
    )
    version_dir = Path(uri)
    assert (version_dir / "prediction.tif").is_file()
    assert (version_dir / "run.json").is_file()
    assert list(version_dir.glob("threshold_*.tif")) == []


def test_publish_predictions_from_output_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    out_dir = tmp_path / "sdm_predictions"
    publish_root = tmp_path / "bucket" / "predictions"
    model_id = "alpha_roost"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(
        json.dumps({"model_id": model_id, "feature_names": [], "metrics": {}}),
        encoding="utf-8",
    )
    (pkg / "thresholds.json").write_text(
        json.dumps(
            {
                "rule": POOLED_OOF_PERCENTILE_RULE,
                "threshold": 0.4,
                "model_limited": False,
            }
        ),
        encoding="utf-8",
    )
    _write_tiny_prediction(out_dir / f"prediction_{model_id}.tif")

    published = publish_predictions_from_output_dir(
        output_dir=out_dir,
        models_dir=models_dir,
        publish_root=str(publish_root),
        prediction_crs="EPSG:27700",
    )
    assert len(published) == 1
    assert version_folder_exists(published[0])


def test_make_predictions_publish_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "preds"
    models_dir = tmp_path / "models"
    publish_root = tmp_path / "published"
    models_dir.mkdir()
    model_id = "testicus_batidae_foraging"
    pkg = models_dir / model_id
    pkg.mkdir()
    (pkg / "package.json").write_text(
        json.dumps({"model_id": model_id, "feature_names": ["band_a"], "metrics": {}}),
        encoding="utf-8",
    )
    (pkg / "thresholds.json").write_text(
        json.dumps(
            {
                "rule": POOLED_OOF_PERCENTILE_RULE,
                "threshold": 0.3,
                "model_limited": False,
            }
        ),
        encoding="utf-8",
    )

    ev = _write_tiny_prediction(tmp_path / "ev.tif")
    df = pd.DataFrame(
        {
            "model_path": [str(models_dir / "dummy.pkl")],
            "latin_name": ["Testicus batidae"],
            "activity_type": ["foraging"],
        }
    )

    def fake_load_model(model_path: Path) -> SimpleNamespace:  # noqa: ARG001
        return SimpleNamespace(named_steps={})

    def fake_apply(models, raster_path, output_path, window_size=128):  # noqa: ARG001
        output_path = Path(output_path)
        with rasterio.open(raster_path) as src:
            profile = src.profile.copy()
            profile.update(count=1)
            data = src.read(1)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.descriptions = (model_id,)

    monkeypatch.setattr(pred_mod, "load_model", fake_load_model)
    monkeypatch.setattr(pred_mod, "apply_models_to_raster", fake_apply)

    published = pred_mod.make_predictions(
        df,
        models_dir,
        ev,
        out_dir,
        boundary_path=None,
        split_files=True,
        prediction_crs="EPSG:27700",
        write_cog=True,
        publish_root=str(publish_root),
        publish_label="pipeline-test",
    )
    assert published is not None
    assert len(published) == 1
    run = json.loads((Path(published[0]) / "run.json").read_text(encoding="utf-8"))
    assert run["label"] == "pipeline-test"


def test_resolve_prediction_publish_root_prefers_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREDICTION_PUBLISH_ROOT", "memory://from-env/predictions")
    assert resolve_prediction_publish_root("/tmp/cli-root") == "/tmp/cli-root"
    assert (
        resolve_prediction_publish_root(None)
        == "memory://from-env/predictions"
    )
    monkeypatch.delenv("PREDICTION_PUBLISH_ROOT")
    assert (
        resolve_prediction_publish_root(None)
        == "gs://habitat-suitability-modelling-data/predictions"
    )
