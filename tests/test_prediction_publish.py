"""Versioned prediction publishing: local/fake FS only (no real bucket)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

import sdm.commands.modelling.predict_sdm_models as pred_mod
from sdm.models.core.prediction import (
    save_prediction_raster,
    write_threshold_binary_raster,
)
from sdm.models.core.threshold import POOLED_OOF_PERCENTILE_RULE
from sdm.utils.io import load_config
from sdm.utils.prediction_publish import (
    binomial_slug_from_latin_name,
    publish_model_prediction_version,
    publish_name_slug,
    publish_predictions_from_output_dir,
    published_artifact_filename,
    resolve_prediction_publish_root,
    build_version_folder_name,
    stable_config_hash,
    threshold_kind_for_rule,
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
    unknown = build_version_folder_name(
        created_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
        git_sha=None,
        config_hash="84798104",
    )
    assert unknown == "20261017_unknown_cfg84798104"


def test_binomial_slug_from_full_latin_name() -> None:
    assert binomial_slug_from_latin_name("Myotis brandtii") == "myotis_brandtii"
    assert binomial_slug_from_latin_name("plecotus_auritus") == "plecotus_auritus"
    assert binomial_slug_from_latin_name("Pipistrellus nathusii") == (
        "pipistrellus_nathusii"
    )
    assert binomial_slug_from_latin_name("Eptesicus serotinus") == (
        "eptesicus_serotinus"
    )
    with pytest.raises(ValueError, match="full snake-case Latin binomial"):
        binomial_slug_from_latin_name("brandtii")
    with pytest.raises(ValueError, match="full snake-case Latin binomial"):
        binomial_slug_from_latin_name("pipistrellus")


def test_publish_name_slug_and_artifact_filename() -> None:
    slug = publish_name_slug(
        latin_name="Myotis brandtii",
        activity_type="Roost",
    )
    assert slug == "myotis_brandtii_roost"
    version = "20261017_unknown_cfg84798104"
    assert (
        published_artifact_filename(slug, version, "suitability", ext="tif")
        == "prediction__myotis_brandtii_roost__20261017_unknown_cfg84798104__suitability.tif"
    )
    assert (
        published_artifact_filename(
            slug, version, threshold_kind_for_rule(POOLED_OOF_PERCENTILE_RULE), ext="tif"
        )
        == "prediction__myotis_brandtii_roost__20261017_unknown_cfg84798104__threshold_pooled_oof_percentile.tif"
    )
    assert (
        published_artifact_filename(slug, version, "run", ext="json")
        == "prediction__myotis_brandtii_roost__20261017_unknown_cfg84798104__run.json"
    )


def test_pinned_publish_filenames_for_existing_models() -> None:
    """All 17 trained models keep byte-identical publish names (no species map)."""
    version = "20261017_unknown_cfg84798104"
    # (latin_name, activity_type) → expected suitability file name.
    # Brandt's is roost-only; Nathusius config exists but does not train.
    pinned = [
        ("Myotis brandtii", "Roost", "myotis_brandtii_roost"),
        ("Myotis daubentonii", "In flight", "myotis_daubentonii_in_flight"),
        ("Myotis daubentonii", "Roost", "myotis_daubentonii_roost"),
        ("Myotis mystacinus", "In flight", "myotis_mystacinus_in_flight"),
        ("Myotis mystacinus", "Roost", "myotis_mystacinus_roost"),
        ("Myotis nattereri", "In flight", "myotis_nattereri_in_flight"),
        ("Myotis nattereri", "Roost", "myotis_nattereri_roost"),
        ("Nyctalus leisleri", "In flight", "nyctalus_leisleri_in_flight"),
        ("Nyctalus leisleri", "Roost", "nyctalus_leisleri_roost"),
        ("Nyctalus noctula", "In flight", "nyctalus_noctula_in_flight"),
        ("Nyctalus noctula", "Roost", "nyctalus_noctula_roost"),
        ("Pipistrellus pipistrellus", "In flight", "pipistrellus_pipistrellus_in_flight"),
        ("Pipistrellus pipistrellus", "Roost", "pipistrellus_pipistrellus_roost"),
        ("Pipistrellus pygmaeus", "In flight", "pipistrellus_pygmaeus_in_flight"),
        ("Pipistrellus pygmaeus", "Roost", "pipistrellus_pygmaeus_roost"),
        ("Plecotus auritus", "In flight", "plecotus_auritus_in_flight"),
        ("Plecotus auritus", "Roost", "plecotus_auritus_roost"),
    ]
    assert len(pinned) == 17
    for latin_name, activity_type, slug in pinned:
        assert publish_name_slug(
            latin_name=latin_name, activity_type=activity_type
        ) == slug
        # No doubled genus from a former short-name → Latin map.
        assert "myotis_myotis_" not in slug
        assert "nyctalus_nyctalus_" not in slug
        assert "pipistrellus_pipistrellus_pipistrellus" not in slug
        assert "plecotus_plecotus_" not in slug
        expected = f"prediction__{slug}__{version}__suitability.tif"
        assert (
            published_artifact_filename(slug, version, "suitability", ext="tif")
            == expected
        )


def test_threshold_raster_filename_includes_rule() -> None:
    assert (
        threshold_kind_for_rule(POOLED_OOF_PERCENTILE_RULE)
        == "threshold_pooled_oof_percentile"
    )
    assert (
        threshold_raster_filename(
            "myotis_brandtii_roost",
            "20261017_unknown_cfg84798104",
            POOLED_OOF_PERCENTILE_RULE,
        )
        == "prediction__myotis_brandtii_roost__20261017_unknown_cfg84798104__threshold_pooled_oof_percentile.tif"
    )


def test_write_threshold_binary_raster_known_answer(tmp_path: Path) -> None:
    data = np.array(
        [
            [0.05, 0.20, -9999.0],
            [0.10, 0.50, 0.09],
        ],
        dtype=np.float32,
    )
    pred = tmp_path / "pred.tif"
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

    out = tmp_path / "threshold.tif"
    write_threshold_binary_raster(pred, out, cutoff=0.10, as_cog=True)

    ok, _, errs = cog_validate(out, quiet=True)
    assert ok, errs
    with rasterio.open(out) as ds:
        binary = ds.read(1)
        assert ds.nodata == 255
        assert binary.dtype == np.uint8
        assert binary[0, 0] == 0
        assert binary[0, 1] == 1
        assert binary[0, 2] == 255
        assert binary[1, 0] == 1
        assert binary[1, 1] == 1
        assert binary[1, 2] == 0


def test_save_prediction_raster_optional_cog(tmp_path: Path) -> None:
    data = np.ones((4, 4), dtype=np.float32) * 0.4
    transform = from_bounds(0, 0, 4, 4, 4, 4)
    path = tmp_path / "suitability.tif"
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
    model_id = "myotis_brandtii_roost"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    package = {
        "schema_version": 1,
        "model_id": model_id,
        "latin_name": "Myotis brandtii",
        "activity_type": "Roost",
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
    created = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)

    uri = publish_model_prediction_version(
        model_id=model_id,
        prediction_raster=prediction,
        publish_root=str(publish_root),
        models_dir=models_dir,
        config={"prediction_crs": "EPSG:27700", "write_cog": True},
        label="unit-test",
        created_at=created,
        git_sha="",  # empty → "unknown" segment (no auto git lookup)
        git_dirty=False,
        write_cog=True,
    )

    version_dir = Path(uri)
    assert version_dir.is_dir()
    assert version_dir.name.startswith("20261017_unknown_cfg")
    version = version_dir.name
    slug = "myotis_brandtii_roost"
    suit_name = f"prediction__{slug}__{version}__suitability.tif"
    thr_name = (
        f"prediction__{slug}__{version}__threshold_pooled_oof_percentile.tif"
    )
    run_name = f"prediction__{slug}__{version}__run.json"
    assert (version_dir / suit_name).is_file()
    assert (version_dir / thr_name).is_file()
    run = json.loads((version_dir / run_name).read_text(encoding="utf-8"))
    assert run["model_id"] == model_id
    assert run["publish_slug"] == slug
    assert run["version"] == version
    assert run["git_sha"] == ""
    assert run["git_dirty"] is False
    assert run["label"] == "unit-test"
    assert run["cutoff"]["rule"] == POOLED_OOF_PERCENTILE_RULE
    assert run["cutoff"]["value"] == pytest.approx(0.25)
    assert run["cv_metrics"]["mean_cv_auc"] == pytest.approx(0.81)
    assert run["files"] == {
        "suitability": suit_name,
        "threshold": thr_name,
        "run": run_name,
    }

    ok_p, _, e_p = cog_validate(version_dir / suit_name, quiet=True)
    assert ok_p, e_p
    ok_t, _, e_t = cog_validate(version_dir / thr_name, quiet=True)
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
            git_sha="",
            git_dirty=False,
            write_cog=True,
        )


def test_publish_skips_threshold_when_model_limited(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_id = "myotis_mystacinus_in_flight"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "latin_name": "Myotis mystacinus",
                "activity_type": "In flight",
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
    version = version_dir.name
    slug = "myotis_mystacinus_in_flight"
    suit_name = f"prediction__{slug}__{version}__suitability.tif"
    run_name = f"prediction__{slug}__{version}__run.json"
    assert (version_dir / suit_name).is_file()
    run = json.loads((version_dir / run_name).read_text(encoding="utf-8"))
    assert run["files"]["suitability"] == suit_name
    assert run["files"]["threshold"] is None
    assert run["files"]["run"] == run_name
    assert list(version_dir.glob("prediction__*__threshold_*.tif")) == []


def test_publish_predictions_from_output_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    out_dir = tmp_path / "sdm_predictions"
    publish_root = tmp_path / "bucket" / "predictions"
    model_id = "nyctalus_noctula_roost"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "latin_name": "Nyctalus noctula",
                "activity_type": "Roost",
                "feature_names": [],
                "metrics": {},
            }
        ),
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
    model_id = "plecotus_auritus_roost"
    pkg = models_dir / model_id
    pkg.mkdir()
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "latin_name": "Plecotus auritus",
                "activity_type": "Roost",
                "feature_names": ["band_a"],
                "metrics": {},
            }
        ),
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
            "latin_name": ["Plecotus auritus"],
            "activity_type": ["Roost"],
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
    version_dir = Path(published[0])
    run_files = list(version_dir.glob("prediction__*__run.json"))
    assert len(run_files) == 1
    run = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run["label"] == "pipeline-test"
    assert run["files"]["suitability"].endswith("__suitability.tif")
    assert run["files"]["threshold"].endswith(
        "__threshold_pooled_oof_percentile.tif"
    )


def test_resolve_prediction_publish_root_prefers_cli_then_env_then_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_cfg = {
        "prediction_publish": {
            "root": "memory://from-config/predictions",
        }
    }
    monkeypatch.setenv("PREDICTION_PUBLISH_ROOT", "memory://from-env/predictions")
    assert resolve_prediction_publish_root("/tmp/cli-root", config=project_cfg) == (
        "/tmp/cli-root"
    )
    assert (
        resolve_prediction_publish_root(None, config=project_cfg)
        == "memory://from-env/predictions"
    )
    monkeypatch.delenv("PREDICTION_PUBLISH_ROOT")
    assert (
        resolve_prediction_publish_root(None, config=project_cfg)
        == "memory://from-config/predictions"
    )


def test_resolve_prediction_publish_root_fails_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PREDICTION_PUBLISH_ROOT", raising=False)
    with pytest.raises(ValueError, match="publish root is not configured"):
        resolve_prediction_publish_root(None, config={})
    with pytest.raises(ValueError, match="publish root is not configured"):
        resolve_prediction_publish_root(
            None,
            config={"prediction_publish": {}},
        )


def test_project_config_yml_keeps_publish_root_for_this_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This project's config.yml supplies the former library default root."""
    monkeypatch.delenv("PREDICTION_PUBLISH_ROOT", raising=False)
    assert resolve_prediction_publish_root(None) == (
        "gs://habitat-suitability-modelling-data/predictions"
    )
    publish_section = load_config().get("prediction_publish") or {}
    assert "species_binomial_by_key" not in publish_section


def test_publish_fails_without_full_latin_name(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_id = "orphan_model"
    pkg = models_dir / model_id
    pkg.mkdir(parents=True)
    (pkg / "package.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "activity_type": "Roost",
                "feature_names": [],
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    prediction = _write_tiny_prediction(tmp_path / f"prediction_{model_id}.tif")
    with pytest.raises(ValueError, match="no full Latin name"):
        publish_model_prediction_version(
            model_id=model_id,
            prediction_raster=prediction,
            publish_root=str(tmp_path / "predictions"),
            models_dir=models_dir,
            config={},
            created_at=datetime(2026, 10, 25, tzinfo=timezone.utc),
            git_sha="1234567",
            git_dirty=False,
        )
