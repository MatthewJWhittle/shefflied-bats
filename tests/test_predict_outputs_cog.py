"""Tests for prediction pipeline COG outputs and CRS warnings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

import sdm.commands.modelling.predict_sdm_models as pred_mod
from sdm.raster.io import cogify_geotiff_inplace


@pytest.fixture
def tiny_ev_27700(tmp_path: Path) -> Path:
    path = tmp_path / "ev.tif"
    h, w = 16, 12
    data = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    transform = from_bounds(
        400_000,
        500_000,
        400_000 + w * 50,
        500_000 + h * 50,
        w,
        h,
    )
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def tiny_ev_3857(tmp_path: Path) -> Path:
    path = tmp_path / "ev_3857.tif"
    h, w = 8, 8
    data = np.ones((h, w), dtype=np.float32)
    transform = from_bounds(-100_000, 6_700_000, -99_600, 6_700_400, w, h)
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_cogify_geotiff_inplace_roundtrip(tmp_path: Path, tiny_ev_27700: Path) -> None:
    plain = tmp_path / "plain.tif"
    plain.write_bytes(tiny_ev_27700.read_bytes())
    cogify_geotiff_inplace(plain)
    ok, _, errs = cog_validate(plain, quiet=True)
    assert ok, errs


def test_make_predictions_finalize_emits_cogs(
    tiny_ev_27700: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "preds"
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    df = pd.DataFrame(
        {
            "model_path": [str(models_dir / "dummy.pkl")],
            "latin_name": ["Testicus batidae"],
            "activity_type": ["foraging"],
        }
    )

    def fake_load_model(model_path: Path) -> SimpleNamespace:
        return SimpleNamespace(named_steps={})

    def fake_apply(models, raster_path, output_path, window_size=128):  # noqa: ARG001
        output_path = Path(output_path)
        with rasterio.open(raster_path) as src:
            profile = src.profile.copy()
            profile.update(count=2)
            stack = np.stack([src.read(1), src.read(1) * 0.5], axis=0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stack)
            dst.descriptions = ("alpha_band", "beta_band")

    monkeypatch.setattr(pred_mod, "load_model", fake_load_model)
    monkeypatch.setattr(pred_mod, "apply_models_to_raster", fake_apply)

    pred_mod.make_predictions(
        df,
        models_dir,
        tiny_ev_27700,
        out_dir,
        boundary_path=None,
        split_files=True,
        prediction_crs="EPSG:27700",
        write_cog=True,
    )

    merged = out_dir / "all_predictions.tif"
    assert merged.is_file()
    assert not (out_dir / "_all_predictions_staging.tif").exists()

    ok, _, errs = cog_validate(merged, quiet=True)
    assert ok, errs

    band_paths = sorted(out_dir.glob("prediction_*.tif"))
    assert len(band_paths) == 2
    for p in band_paths:
        ok_b, _, e_b = cog_validate(p, quiet=True)
        assert ok_b, e_b


def test_make_predictions_logs_crs_mismatch_warning(
    tiny_ev_3857: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    out_dir = tmp_path / "preds"
    models_dir = tmp_path / "models"
    models_dir.mkdir()

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

    monkeypatch.setattr(pred_mod, "load_model", fake_load_model)
    monkeypatch.setattr(pred_mod, "apply_models_to_raster", fake_apply)

    caplog.set_level("WARNING")

    pred_mod.make_predictions(
        df,
        models_dir,
        tiny_ev_3857,
        out_dir,
        boundary_path=None,
        split_files=False,
        prediction_crs="EPSG:27700",
        write_cog=False,
    )

    assert any("differs from prediction output CRS" in r.message for r in caplog.records)

    merged = out_dir / "all_predictions.tif"
    assert merged.is_file()
    with rasterio.open(merged) as ds:
        assert ds.crs.to_epsg() == 27700
