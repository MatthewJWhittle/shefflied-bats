"""Tests for predict model-run directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdm.utils.model_runs import (
    list_local_run_ids,
    resolve_predict_models_dir,
    resolve_run_prefix,
    run_segment_from_uri,
)


def _touch_run(root: Path, run_id: str) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "model_results.csv").write_text("identifier\n", encoding="utf-8")
    return run_dir


def test_newest_run_is_default(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    older = _touch_run(root, "20261024T100000Z-aaaa1111")
    newer = _touch_run(root, "20261025T200629Z-8ce0f946")
    assert list_local_run_ids(root) == [
        "20261024T100000Z-aaaa1111",
        "20261025T200629Z-8ce0f946",
    ]
    assert resolve_predict_models_dir(root) == newer
    assert resolve_predict_models_dir(root) != older


def test_unique_run_prefix(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    chosen = _touch_run(root, "20261025T200629Z-8ce0f946")
    _touch_run(root, "20261024T100000Z-aaaa1111")
    assert (
        resolve_predict_models_dir(root, run="20261025T2006") == chosen
    )
    assert resolve_run_prefix(
        list_local_run_ids(root), "20261025T200629Z-8ce0f946"
    ) == "20261025T200629Z-8ce0f946"


def test_ambiguous_run_prefix(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    _touch_run(root, "20261025T200629Z-8ce0f946")
    _touch_run(root, "20261025T200640Z-bbbb2222")
    with pytest.raises(ValueError, match="matches more than one run"):
        resolve_predict_models_dir(root, run="20261025T2006")


def test_no_runs_errors_clearly(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="No trained model runs found"):
        resolve_predict_models_dir(root)
    with pytest.raises(FileNotFoundError, match="No trained model runs found"):
        resolve_predict_models_dir(root, run="20261025T2006")


def test_explicit_models_dir_override(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    _touch_run(root, "20261025T200629Z-8ce0f946")
    explicit = tmp_path / "custom_run"
    explicit.mkdir()
    (explicit / "model_results.csv").write_text("identifier\n", encoding="utf-8")
    assert resolve_predict_models_dir(explicit) == explicit
    with pytest.raises(ValueError, match="--run cannot be combined"):
        resolve_predict_models_dir(explicit, run="20261025T2006")


def test_flat_pkl_without_index_errors(tmp_path: Path) -> None:
    root = tmp_path / "sdm_models"
    root.mkdir()
    (root / "some_species_roost.pkl").write_bytes(b"not-a-model")
    with pytest.raises(FileNotFoundError, match="has .pkl files but no"):
        resolve_predict_models_dir(root)


def test_run_segment_from_uri() -> None:
    assert (
        run_segment_from_uri("data/sdm_models/runs/20261025T200629Z-8ce0f946")
        == "20261025T200629Z-8ce0f946"
    )
    assert (
        run_segment_from_uri(
            "gs://bucket/runs/20261025T200629Z-8ce0f946/"
        )
        == "20261025T200629Z-8ce0f946"
    )
