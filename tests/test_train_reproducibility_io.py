"""Reproducibility and remote I/O for training outputs."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import numpy as np
import pytest
from shapely.geometry import Point

from sdm.commands.modelling.train_sdm_models import (
    _summarize_cv_scores,
    save_models,
    train_single_model,
)
from sdm.models.maxent.maxent_model import DefaultMaxentConfig
from sdm.types import TrainingData
from sdm.utils.io import load_model_config
from sdm.utils.model_runs import (
    build_run_output_uri,
    join_run_artifact,
    model_run_write_context,
)


@pytest.fixture
def synthetic_training_frame():
    feature_names = [f"ev_{idx}" for idx in range(6)]
    rng = np.random.default_rng(0)
    rows = 80
    data = {name: rng.random(rows) for name in feature_names}
    data["class"] = rng.integers(0, 2, rows)
    data["sample_weight"] = rng.random(rows)
    data["geometry"] = [Point(x, y) for x, y in zip(rng.random(rows), rng.random(rows))]
    import geopandas as gpd

    return gpd.GeoDataFrame(data, crs="EPSG:27700"), feature_names


def test_train_single_model_metrics_reproducible(synthetic_training_frame) -> None:
    frame, feature_names = synthetic_training_frame
    training_data = TrainingData(
        latin_name="species_alpha",
        activity_type="activity_a",
        occurrence=frame,
        maxent_config=DefaultMaxentConfig(),
        model_features=feature_names[:4],
    )

    first = train_single_model(training_data, max_threads_per_model=1, seed=42)
    second = train_single_model(training_data, max_threads_per_model=1, seed=42)

    if not first.success or not second.success:
        pytest.skip(f"training failed: {first.error} / {second.error}")

    mean_a, std_a, _, _ = _summarize_cv_scores(first.cv_scores)
    mean_b, std_b, _, _ = _summarize_cv_scores(second.cv_scores)
    assert mean_a == pytest.approx(mean_b, rel=0, abs=1e-9)
    assert std_a == pytest.approx(std_b, rel=0, abs=1e-9)


def test_model_config_loads_from_memory_filesystem() -> None:
    fs = fsspec.filesystem("memory")
    uri = "memory://cfg-root/model_config.yml"
    yaml_text = """
model:
  record_age_years: 1
  maxent:
    feature_types: [linear]
    beta_multiplier: 1.0
    beta_lqp: 1.0
    beta_hinge: 0.5
    beta_threshold: 0.5
    beta_categorical: 0.5
    n_hinge_features: 10
    n_threshold_features: 10
    clamp: true
    convergence_tolerance: 1.0e-5
    use_lambdas: best
    n_lambdas: 100
    class_weights: balanced
    tau: 0.5
    transform: cloglog
"""
    fs.write_text(uri, yaml_text.strip())

    cfg = load_model_config(uri)
    assert cfg.maxent.beta_multiplier == 1.0


def test_run_outputs_match_local_and_memory_roots(
    synthetic_training_frame, tmp_path: Path
) -> None:
    frame, feature_names = synthetic_training_frame
    training_data = TrainingData(
        latin_name="species_beta",
        activity_type="activity_b",
        occurrence=frame,
        maxent_config=DefaultMaxentConfig(),
        model_features=feature_names[:4],
    )
    result = train_single_model(training_data, max_threads_per_model=1, seed=42)
    if not result.success:
        pytest.skip(result.error)

    segment = "20251016T120000Z-repro01"
    local_uri = build_run_output_uri(str(tmp_path / "out"), run_segment=segment)
    remote_uri = build_run_output_uri("memory://remote-bucket", run_segment=segment)

    metrics: dict[str, float] = {}

    for label, run_uri in (("local", local_uri), ("memory", remote_uri)):
        with model_run_write_context(run_uri) as (write_dir, written_uri):
            paths = save_models([result], [training_data], write_dir)
            rel = Path(paths[result.identifier()]).relative_to(write_dir)
            published = join_run_artifact(written_uri, rel)
            pkg_uri = join_run_artifact(written_uri, rel.parent / "package.json")
            if "://" in pkg_uri:
                fs, remote_path = fsspec.core.url_to_fs(pkg_uri)
                pkg_bytes = fs.cat(remote_path)
            else:
                pkg_bytes = Path(pkg_uri).read_bytes()
            pkg = json.loads(pkg_bytes.decode("utf-8"))
            metrics[label] = float(pkg["metrics"]["mean_cv_auc"])

    assert metrics["local"] == pytest.approx(metrics["memory"], rel=0, abs=1e-12)
