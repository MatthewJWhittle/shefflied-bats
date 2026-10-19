"""Step timing / peak-memory recorder for EV rebuilds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdm.utils.step_metrics import StepMetricsRecorder


def test_recorder_records_ok_stub_step(tmp_path: Path) -> None:
    recorder = StepMetricsRecorder()

    def _tiny_step() -> str:
        return "done"

    assert recorder.run("stub_step", _tiny_step) == "done"
    assert len(recorder.steps) == 1
    metric = recorder.steps[0]
    assert metric.step == "stub_step"
    assert metric.status == "ok"
    assert metric.error is None
    assert metric.duration_seconds >= 0.0
    # Peak RSS is platform-dependent but should be a positive int on Linux CI.
    if metric.peak_rss_bytes is not None:
        assert metric.peak_rss_bytes > 0

    paths = recorder.write(tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["steps"][0]["step"] == "stub_step"
    assert payload["total_duration_seconds"] >= 0.0
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "stub_step" in csv_text
    assert "duration_seconds" in csv_text


def test_recorder_records_error_then_reraises(tmp_path: Path) -> None:
    recorder = StepMetricsRecorder()

    def _boom() -> None:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        recorder.run("failing_step", _boom)

    assert recorder.steps[0].status == "error"
    assert "RuntimeError" in (recorder.steps[0].error or "")
    paths = recorder.write(tmp_path, stem="failed_run")
    assert paths["json"].name == "failed_run.json"
