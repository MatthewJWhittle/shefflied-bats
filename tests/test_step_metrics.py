"""Step timing / peak-memory recorder for EV rebuilds."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import pytest

from sdm.utils.step_metrics import StepMetricsRecorder


def _child_allocate_and_hold(nbytes: int, hold_seconds: float) -> None:
    """Allocate RSS in a child process so the parent sampler can observe it."""
    buf = bytearray(nbytes)
    time.sleep(hold_seconds)
    del buf


def test_recorder_records_ok_stub_step(tmp_path: Path) -> None:
    recorder = StepMetricsRecorder(sample_interval_seconds=0.02)

    def _tiny_step() -> str:
        return "done"

    assert recorder.run("stub_step", _tiny_step) == "done"
    assert len(recorder.steps) == 1
    metric = recorder.steps[0]
    assert metric.step == "stub_step"
    assert metric.status == "ok"
    assert metric.error is None
    assert metric.duration_seconds >= 0.0
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
    recorder = StepMetricsRecorder(sample_interval_seconds=0.02)

    def _boom() -> None:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        recorder.run("failing_step", _boom)

    assert recorder.steps[0].status == "error"
    assert "RuntimeError" in (recorder.steps[0].error or "")
    paths = recorder.write(tmp_path, stem="failed_run")
    assert paths["json"].name == "failed_run.json"


def test_later_low_memory_step_reports_lower_peak() -> None:
    """Per-step peak must not inherit an earlier process high-water mark.

    High RSS is allocated in a child process so the sampler sees it via the
    process tree; after the child exits, a quiet step reports a lower peak.
    """
    recorder = StepMetricsRecorder(sample_interval_seconds=0.02)
    chunk = 40 * 1024 * 1024

    def high_memory_step() -> None:
        child = multiprocessing.Process(
            target=_child_allocate_and_hold,
            args=(chunk, 0.35),
        )
        child.start()
        child.join(timeout=5)
        assert child.exitcode == 0

    def low_memory_step() -> None:
        time.sleep(0.25)

    recorder.run("high_memory", high_memory_step)
    recorder.run("low_memory", low_memory_step)

    high_peak = recorder.steps[0].peak_rss_bytes
    low_peak = recorder.steps[1].peak_rss_bytes
    assert high_peak is not None and low_peak is not None
    assert high_peak > low_peak
    # Child held ~40 MiB; require a clear gap above sampling noise.
    assert high_peak - low_peak > 20 * 1024 * 1024
