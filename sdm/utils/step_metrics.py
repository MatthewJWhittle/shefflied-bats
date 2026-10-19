"""Record wall time and per-step peak RSS for sequential rebuild steps."""

from __future__ import annotations

import csv
import json
import resource
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.05


@dataclass
class StepMetric:
    """One timed step in a rebuild run."""

    step: str
    duration_seconds: float
    peak_rss_bytes: int | None
    status: str
    error: str | None = None


@dataclass
class StepMetricsRecorder:
    """Run callables while sampling peak resident memory for each step.

    Peak RSS is measured with a background sampling thread (psutil process tree
    when available) so a later low-memory step does not inherit an earlier
    process high-water mark from ``resource.ru_maxrss``.
    """

    steps: list[StepMetric] = field(default_factory=list)
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS

    def run(self, step_name: str, fn: Callable[[], T]) -> T:
        started = time.perf_counter()
        with _RssSampler(interval_seconds=self.sample_interval_seconds) as sampler:
            try:
                result = fn()
            except Exception as exc:
                duration = time.perf_counter() - started
                self.steps.append(
                    StepMetric(
                        step=step_name,
                        duration_seconds=round(duration, 6),
                        peak_rss_bytes=sampler.peak_rss_bytes,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
        duration = time.perf_counter() - started
        self.steps.append(
            StepMetric(
                step=step_name,
                duration_seconds=round(duration, 6),
                peak_rss_bytes=sampler.peak_rss_bytes,
                status="ok",
            )
        )
        return result

    def write(
        self,
        output_dir: Path,
        *,
        stem: str = "rebuild_metrics",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write JSON and CSV metrics next to rebuild outputs.

        ``extra`` is merged into the JSON payload (e.g. provenance or
        ``build_settings``). CSV remains step rows only.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}.csv"
        payload: dict[str, Any] = {
            "steps": [asdict(step) for step in self.steps],
            "total_duration_seconds": round(
                sum(step.duration_seconds for step in self.steps), 6
            ),
        }
        if extra:
            payload.update(extra)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "duration_seconds",
                    "peak_rss_bytes",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()
            for step in self.steps:
                writer.writerow(asdict(step))
        return {"json": json_path, "csv": csv_path}


class _RssSampler:
    """Sample process-tree RSS while a step runs; track the peak."""

    def __init__(self, *, interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = max(interval_seconds, 0.01)
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _RssSampler:
        self._record(process_tree_rss_bytes())
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="sdm-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        self._record(process_tree_rss_bytes())

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record(process_tree_rss_bytes())

    def _record(self, sample: int | None) -> None:
        if sample is None:
            return
        if self.peak_rss_bytes is None or sample > self.peak_rss_bytes:
            self.peak_rss_bytes = sample


def process_tree_rss_bytes() -> int | None:
    """Current RSS for this process and its children (bytes)."""
    try:
        import psutil

        proc = psutil.Process()
        total = int(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except Exception:
        return _fallback_rss_bytes()


def _fallback_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def write_metrics_payload(path: Path, steps: list[dict[str, Any]]) -> None:
    """Small helper for tests that build dict payloads directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"steps": steps}, indent=2) + "\n", encoding="utf-8")
