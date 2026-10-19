"""Record wall time and peak RSS for sequential rebuild steps."""

from __future__ import annotations

import csv
import json
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


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
    """Run callables while collecting duration and peak resident memory."""

    steps: list[StepMetric] = field(default_factory=list)

    def run(self, step_name: str, fn: Callable[[], T]) -> T:
        rss_before = _sample_rss_bytes()
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            duration = time.perf_counter() - started
            peak = _peak_rss_bytes(rss_before)
            self.steps.append(
                StepMetric(
                    step=step_name,
                    duration_seconds=round(duration, 6),
                    peak_rss_bytes=peak,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        duration = time.perf_counter() - started
        peak = _peak_rss_bytes(rss_before)
        self.steps.append(
            StepMetric(
                step=step_name,
                duration_seconds=round(duration, 6),
                peak_rss_bytes=peak,
                status="ok",
            )
        )
        return result

    def write(self, output_dir: Path, *, stem: str = "rebuild_metrics") -> dict[str, Path]:
        """Write JSON and CSV metrics next to rebuild outputs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}.csv"
        payload = {
            "steps": [asdict(step) for step in self.steps],
            "total_duration_seconds": round(
                sum(step.duration_seconds for step in self.steps), 6
            ),
        }
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


def _sample_rss_bytes() -> int | None:
    proc_rss = _rss_from_proc()
    if proc_rss is not None:
        return proc_rss
    return _rss_from_resource()


def _rss_from_proc() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Value is in kB on Linux.
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _rss_from_resource() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    # Linux reports kilobytes; macOS reports bytes.
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _peak_rss_bytes(rss_before: int | None) -> int | None:
    samples = [value for value in (rss_before, _sample_rss_bytes()) if value is not None]
    if not samples:
        return None
    return max(samples)


def write_metrics_payload(path: Path, steps: list[dict[str, Any]]) -> None:
    """Small helper for tests that build dict payloads directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"steps": steps}, indent=2) + "\n", encoding="utf-8")
