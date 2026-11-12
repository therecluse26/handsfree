"""Latency measurement.

Each processed frame carries named timestamps ("marks"), in pipeline order,
for example ``capture, received, start, face, gaze``. The tracker records the
time between each pair of neighbouring marks and the total from the first
mark to the last one. It keeps a rolling window and reports median, p95 and
max in milliseconds.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

STAGE_LABELS = {
    "capture→received": "camera driver → app",
    "received→start": "wait for pipeline",
    "start→face": "face model",
    "face→gaze": "gaze features",
    "gaze→display": "hand-off to window",
    "gaze→output": "map + cursor",
}


@dataclass(frozen=True, slots=True)
class StageStats:
    name: str
    median_ms: float
    p95_ms: float
    max_ms: float
    count: int


class LatencyTracker:
    def __init__(self, window: int = 300) -> None:
        self._window = window
        self._stages: dict[str, deque[float]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def record(self, marks: dict[str, float]) -> None:
        """Record one frame. ``marks`` maps names to monotonic seconds, in pipeline order."""
        items = list(marks.items())
        if len(items) < 2:
            return
        with self._lock:
            for (a, ta), (b, tb) in zip(items, items[1:]):
                self._add(f"{a}→{b}", tb - ta)
            self._add("total", items[-1][1] - items[0][1])

    def _add(self, name: str, seconds: float) -> None:
        q = self._stages.get(name)
        if q is None:
            q = self._stages[name] = deque(maxlen=self._window)
            self._order.append(name)
        q.append(seconds * 1000.0)

    def summary(self) -> list[StageStats]:
        with self._lock:
            names = [n for n in self._order if n != "total"] + (["total"] if "total" in self._stages else [])
            out = []
            for n in names:
                a = np.fromiter(self._stages[n], dtype=float)
                out.append(
                    StageStats(n, float(np.median(a)), float(np.percentile(a, 95)), float(a.max()), len(a))
                )
            return out

    def format_table(self) -> str:
        rows = self.summary()
        if not rows:
            return "no frames measured yet"
        lines = [f"{'stage':<34}{'median':>9}{'p95':>9}{'max':>9}"]
        for s in rows:
            label = STAGE_LABELS.get(s.name, s.name)
            lines.append(f"{label:<34}{s.median_ms:>7.1f}ms{s.p95_ms:>7.1f}ms{s.max_ms:>7.1f}ms")
        return "\n".join(lines)


class RateMeter:
    """Events per second over a rolling window of timestamps."""

    def __init__(self, window: int = 60) -> None:
        self._t: deque[float] = deque(maxlen=window)

    def tick(self, t: float) -> None:
        self._t.append(t)

    @property
    def rate(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0
