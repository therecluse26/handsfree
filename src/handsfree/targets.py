"""A fullscreen window that shows targets one at a time and records gaze features.

The signal probe, calibration and the accuracy test all use it.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .config import Config
from .face import face_brightness
from .pipeline import Pipeline
from .preview import PipelineWorker
from .sessions import TargetSession, now_iso
from .types import Target



class TargetSequenceWindow(QWidget):
    """Shows each target, waits for the settle time, then records.

    After the window closes: ``completed`` tells if the user saw every target,
    ``samples[i]`` holds the feature vectors recorded for target ``i``, and
    ``times[i]`` holds their times in seconds since that recording began.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        targets: list[Target],
        intro: str,
        settle_s: float = 1.0,
        record_s: float = 1.5,
        autostart: bool = False,
        prompt_read_s: float = 5.0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("handsfree — targets")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.pipeline = pipeline
        self.targets = targets
        self.intro = intro
        self.settle_s = settle_s
        self.record_s = record_s
        self.prompt_read_s = prompt_read_s
        self.samples: list[list[np.ndarray]] = [[] for _ in targets]
        self.times: list[list[float]] = [[] for _ in targets]
        self.brightness: list[float] = []
        self.feature_names: tuple[str, ...] = ()
        self.completed = False
        self._index = -1
        self._t_target = 0.0
        self._face_seen = False
        self.worker = PipelineWorker(pipeline)
        self.worker.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(8)
        if autostart:
            self._next()

    def _durations(self) -> tuple[float, float]:
        """(settle, record) seconds for the current target."""
        t = self.targets[self._index]
        # A target with a prompt waits longer, so the user can read the prompt first.
        settle = max(self.settle_s, self.prompt_read_s) if t.prompt else self.settle_s
        return settle, (t.record_s if t.record_s is not None else self.record_s)

    def _next(self) -> None:
        self._index += 1
        self._t_target = time.monotonic()
        if self._index >= len(self.targets):
            self.completed = True
            self.close()

    def _tick(self) -> None:
        r = self.worker.take()
        now = time.monotonic()
        active = 0 <= self._index < len(self.targets)
        settle, record = self._durations() if active else (0.0, 0.0)
        if r is not None:
            self._face_seen = r.gaze is not None
            if r.gaze is not None and r.face is not None:
                self.feature_names = r.gaze.names
                since = now - self._t_target - settle
                if active and since >= 0 and not r.gaze.blinking:
                    self.samples[self._index].append(r.gaze.vector)
                    self.times[self._index].append(since)
                    self.brightness.append(face_brightness(r.frame.image, r.face.landmarks))
        if active and now - self._t_target >= settle + record:
            self._next()
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        font = QFont(self.font())
        font.setPointSize(16)
        p.setFont(font)
        if self.worker.error:
            p.setPen(QColor(255, 120, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"Error: {self.worker.error}\n\nEsc: close")
            return
        if self._index < 0:
            p.setPen(QColor(225, 225, 225))
            status = "" if self._face_seen else "\n\n(no face found yet)"
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.intro + status)
            return
        if self._index >= len(self.targets):
            return
        t = self.targets[self._index]
        settle_s, record_s = self._durations()
        c = QPointF(t.x * self.width(), t.y * self.height())
        elapsed = time.monotonic() - self._t_target
        settle = min(1.0, elapsed / settle_s) if settle_s > 0 else 1.0
        recording = settle >= 1.0
        colour = QColor(80, 255, 140) if recording else QColor(240, 240, 240)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, 2))
        p.drawEllipse(c, 46 - 32 * settle, 46 - 32 * settle)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        p.drawEllipse(c, 4, 4)
        if t.prompt:
            p.setPen(QColor(225, 225, 225))
            box = QRectF(0, c.y() - 190, self.width(), 130)
            p.drawText(box, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, t.prompt)
            left = max(0.0, settle_s + record_s - elapsed)
            state = f"recording, {left:.0f} s" if recording else f"read this first. Starts in {math.ceil(settle_s - elapsed)} s"
            p.setPen(colour)
            p.drawText(QRectF(0, c.y() + 60, self.width(), 40), Qt.AlignmentFlag.AlignHCenter, state)
        font.setPointSize(10)
        p.setFont(font)
        p.setPen(QColor(120, 120, 130) if self._face_seen else QColor(255, 120, 120))
        note = "" if self._face_seen else "   NO FACE"
        p.drawText(
            self.rect().adjusted(0, 0, 0, -12),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            f"{self._index + 1} / {len(self.targets)}{note}",
        )

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif event.key() == Qt.Key.Key_Space and self._index < 0 and not self.worker.error:
            self._next()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.worker.stop()
        super().closeEvent(event)


def run_target_session(
    cfg: Config,
    kind: str,
    targets: list[Target],
    intro: str,
    settle_s: float,
    record_s: float,
    autostart: bool = False,
    prompt_read_s: float = 5.0,
) -> TargetSession | None:
    """Show the targets fullscreen. Return None if the user cancelled."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    pipeline = Pipeline.from_config(cfg)
    window = TargetSequenceWindow(pipeline, targets, intro, settle_s, record_s, autostart, prompt_read_s)
    window.showFullScreen()
    app.exec()
    pipeline.close()
    if not window.completed:
        return None
    screen = window.screen() or app.primaryScreen()
    geo, mm = screen.geometry(), screen.physicalSize()
    k = len(window.feature_names)
    return TargetSession(
        kind=kind,
        created=now_iso(),
        targets=targets,
        samples=[np.array(s, dtype=float).reshape(len(s), k) for s in window.samples],
        times=[np.array(t, dtype=float) for t in window.times],
        names=window.feature_names,
        brightness=float(np.mean(window.brightness)) if window.brightness else 0.0,
        screen_px=(geo.width(), geo.height()),
        screen_mm=(float(mm.width()), float(mm.height())),
    )
