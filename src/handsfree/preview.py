"""Live preview window: camera image with eye overlays, and the gaze estimate.

Keys: Q or Esc quits, C re-centres the uncalibrated gaze point, M toggles
the mirror view.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .config import Config
from .face import LOW_LIGHT, face_brightness
from .latency import LatencyTracker, RateMeter
from .calibration import CalibratedMapper, Calibration
from .mapping import UncalibratedMapper
from .pipeline import Pipeline, PipelineResult

GESTURES = (
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "browDownLeft",
    "mouthLeft",
    "mouthRight",
    "mouthPucker",
    "cheekPuff",
    "jawOpen",
)


class PipelineWorker(threading.Thread):
    """Runs the pipeline in a loop and keeps only the newest result."""

    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__(daemon=True, name="handsfree-pipeline")
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._latest: PipelineResult | None = None
        self._stop_flag = threading.Event()
        self.error: str | None = None

    def run(self) -> None:
        misses = 0
        try:
            while not self._stop_flag.is_set():
                result = self._pipeline.step()
                if result is None:
                    misses += 1
                    if misses > 100:
                        self.error = "the camera sends no frames"
                        return
                    time.sleep(0.01)
                    continue
                misses = 0
                with self._lock:
                    self._latest = result
        except Exception as e:  # shown in the window; the thread must not die silently
            self.error = f"{type(e).__name__}: {e}"

    def take(self) -> PipelineResult | None:
        with self._lock:
            result, self._latest = self._latest, None
            return result

    def stop(self) -> None:
        self._stop_flag.set()
        self.join(timeout=2.0)


class CameraView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.result: PipelineResult | None = None
        self.mirror = True
        self.message = "Starting camera…"

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 24))
        r = self.result
        if r is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        img = r.frame.image
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        if self.mirror:
            qimg = qimg.mirrored(True, False)
        scale = min(self.width() / w, self.height() / h)
        ox = (self.width() - w * scale) / 2
        oy = (self.height() - h * scale) / 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(ox, oy, w * scale, h * scale), qimg)

        def pt(x: float, y: float) -> QPointF:
            if self.mirror:
                x = w - x
            return QPointF(ox + x * scale, oy + y * scale)

        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if r.face is None or r.gaze is None:
            p.setPen(QColor(255, 120, 120))
            p.drawText(QRectF(ox, oy, w * scale, 30), Qt.AlignmentFlag.AlignCenter, "NO FACE")
            return
        lm = r.face.landmarks
        p.setPen(QPen(QColor(255, 255, 255, 70), 1.5))
        for x, y in lm[::3, :2]:
            p.drawPoint(pt(x * w, y * h))
        colour = QColor(255, 190, 60) if r.gaze.blinking else QColor(80, 255, 140)
        for eye in r.gaze.eyes:
            c = pt(*eye.center_px)
            half = eye.width_px * scale / 2
            p.setPen(QPen(QColor(90, 170, 255), 1.5))
            p.drawEllipse(c, half, half * 0.45)
            p.drawLine(QPointF(c.x() - 3, c.y()), QPointF(c.x() + 3, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 3), QPointF(c.x(), c.y() + 3))
            p.setPen(QPen(colour, 2))
            p.drawEllipse(pt(*eye.iris_px), half * 0.3, half * 0.3)


class EyeZoomView(QWidget):
    """Both eyes, enlarged, with the tracked iris ring and eye corners."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(130)
        self.result: PipelineResult | None = None
        self.mirror = True

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 24))
        r = self.result
        if r is None or r.gaze is None:
            return
        img = r.frame.image
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        eyes = list(r.gaze.eyes)
        if self.mirror:
            eyes.reverse()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        cell_w = self.width() / 2
        for i, eye in enumerate(eyes):
            cell = QRectF(i * cell_w + 4, 4, cell_w - 8, self.height() - 8)
            cx, cy = eye.center_px
            half_w = eye.width_px * 0.85
            scale = cell.width() / (2 * half_w)
            half_h = cell.height() / scale / 2
            mid = cell.center()
            p.save()
            p.setClipRect(cell)
            if self.mirror:
                p.translate(mid.x(), 0)
                p.scale(-1, 1)
                p.translate(-mid.x(), 0)
            p.drawImage(cell, qimg, QRectF(cx - half_w, cy - half_h, 2 * half_w, 2 * half_h))
            p.restore()

            def pt(x: float, y: float) -> QPointF:
                dx = (x - cx) * scale
                return QPointF(mid.x() + (-dx if self.mirror else dx), mid.y() + (y - cy) * scale)

            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(90, 170, 255), 1.5))
            for corner in (eye.corner_a_px, eye.corner_b_px):
                c = pt(*corner)
                p.drawLine(QPointF(c.x(), c.y() - 8), QPointF(c.x(), c.y() + 8))
            p.drawLine(QPointF(mid.x() - 5, mid.y()), QPointF(mid.x() + 5, mid.y()))
            p.drawLine(QPointF(mid.x(), mid.y() - 5), QPointF(mid.x(), mid.y() + 5))
            p.setPen(QPen(QColor(255, 190, 60) if r.gaze.blinking else QColor(80, 255, 140), 1.5))
            ic = pt(*eye.iris_px)
            p.drawEllipse(ic, eye.iris_radius_px * scale, eye.iris_radius_px * scale)
            p.drawEllipse(ic, 1.5, 1.5)
            p.setPen(QColor(230, 230, 230))
            p.drawText(
                cell.adjusted(6, 4, -6, -4),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                f"h {eye.h:+.3f}   v {eye.v:+.3f}",
            )


class GazePad(QWidget):
    """A small picture of the screen with the estimated gaze point on it."""

    def __init__(self, trail_length: int, aspect: float) -> None:
        super().__init__()
        self.setMinimumSize(320, 200)
        self.aspect = aspect
        self.trail: deque[tuple[float, float]] = deque(maxlen=max(1, trail_length + 1))
        self.blinking = False

    def push(self, point: tuple[float, float] | None, blinking: bool) -> None:
        self.blinking = blinking
        if point is not None:
            self.trail.append(point)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = min(self.width(), self.height() * self.aspect)
        h = w / self.aspect
        box = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.fillRect(box, QColor(32, 34, 40))
        p.setPen(QPen(QColor(70, 74, 84), 1))
        p.drawRect(box)
        p.drawLine(QPointF(box.center().x(), box.top()), QPointF(box.center().x(), box.bottom()))
        p.drawLine(QPointF(box.left(), box.center().y()), QPointF(box.right(), box.center().y()))
        n = len(self.trail)
        p.setPen(Qt.PenStyle.NoPen)
        for i, (x, y) in enumerate(self.trail):
            last = i == n - 1
            if last:
                c = QColor(255, 190, 60) if self.blinking else QColor(80, 255, 140)
            else:
                c = QColor(80, 255, 140, int(20 + 110 * (i + 1) / n))
            p.setBrush(c)
            radius = 9.0 if last else 4.0
            p.drawEllipse(QPointF(box.left() + x * box.width(), box.top() + y * box.height()), radius, radius)


class PreviewWindow(QWidget):
    def __init__(self, cfg: Config, pipeline: Pipeline, seconds: float | None = None) -> None:
        super().__init__()
        calibration = Calibration.load_or_none()
        self.calibrated = calibration is not None
        state = f"calibrated, {calibration.model.spec.name}" if calibration else "uncalibrated"
        self.setWindowTitle(f"handsfree — preview ({state})")
        self.cfg = cfg
        self.pipeline = pipeline
        self.latency = LatencyTracker()
        self.rate = RateMeter()
        self.mapper = (
            CalibratedMapper(calibration.model)
            if calibration
            else UncalibratedMapper(cfg.preview.gain_x, cfg.preview.gain_y)
        )
        self.frames = 0
        self.faces = 0

        screen = QApplication.primaryScreen().geometry()
        self.view = CameraView()
        self.view.mirror = cfg.preview.mirror
        self.zoom = EyeZoomView()
        self.zoom.mirror = cfg.preview.mirror
        self.pad = GazePad(cfg.preview.trail_length, screen.width() / max(1, screen.height()))
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(9)
        self.stats = QLabel()
        self.stats.setFont(mono)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        hint = QLabel("Q / Esc: quit    C: re-centre gaze point    M: mirror")
        hint.setFont(QFont(mono))

        side = QVBoxLayout()
        side.addWidget(QLabel(f"Estimated gaze point on the screen ({state})"))
        side.addWidget(self.pad, 2)
        side.addWidget(self.stats, 3)
        side.addWidget(hint)
        left = QVBoxLayout()
        left.addWidget(self.view, 3)
        left.addWidget(QLabel("Eyes, enlarged. Green ring = tracked iris. Blue lines = eye corners and eye centre."))
        left.addWidget(self.zoom, 1)
        row = QHBoxLayout(self)
        row.addLayout(left, 3)
        row.addLayout(side, 2)
        self.resize(1250, 820)

        self.worker = PipelineWorker(pipeline)
        self.worker.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(8)
        self._last_stats = 0.0
        if seconds:
            QTimer.singleShot(int(seconds * 1000), self.close)

    def _tick(self) -> None:
        if self.worker.error:
            self.view.result = None
            self.view.message = f"Error: {self.worker.error}"
            self.view.update()
            return
        r = self.worker.take()
        if r is None:
            return
        now = time.monotonic()
        self.latency.record({**r.marks, "display": now})
        self.rate.tick(r.frame.t_received)
        self.frames += 1
        self.view.result = r
        self.view.update()
        self.zoom.result = r
        self.zoom.update()
        if r.gaze is not None:
            self.faces += 1
            self.pad.push(self.mapper.map(r.gaze), r.gaze.blinking)
            self.pad.update()
        if now - self._last_stats > 0.2:
            self._last_stats = now
            self.stats.setText(self._stats_text(r))

    def _stats_text(self, r: PipelineResult) -> str:
        h, w = r.frame.image.shape[:2]
        lines = [f"camera   {w}x{h}   {self.rate.rate:5.1f} fps", ""]
        if r.face is None or r.gaze is None:
            lines.append("face     NOT FOUND")
        else:
            g = r.gaze
            light = face_brightness(r.frame.image, r.face.landmarks)
            warn = "   LOW LIGHT: tracking is poor" if light < LOW_LIGHT else ""
            lines.append(f"light    {light:3.0f}/255{warn}")
            lines.append(f"bs_x     {g.get('bs_x'):+.3f}    bs_y    {g.get('bs_y'):+.3f}")
            lines.append(f"gaze_x   {g.get('gaze_x'):+.3f}    gaze_y  {g.get('gaze_y'):+.3f}")
            lines.append(
                f"eyes     open {g.eyes[0].openness:.2f} / {g.eyes[1].openness:.2f}"
                f"   {'BLINK' if g.blinking else ''}"
            )
            hd = r.face.head
            if hd:
                lines.append(f"head     yaw {hd.yaw:+5.1f}°  pitch {hd.pitch:+5.1f}°  roll {hd.roll:+5.1f}°")
                lines.append(f"distance {abs(hd.position[2]):4.0f} cm (estimate)")
            lines.append("")
            for name in GESTURES:
                v = r.face.blendshapes.get(name, 0.0)
                lines.append(f"{name:<17}{'█' * round(v * 20):<20} {v:.2f}")
        lines += ["", self.latency.format_table()]
        return "\n".join(lines)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif key == Qt.Key.Key_C and isinstance(self.mapper, UncalibratedMapper):
            self.mapper.recenter()
            self.pad.trail.clear()
        elif key == Qt.Key.Key_M:
            self.view.mirror = not self.view.mirror
            self.zoom.mirror = self.view.mirror

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.worker.stop()
        self.pipeline.close()
        super().closeEvent(event)


def run_preview(cfg: Config, seconds: float | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    pipeline = Pipeline.from_config(cfg)
    window = PreviewWindow(cfg, pipeline, seconds)
    window.show()
    app.exec()
    print(f"frames shown: {window.frames}, with a face: {window.faces}")
    print(window.latency.format_table())
    return 0
