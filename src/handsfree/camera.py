"""Camera input.

``FrameSource`` is the interface. ``OpenCVCamera`` is the V4L2 webcam
implementation. Frames stay in memory; nothing is written to disk or sent
over the network.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Protocol

from .config import CameraConfig
from .types import Frame


class CameraError(RuntimeError):
    pass


class FrameSource(Protocol):
    def read(self) -> Frame | None:
        """Block until the next frame. Return None if no frame came."""
        ...

    def close(self) -> None: ...


class OpenCVCamera:
    def __init__(self, cfg: CameraConfig) -> None:
        import cv2

        self._cv2 = cv2
        self.cfg = cfg
        device: str | int = int(cfg.device) if cfg.device.isdigit() else cfg.device
        if cfg.fixed_framerate and isinstance(device, str):
            _set_fixed_framerate(device)
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera {cfg.device!r}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        # A reader thread takes every frame from the driver and keeps only the
        # newest one. The driver queue stays empty, so the camera never drops
        # frames, and the consumer never gets a stale frame.
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._reader, daemon=True, name="handsfree-camera")
        self._thread.start()

    def _reader(self) -> None:
        seq = 0
        while not self._closed:
            ok, image = self._cap.read()
            t_received = time.monotonic()
            if not ok or image is None:
                time.sleep(0.005)
                continue
            # V4L2 stamps each buffer on the monotonic clock. Use that stamp
            # when it is plausible; it includes the time spent in the driver.
            t_capture = self._cap.get(self._cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not 0.0 <= t_received - t_capture < 1.0:
                t_capture = t_received
            seq += 1
            with self._cond:
                self._latest = Frame(image=image, seq=seq, t_capture=t_capture, t_received=t_received)
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Wait for a frame newer than the last one returned. None on timeout."""
        with self._cond:
            if self._latest is None:
                self._cond.wait_for(lambda: self._latest is not None or self._closed, timeout)
            frame, self._latest = self._latest, None
            return frame

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=2.0)
        self._cap.release()


def _set_fixed_framerate(device: str) -> None:
    """Tell the camera not to lower its frame rate in dim light. Best effort."""
    tool = shutil.which("v4l2-ctl")
    if not tool:
        return
    subprocess.run(
        [tool, "-d", device, "-c", "exposure_dynamic_framerate=0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5,
    )
