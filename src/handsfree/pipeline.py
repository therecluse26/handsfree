"""The per-frame pipeline: camera -> face tracker -> gaze estimator."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .camera import FrameSource, OpenCVCamera
from .config import Config
from .face import FaceTracker, MediaPipeFaceTracker
from .gaze import GazeEstimator, LandmarkGazeEstimator
from .types import FaceObservation, Frame, GazeFeatures


@dataclass(frozen=True, slots=True)
class PipelineResult:
    frame: Frame
    face: FaceObservation | None
    gaze: GazeFeatures | None
    marks: dict[str, float]
    """Timestamps (monotonic seconds) in pipeline order, for latency measurement."""


class Pipeline:
    def __init__(self, source: FrameSource, tracker: FaceTracker, estimator: GazeEstimator) -> None:
        self.source = source
        self.tracker = tracker
        self.estimator = estimator

    @classmethod
    def from_config(cls, cfg: Config) -> Pipeline:
        tracker = MediaPipeFaceTracker(cfg.face)  # fails early if the model is absent
        return cls(OpenCVCamera(cfg.camera), tracker, LandmarkGazeEstimator(cfg.gaze))

    def step(self) -> PipelineResult | None:
        frame = self.source.read()
        if frame is None:
            return None
        marks = {"capture": frame.t_capture, "received": frame.t_received, "start": time.monotonic()}
        face = self.tracker.process(frame)
        marks["face"] = time.monotonic()
        gaze = self.estimator.estimate(face) if face is not None else None
        marks["gaze"] = time.monotonic()
        return PipelineResult(frame=frame, face=face, gaze=gaze, marks=marks)

    def close(self) -> None:
        self.source.close()
        self.tracker.close()
