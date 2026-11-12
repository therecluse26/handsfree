"""Gaze estimation: face observation in, gaze features out.

``GazeEstimator`` is the interface. ``LandmarkGazeEstimator`` measures where
the iris sits between the eye corners, and adds the head pose. A mapper
(see ``mapping.py``) turns these features into a screen point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import GazeConfig
from .types import EyeState, FaceObservation, GazeFeatures


class GazeEstimator(Protocol):
    def estimate(self, face: FaceObservation) -> GazeFeatures: ...


@dataclass(frozen=True, slots=True)
class EyeIndices:
    """MediaPipe landmark numbers for one eye."""

    corner_a: int
    """The corner nearer the left edge of the un-mirrored image."""
    corner_b: int
    upper: int
    lower: int
    iris: int


# In the un-mirrored camera image the user's right eye is on the left side.
IMAGE_LEFT_EYE = EyeIndices(corner_a=33, corner_b=133, upper=159, lower=145, iris=468)
IMAGE_RIGHT_EYE = EyeIndices(corner_a=362, corner_b=263, upper=386, lower=374, iris=473)

FEATURE_NAMES = (
    "gaze_x",
    "gaze_y",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_x",
    "head_y",
    "head_z",
)


def eye_state(landmarks: np.ndarray, image_size: tuple[int, int], idx: EyeIndices) -> EyeState:
    size = np.array(image_size, dtype=float)
    a = landmarks[idx.corner_a, :2] * size
    b = landmarks[idx.corner_b, :2] * size
    iris = landmarks[idx.iris, :2] * size
    upper = landmarks[idx.upper, :2] * size
    lower = landmarks[idx.lower, :2] * size
    center = (a + b) / 2.0
    width = float(np.linalg.norm(b - a)) or 1.0
    x_axis = (b - a) / width
    y_axis = np.array([-x_axis[1], x_axis[0]])  # points down when the eye is level
    d = iris - center
    return EyeState(
        h=float(d @ x_axis) / width,
        v=float(d @ y_axis) / width,
        openness=float(abs((lower - upper) @ y_axis)) / width,
        iris_px=(float(iris[0]), float(iris[1])),
        center_px=(float(center[0]), float(center[1])),
        width_px=width,
    )


class LandmarkGazeEstimator:
    def __init__(self, cfg: GazeConfig | None = None) -> None:
        self.cfg = cfg or GazeConfig()

    def estimate(self, face: FaceObservation) -> GazeFeatures:
        left = eye_state(face.landmarks, face.image_size, IMAGE_LEFT_EYE)
        right = eye_state(face.landmarks, face.image_size, IMAGE_RIGHT_EYE)
        # The camera faces the user. A look to the right edge of the screen
        # moves the iris to the left in the image, so the sign is flipped.
        gaze_x = -(left.h + right.h) / 2.0
        gaze_y = (left.v + right.v) / 2.0
        head = face.head
        pose = (
            (head.yaw, head.pitch, head.roll, *head.position) if head else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        blinking = (left.openness + right.openness) / 2.0 < self.cfg.blink_openness
        return GazeFeatures(
            t_capture=face.t_capture,
            vector=np.array([gaze_x, gaze_y, *pose], dtype=float),
            names=FEATURE_NAMES,
            eyes=(left, right),
            blinking=blinking,
        )
