"""Data types that flow between the components.

Each component takes one of these and returns the next one:

    FrameSource -> Frame -> FaceTracker -> FaceObservation
                -> GazeEstimator -> GazeFeatures -> GazeMapper -> screen point

All timestamps are seconds on the ``time.monotonic()`` clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """One camera image."""

    image: np.ndarray
    """H x W x 3, uint8, BGR channel order (OpenCV order)."""
    seq: int
    """Frame number from the source. It increases by 1 per frame read."""
    t_capture: float
    """When the camera driver captured the frame. Falls back to ``t_received``."""
    t_received: float
    """When our code received the frame from the driver."""


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Head rotation and position in camera space."""

    yaw: float
    """Degrees. Rotation around the vertical axis."""
    pitch: float
    """Degrees. Rotation around the horizontal axis."""
    roll: float
    """Degrees. Rotation around the camera's viewing axis."""
    position: tuple[float, float, float]
    """Centimetres, camera space (x right, y up, z towards the camera)."""


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """What the face tracker found in one frame."""

    t_capture: float
    image_size: tuple[int, int]
    """(width, height) of the source frame in pixels."""
    landmarks: np.ndarray
    """N x 3 float array. x and y are normalised to 0..1 of the image; z is relative depth."""
    blendshapes: dict[str, float]
    """Face-gesture scores, 0..1, keyed by name (for example ``browInnerUp``)."""
    head: HeadPose | None


@dataclass(frozen=True, slots=True)
class EyeState:
    """Iris position inside one eye, in the eye's own frame.

    The eye frame has its origin at the midpoint of the two eye corners. Its
    x axis runs from the image-left corner to the image-right corner. Its y axis
    points down in the image. Distances are in eye widths (corner to corner).
    """

    h: float
    """Iris offset along the eye's x axis, in eye widths (image direction)."""
    v: float
    """Iris offset along the eye's y axis (down is positive), in eye widths."""
    openness: float
    """Lid gap divided by eye width. About 0.25-0.35 when open, near 0 when closed."""
    iris_px: tuple[float, float]
    iris_radius_px: float
    center_px: tuple[float, float]
    corner_a_px: tuple[float, float]
    """The eye corner nearer the left edge of the un-mirrored image."""
    corner_b_px: tuple[float, float]
    width_px: float


@dataclass(frozen=True, slots=True)
class GazeFeatures:
    """Gaze-related measurements from one frame.

    ``vector`` is the input for a gaze mapper (calibration). Its sign
    convention is the user's view: ``gaze_x`` grows when the user looks
    towards the right edge of the screen, ``gaze_y`` grows when the user
    looks down.
    """

    t_capture: float
    vector: np.ndarray
    names: tuple[str, ...]
    eyes: tuple[EyeState, EyeState]
    """(image-left eye, image-right eye) in the un-mirrored camera image."""
    blinking: bool

    def get(self, name: str) -> float:
        return float(self.vector[self.names.index(name)])


@dataclass(frozen=True, slots=True)
class Target:
    """A point on the screen that the user looks at."""

    name: str
    x: float
    """0..1 across the screen, left to right."""
    y: float
    """0..1 down the screen, top to bottom."""
    kind: str = "fixation"
    """"fixation": look at it, head still. "warmup": shown, but its samples are not used.
    "head": look at it and move the head as the prompt says."""
    record_s: float | None = None
    """Recording time for this target. None means the session default."""
    prompt: str = ""
    """Instruction shown above the target."""
