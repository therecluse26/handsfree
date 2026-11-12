"""Gaze mapping: gaze features in, screen point out.

A screen point is (x, y) with both values in 0..1. (0, 0) is the top-left
corner of the screen. Calibration (a later step) adds a mapper that is fitted
to the user. ``UncalibratedMapper`` exists for the live preview only.
"""

from __future__ import annotations

from typing import Protocol

from .types import GazeFeatures


class GazeMapper(Protocol):
    def map(self, features: GazeFeatures) -> tuple[float, float] | None:
        """Return the screen point, or None if there is no usable estimate."""
        ...


class UncalibratedMapper:
    """Iris offset from a centre reference, times a fixed gain. Eyes only."""

    def __init__(self, gain_x: float, gain_y: float) -> None:
        self.gain_x = gain_x
        self.gain_y = gain_y
        self._center: tuple[float, float] | None = None
        self._recenter = True

    def recenter(self) -> None:
        """Use the next open-eye sample as "the user looks at the screen centre"."""
        self._recenter = True

    def map(self, features: GazeFeatures) -> tuple[float, float] | None:
        if features.blinking:
            return None
        gx, gy = features.get("gaze_x"), features.get("gaze_y")
        if self._recenter or self._center is None:
            self._center = (gx, gy)
            self._recenter = False
        x = 0.5 + self.gain_x * (gx - self._center[0])
        y = 0.5 + self.gain_y * (gy - self._center[1])
        return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))
