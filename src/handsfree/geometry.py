"""Small geometry helpers with no heavy imports."""

from __future__ import annotations

import math

import numpy as np

from .types import HeadPose


def head_pose_from_matrix(m: np.ndarray) -> HeadPose:
    """Convert a 4x4 face transform (MediaPipe convention) to a HeadPose.

    The rotation is read as R = Ry(yaw) @ Rx(pitch) @ Rz(roll). The result
    does not change if the matrix also contains a uniform scale.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.atan2(-r[1, 2], math.hypot(r[1, 0], r[1, 1]))
    roll = math.atan2(r[1, 0], r[1, 1])
    t = np.asarray(m, dtype=float)[:3, 3]
    return HeadPose(
        yaw=math.degrees(yaw),
        pitch=math.degrees(pitch),
        roll=math.degrees(roll),
        position=(float(t[0]), float(t[1]), float(t[2])),
    )
