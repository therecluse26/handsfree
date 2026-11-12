"""Eye image normalisation: one standard view of an eye for every camera and distance.

A gaze model must work with different cameras, distances and positions of the
face in the image. So the model does not get the camera's pixels directly.
Each eye is warped to the image that a standard virtual camera sees:

- the virtual camera looks straight at the eye (the eye is in the image centre),
- its horizontal axis is parallel to the line between the two eyes (no roll),
- the eye has a fixed scale (``PX_PER_MM`` pixels for each millimetre).

The warp is a homography, W = Kn · S · R · K⁻¹ (Zhang et al. 2018, "Revisiting
data normalization for appearance-based gaze estimation"). The gaze direction
is rotated into the virtual camera with the same R. So the model learns one
relation, from the standard eye image to the gaze angles in the virtual
camera. Real frames and synthetic renders use this same code.

Camera space is the handsfree convention: x to the right of the image, y up,
the person at negative z, millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = (96, 64)
"""(width, height) of the normalised eye image."""
PX_PER_MM = 2.0
"""Scale of the normalised eye image. The image covers 48 x 32 mm."""
_DISTANCE = 600.0  # of the virtual camera, mm. Only its ratio to the focal length matters.

_FLIP = np.diag([1.0, -1.0, -1.0])  # handsfree camera space <-> image space (x right, y down, z forwards)


@dataclass(frozen=True)
class EyeView:
    """The normalised view of one eye."""

    warp: np.ndarray
    """3 x 3 homography from camera-image pixels to normalised-image pixels."""
    rotation: np.ndarray
    """3 x 3 rotation from camera space to the virtual camera's space (both in the handsfree convention)."""
    eye_mm: np.ndarray
    """The eye position that the view was made for, camera space."""

    def image(self, source: np.ndarray, origin: tuple[float, float] = (0.0, 0.0), grey: bool = True) -> np.ndarray:
        """Warp ``source`` to the normalised image. ``origin`` is the camera-pixel position of the source's top-left pixel."""
        import cv2

        shift = np.array([[1.0, 0.0, origin[0]], [0.0, 1.0, origin[1]], [0.0, 0.0, 1.0]])
        out = cv2.warpPerspective(source, self.warp @ shift, SIZE, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        if grey and out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return out

    def angles(self, direction_cam: np.ndarray) -> np.ndarray:
        """A gaze direction in camera space as (yaw, pitch) degrees in the virtual camera.

        Yaw is positive towards the image's right side. Pitch is positive upwards.
        (0, 0) is a look straight into the camera.
        """
        g = self.rotation @ (np.asarray(direction_cam, float) / np.linalg.norm(direction_cam))
        return np.degrees([np.arctan2(g[0], g[2]), np.arcsin(np.clip(g[1], -1.0, 1.0))])

    def direction(self, yaw_pitch_deg: np.ndarray) -> np.ndarray:
        """The inverse of ``angles``: a unit gaze direction in camera space."""
        yaw, pitch = np.radians(yaw_pitch_deg)
        g = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
        return self.rotation.T @ g


def eye_view(eye_mm: np.ndarray, other_eye_mm: np.ndarray, focal_px: float, image_size: tuple[int, int], is_image_left: bool) -> EyeView:
    """The normalised view of the eye at ``eye_mm``.

    ``other_eye_mm`` is the person's other eye; the line between the two sets the roll.
    ``focal_px`` and ``image_size`` (width, height) describe the real camera.
    """
    e = np.asarray(eye_mm, float)
    o = np.asarray(other_eye_mm, float)
    distance = np.linalg.norm(e)
    right = (o - e) if is_image_left else (e - o)  # points to the image's right side
    # Image space: x right, y down, z forwards.
    z = _FLIP @ e / distance
    y = np.cross(z, _FLIP @ right)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    r_img = np.stack([x, y, z])
    k = np.array([[focal_px, 0.0, image_size[0] / 2], [0.0, focal_px, image_size[1] / 2], [0.0, 0.0, 1.0]])
    focal_n = PX_PER_MM * _DISTANCE
    kn = np.array([[focal_n, 0.0, SIZE[0] / 2], [0.0, focal_n, SIZE[1] / 2], [0.0, 0.0, 1.0]])
    scale = np.diag([1.0, 1.0, _DISTANCE / distance])
    warp = kn @ scale @ r_img @ np.linalg.inv(k)
    return EyeView(warp, _FLIP @ r_img @ _FLIP, e)


def backproject(pixel: tuple[float, float], depth_mm: float, focal_px: float, image_size: tuple[int, int]) -> np.ndarray:
    """The camera-space point at ``pixel`` whose distance along the camera axis is ``depth_mm``."""
    x = (pixel[0] - image_size[0] / 2) / focal_px * depth_mm
    y = -(pixel[1] - image_size[1] / 2) / focal_px * depth_mm
    return np.array([x, y, -depth_mm])


def to_model_input(images: np.ndarray, flip: np.ndarray | None = None) -> np.ndarray:
    """Grey normalised images (N x H x W uint8) to the model input (N x 1 x H x W float32).

    Each image gets zero mean and unit variance, so exposure and contrast do not matter.
    ``flip`` marks the images to mirror: the model sees every eye as an image-left eye.
    """
    x = images.astype(np.float32)
    if flip is not None:
        x = np.where(np.asarray(flip, bool)[:, None, None], x[:, :, ::-1], x)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return ((x - mean) / np.maximum(std, 1.0))[:, None]
