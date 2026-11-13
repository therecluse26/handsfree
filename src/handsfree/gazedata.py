"""Examples for the eye-image gaze model: normalised eye images with gaze angles.

Synthetic sets give training examples; recordings give test examples. Both go
through the same normalisation (eyenorm), so the model sees the same kind of
image in training and in use.

One example is one eye: a 96 x 64 grey image and the gaze angles (yaw, pitch)
in the eye's virtual camera. Eyes on the image's right side are mirrored, and
their yaw changes sign, so the model learns one eye type.

An example also has the head direction in the same virtual camera. The eye image
alone shows the head direction badly, and the same eye image means another gaze
when the head is turned. A model can get the head direction as a second input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import eyenorm
from .benchmark import CANONICAL_CM, EYES, MEDIAPIPE_VFOV_DEG, DeviceGeometry
from .camsim import CameraSim, Tone
from .recording import Recording
from .synth import SynthSet


@dataclass
class Examples:
    images: np.ndarray
    """N x 64 x 96 uint8, grey, already mirrored for image-right eyes."""
    angles: np.ndarray
    """N x 2 float32: (yaw, pitch) degrees in the virtual camera, yaw already mirrored."""
    group: np.ndarray
    """N int: the person (synthetic) or the frame (real). Used to split and to pair the eyes."""
    views: list[eyenorm.EyeView]
    is_right: np.ndarray
    """N bool: the eye is on the image's right side (it was mirrored)."""
    head: np.ndarray
    """N x 2 float32: the head direction as (yaw, pitch) degrees in the virtual camera, yaw mirrored as the gaze.

    (0, 0) is a head that points at the camera. Synthetic examples have the exact direction.
    Real examples have ``tracker_head``: the face tracker's measurement.
    """

    def model_input(self) -> np.ndarray:
        return eyenorm.to_model_input(self.images)


def _mirror(image: np.ndarray, angles: np.ndarray, is_right: bool) -> tuple[np.ndarray, np.ndarray]:
    return (image[:, ::-1], angles * [-1.0, 1.0]) if is_right else (image, angles)


HEAD_FIT = ((0.05, 1.06, -0.56), (4.94, 1.07, -0.43))
"""For yaw and for pitch: (constant, x tracker angle, x eye position angle), see ``tracker_head``."""
HEAD_ERROR_DEG = (3.2, 3.6)
"""Error of ``tracker_head`` (yaw, pitch; standard deviation). The training adds it to the exact head direction."""


def tracker_head(view: eyenorm.EyeView, matrix: np.ndarray, is_right: bool) -> np.ndarray:
    """The head direction (yaw, pitch degrees in the eye's virtual camera) from the face tracker's transform.

    The tracker's direction is not the direction of the synthetic persons' labels: it has another
    zero for the pitch, and it changes with the position of the face in the image. The numbers of
    ``HEAD_FIT`` come from 298 synthetic faces in full camera frames (fields of view 36 to 70
    degrees), where the exact direction is known. After the fit the two agree to ``HEAD_ERROR_DEG``.
    """
    tracker = view.angles(np.asarray(matrix, float)[:3, 2])
    e = view.eye_mm
    position = np.degrees([np.arctan2(e[0], -e[2]), np.arctan2(e[1], -e[2])])
    out = np.array([c + a * t + b * p for (c, a, b), t, p in zip(HEAD_FIT, tracker, position)])
    return out * [-1.0, 1.0] if is_right else out


def synth_examples(s: SynthSet, indices, rng: np.random.Generator, augment: bool = True, copies: int = 1) -> Examples:
    """Examples from a synthetic set. With ``augment``, each copy gets another simulated camera and tracker error."""
    import cv2

    images, angles, group, views, right, head = [], [], [], [], [], []
    for i in indices:
        label = s.labels[i]
        cam = label["camera"]
        x0, y0 = label["rect"][:2]
        linear = s.linear(i)
        person = hash(label["image"].rsplit("_", 1)[0]) & 0x7FFFFFFF
        for _ in range(copies):
            if augment:
                img = CameraSim.random(rng).apply(Tone.random(rng).apply(linear), rng, grid_offset=(x0, y0))
                res = float(rng.uniform(0.45, 1.0)) if rng.random() < 0.5 else 1.0  # a camera with fewer pixels
            else:
                img, res = Tone().apply(linear), 1.0
            if res < 1.0:
                img = cv2.resize(img, None, fx=res, fy=res, interpolation=cv2.INTER_AREA)
            focal, size = cam["focal_px"] * res, (cam["width"] * res, cam["height"] * res)
            eyes = [np.array(e["tracker_centre_cam_mm"], float) for e in label["eyes"]]
            if augment:
                # The face tracker's errors: the eye point is a little off, and the distance is wrong by some percent.
                depth = rng.uniform(0.9, 1.1)
                eyes = [(e + rng.uniform(-1.5, 1.5, 3) * [1, 1, 0]) * depth * rng.uniform(0.98, 1.02) for e in eyes]
            for k, e in enumerate(label["eyes"]):
                view = eyenorm.eye_view(eyes[k], eyes[1 - k], focal, size, is_image_left=(k == 0))
                im, an = _mirror(view.image(img, origin=(x0 * res, y0 * res)), view.angles(e["sight_cam"]), k == 1)
                images.append(im)
                angles.append(an)
                group.append(person)
                views.append(view)
                right.append(k == 1)
                forward = label.get("head_forward_cam")  # sets from before the head label have none
                head.append(view.angles(forward) * ([-1.0, 1.0] if k == 1 else [1.0, 1.0]) if forward else (np.nan, np.nan))
    return Examples(np.stack(images), np.array(angles, np.float32), np.array(group), views, np.array(right), np.array(head, np.float32))


def eye_positions(rec: Recording, geom: DeviceGeometry) -> tuple[np.ndarray, float]:
    """Eye positions of every frame (N x 2 x 3, camera space, mm) and the camera's focal length in pixels.

    The pixel position is the face tracker's eye point. The distance comes from the
    tracker's face transform, corrected for the camera's real field of view.
    """
    f = rec.frames
    size = tuple(rec.meta["image_size"])
    vfov = geom.vertical_fov_deg or MEDIAPIPE_VFOV_DEG
    focal = (size[1] / 2) / np.tan(np.radians(vfov) / 2)
    mat = f["matrix"].astype(float)
    out = np.zeros((len(rec), 2, 3))
    for k, eye in enumerate(EYES):
        a, b = eye["corners"]
        canon = (np.array(CANONICAL_CM[a]) + np.array(CANONICAL_CM[b])) / 2
        depth = -(np.einsum("nj,j->n", mat[:, 2, :3], canon) + mat[:, 2, 3]) * 10.0 * geom.depth_scale
        px = f["eye_center"][:, k].astype(float)
        out[:, k, 0] = (px[:, 0] - size[0] / 2) / focal * depth
        out[:, k, 1] = -(px[:, 1] - size[1] / 2) / focal * depth
        out[:, k, 2] = -depth
    return out, focal


def recording_examples(rec: Recording, geom: DeviceGeometry, indices=None) -> Examples:
    """Examples from a recording. The angles are the directions from each eye to the target."""
    f = rec.frames
    size = tuple(rec.meta["image_size"])
    eyes_mm, focal = eye_positions(rec, geom)
    crops = rec.eyes()
    target = geom.to_camera(f["target_xy"] * np.array(geom.screen_mm))
    images, angles, group, views, right, head = [], [], [], [], [], []
    for i in range(len(rec)) if indices is None else indices:
        for k in (0, 1):
            view = eyenorm.eye_view(eyes_mm[i, k], eyes_mm[i, 1 - k], focal, size, is_image_left=(k == 0))
            im, an = _mirror(view.image(crops[i, k], origin=tuple(f["eye_origin"][i, k])), view.angles(target[i] - eyes_mm[i, k]), k == 1)
            images.append(im)
            angles.append(an)
            group.append(i)
            views.append(view)
            right.append(k == 1)
            head.append(tracker_head(view, f["matrix"][i], k == 1))
    return Examples(np.stack(images), np.array(angles, np.float32), np.array(group), views, np.array(right), np.array(head, np.float32))
