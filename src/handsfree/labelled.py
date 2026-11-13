"""A labelled set: laptop-webcam images of real persons with known gaze targets, for measurements only.

The reference photos (reference.py) show persons who look into the camera. They
give the model's constant error for many persons, but not its gain: how much of
a real eye movement the model follows. For that the true gaze must change.

MPIIFaceGaze has it: 15 persons at their own laptops, about 3000 webcam frames
each, with the gaze target on the screen as a 3D point in the camera's space.
That is the situation of this program.

MPIIFaceGaze (Zhang et al. 2017, "It's Written All Over Your Face: Full-Face
Appearance-Based Gaze Estimation") is licensed CC BY-NC-SA 4.0, and its page
says: "The data is only to be used for non-commercial scientific purposes."
It is used to measure. The gaze model does not train on it, and it is not part
of this repository. The zip file is read in place; nothing is unpacked.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import eyenorm
from .gaze import IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE
from .gazedata import tracker_head
from .models import ssl_context
from .sessions import data_dir

URL = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"
"""942 MB."""
PER_PERSON = 300
"""Frames of each person in a measurement, spread over the person's days."""


def labelled_dir() -> Path:
    return data_dir() / "reference" / "mpiifacegaze"


def archive() -> Path:
    return labelled_dir() / "MPIIFaceGaze.zip"


def fetch(log=print) -> None:
    """Download the zip file when it is not there yet. A broken download continues where it stopped."""
    labelled_dir().mkdir(parents=True, exist_ok=True)
    if archive().exists():
        return
    part = archive().with_suffix(".part")
    done = part.stat().st_size if part.exists() else 0
    log(f"downloading MPIIFaceGaze.zip (942 MB){f', from byte {done}' if done else ''} …")
    headers = {"User-Agent": "handsfree"} | ({"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), context=ssl_context(), timeout=120) as r:
        if done and r.status != 206:  # the server sent the whole file again
            done = 0
        with open(part, "ab" if done else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(part) as z:  # a cut-off file has no directory and fails here
        z.namelist()
    part.rename(archive())


def rows(z: zipfile.ZipFile) -> dict[str, list[tuple[str, np.ndarray]]]:
    """The annotation of each person: (image member name, 27 numbers) for each frame.

    The numbers are, in the file's order: gaze point on the screen in pixels (2), six face points
    in the image (12), head rotation and translation (6), face centre (3) and gaze target (3) in
    the camera's space (x right, y down, z forwards, mm), and a last column that is not a number.
    """
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name in sorted(z.namelist()):
        parts = name.rsplit("/", 2)
        if len(parts) < 2 or not name.endswith(".txt") or parts[-1] != parts[-2] + ".txt":
            continue  # the annotation file is pNN/pNN.txt
        folder = name.rsplit("/", 1)[0]
        frames = []
        for line in z.read(name).decode("utf-8", "replace").splitlines():
            cells = line.split()
            if len(cells) >= 27:
                frames.append((f"{folder}/{cells[0]}", np.array(cells[1:27], float)))
        out[parts[-2]] = frames
    return out


def camera(frames: list[tuple[str, np.ndarray]]) -> tuple[float, float, float, float]:
    """The camera's focal lengths and centre (fx, fy, cx, cy; pixels) from a person's annotation.

    The face centre is given in 3D and the face points in the image. Their relation is the
    camera's projection, so the calibration file is not needed.
    """
    a = np.array([f for _, f in frames])
    u, v = a[:, 2:14:2].mean(axis=1), a[:, 3:14:2].mean(axis=1)
    x, y, zc = a[:, 20], a[:, 21], a[:, 22]
    fx, cx = np.polyfit(x / zc, u, 1)
    fy, cy = np.polyfit(y / zc, v, 1)
    return float(fx), float(fy), float(cx), float(cy)


@dataclass
class LabelledEyes:
    images: np.ndarray
    """N x 64 x 96 uint8: normalised grey eye images, image-right eyes mirrored. Two for each frame."""
    person: np.ndarray
    """N int: the person of each image (0 is p00)."""
    angles: np.ndarray
    """N x 2: the true gaze (yaw, pitch degrees) in the eye's virtual camera, yaw mirrored as the model's."""
    head: np.ndarray
    """N x 2: the head direction as the gaze model gets it (``gazedata.tracker_head``)."""
    depth_ratio: np.ndarray
    """N: this program's distance of the face / the data set's distance. 1.0 is correct."""

    def save(self, path: Path) -> None:
        np.savez_compressed(path, images=self.images, person=self.person, angles=self.angles, head=self.head, depth_ratio=self.depth_ratio)

    @classmethod
    def load(cls, path: Path) -> LabelledEyes:
        with np.load(path) as d:
            return cls(d["images"], d["person"], d["angles"], d["head"], d["depth_ratio"])


def build(per_person: int = PER_PERSON, log=print) -> LabelledEyes:
    """Normalised eye images with the true gaze, made as the program makes them from a camera frame.

    The face tracker finds the eyes and the distance. The camera's field of view comes from the
    data set. The image centre is taken as the camera's centre, as the program does.
    """
    import cv2

    from .reference import _landmarker, eye_views

    mp, landmarker = _landmarker()
    images, person, angles, head, ratio = [], [], [], [], []
    with zipfile.ZipFile(archive()) as z:
        for p, (name, frames) in enumerate(rows(z).items()):
            _, fy, _, _ = camera(frames)
            used = 0
            for i in np.unique(np.linspace(0, len(frames) - 1, per_person).round().astype(int)):
                member, a = frames[i]
                img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                size = (img.shape[1], img.shape[0])
                result = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                if not result.face_landmarks or result.facial_transformation_matrixes is None or not len(result.facial_transformation_matrixes):
                    continue
                landmarks = np.array([(q.x, q.y, q.z) for q in result.face_landmarks[0]], dtype=np.float32)
                matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
                vfov = 2 * np.degrees(np.arctan(size[1] / 2 / fy))
                views, _ = eye_views(landmarks, matrix, size, vfov)
                target = a[23:26] * [1.0, -1.0, -1.0]  # to this program's camera space: y up, the person at negative z
                for k in (0, 1):
                    grey = views[k].image(img)
                    true = views[k].angles(target - views[k].eye_mm)
                    images.append(grey[:, ::-1] if k == 1 else grey)
                    angles.append(true * [-1.0, 1.0] if k == 1 else true)
                    head.append(tracker_head(views[k], matrix, k == 1))
                    person.append(p)
                    ratio.append(-views[k].eye_mm[2] / a[22])
                used += 1
            log(f"  {name}: {used} frames, vertical field of view {2 * np.degrees(np.arctan(size[1] / 2 / fy)):.0f} deg")
    landmarker.close()
    return LabelledEyes(np.stack(images), np.array(person), np.array(angles), np.array(head), np.array(ratio))


def cached(per_person: int = PER_PERSON, log=print) -> LabelledEyes:
    path = labelled_dir() / f"eyes-{per_person}.npz"
    if path.exists():
        return LabelledEyes.load(path)
    lab = build(per_person, log=log)
    lab.save(path)
    return lab


def frame_gaze(angles: np.ndarray) -> np.ndarray:
    """The mean of the two eyes of each frame (F x 2: yaw, pitch). Yaw is positive towards the image's right side."""
    a = np.asarray(angles, float).reshape(-1, 2, 2)
    return np.stack([(a[:, 0, 0] - a[:, 1, 0]) / 2, a[:, :, 1].mean(axis=1)], axis=1)


def predict(lab: LabelledEyes, model: str = "") -> np.ndarray:
    """The eye-image model's gaze for each eye image (N x 2 degrees, the model's frame)."""
    from .benchmark import EyeImageModel

    return EyeImageModel(model=model).angles(lab.images, lab.head).astype(float)


def format_gaze(lab: LabelledEyes, pred: np.ndarray) -> str:
    """A table: for each person the error, the constant error and the gain of the model, with no calibration."""
    from .gazetrain import angular_error_deg

    true_f, pred_f, who = frame_gaze(lab.angles), frame_gaze(pred), lab.person[::2]
    lines = [
        "Each person looks at points on a laptop screen. error: angle between the model's gaze and the true gaze,",
        "mean of the two eyes, no calibration. bias: the model's constant error for this person (model - true).",
        "gain: the model's movement / the true movement, 1.00 is correct. rest: the error that stays when the",
        "person's own bias and gain are removed (a measure of the noise).",
        "",
        f"{'person':<8}{'frames':>7}{'true pitch':>11}{'error':>7}{'bias yaw':>10}{'bias pitch':>11}{'gain yaw':>10}{'gain pitch':>11}{'rest':>7}{'distance':>10}",
    ]
    stats = []
    for p in np.unique(who):
        m = who == p
        t, q = true_f[m], pred_f[m]
        err = float(angular_error_deg(q, t).mean())
        bias = (q - t).mean(axis=0)
        fit = [np.polyfit(t[:, j], q[:, j], 1) for j in (0, 1)]
        rest = np.stack([q[:, j] - np.polyval(fit[j], t[:, j]) for j in (0, 1)], axis=1)
        row = [int(m.sum()), float(t[:, 1].mean()), err, float(bias[0]), float(bias[1]), float(fit[0][0]), float(fit[1][0]), float(np.linalg.norm(rest, axis=1).mean()), float(np.median(lab.depth_ratio[::2][m]))]
        stats.append(row)
        lines.append(f"p{p:02d}     {row[0]:>7d}{row[1]:>11.1f}{row[2]:>7.1f}{row[3]:>+10.1f}{row[4]:>+11.1f}{row[5]:>10.2f}{row[6]:>11.2f}{row[7]:>7.1f}{row[8]:>10.2f}")
    s = np.array(stats)
    lines.append(f"{'mean':<8}{int(s[:, 0].sum()):>7d}{s[:, 1].mean():>11.1f}{s[:, 2].mean():>7.1f}{s[:, 3].mean():>+10.1f}{s[:, 4].mean():>+11.1f}{s[:, 5].mean():>10.2f}{s[:, 6].mean():>11.2f}{s[:, 7].mean():>7.1f}{s[:, 8].mean():>10.2f}")
    lines.append(f"{'spread':<8}{'':>7}{'':>11}{s[:, 2].std():>7.1f}{s[:, 3].std():>10.1f}{s[:, 4].std():>11.1f}{s[:, 5].std():>10.2f}{s[:, 6].std():>11.2f}")
    lines.append("")
    lines.append("distance: this program's distance of the face / the data set's distance. spread: between the persons.")
    return "\n".join(lines)
