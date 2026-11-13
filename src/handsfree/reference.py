"""Reference images of many real persons, for measurements only.

One test person is not enough to judge a model for everyone. Portrait photos
of thousands of persons give two measurements with no gaze recording:

1. **Realism.** A classifier tries to tell normalised eye images of real
   persons from synthetic ones. Where it succeeds, the synthetic eyes differ.
2. **Constant gaze error.** In a portrait most persons look into the camera.
   So the true gaze is known: it points at the camera, which is (0, 0) in the
   normalised eye view. The model's answer for each person is its constant
   error for that person.

The photos are FFHQ (Flickr-Faces-HQ, NVIDIA; CC BY-NC-SA 4.0, single photos
have their own licences). They are used to measure. The gaze model does not
train on them, and they are not part of this repository.
"""

from __future__ import annotations

import json
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import eyenorm
from .benchmark import CANONICAL_CM, EYES, MEDIAPIPE_VFOV_DEG
from .camsim import CameraSim, Tone
from .models import ssl_context
from .gaze import IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE, eye_state
from .sessions import data_dir

FFHQ_URL = "https://huggingface.co/datasets/gaunernst/ffhq-1024-wds/resolve/main/{name}"
FFHQ_SHARDS = ("00000.tar", "01000.tar", "02000.tar")
"""Three files with 1000 photos each, about 1 GB each."""
FEATURES_URL = "https://raw.githubusercontent.com/DCGM/ffhq-features-dataset/master/json/{id}.json"
"""Labels of the FFHQ photos (DCGM/ffhq-features-dataset). Only the glasses label is used."""
PORTRAIT_VFOV_DEG = 14.0
"""Assumed field of view of a portrait crop. A face fills the crop, and the camera is 1 to 2 m away."""


def reference_dir() -> Path:
    return data_dir() / "reference" / "ffhq"


def fetch(log=print) -> None:
    """Download the photo files that are not there yet."""
    reference_dir().mkdir(parents=True, exist_ok=True)
    for name in FFHQ_SHARDS:
        path = reference_dir() / name
        if path.exists():
            continue
        log(f"downloading {name} (about 1 GB) …")
        request = urllib.request.Request(FFHQ_URL.format(name=name), headers={"User-Agent": "handsfree"})
        part = path.with_suffix(".part")
        with urllib.request.urlopen(request, context=ssl_context(), timeout=120) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        part.rename(path)


def photos(limit: int | None = None) -> Iterator[tuple[str, np.ndarray]]:
    """(name, image as uint8 BGR) for each photo. The files are read in place; nothing is unpacked."""
    import cv2

    count = 0
    for shard in sorted(reference_dir().glob("*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                data = tar.extractfile(member).read()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                yield member.name, img
                count += 1
                if limit is not None and count >= limit:
                    return


@dataclass
class ReferenceEyes:
    images: np.ndarray
    """N x 64 x 96 uint8: normalised grey eye images, image-right eyes mirrored. Two for each photo."""
    name: list[str]
    """The photo of each image."""
    openness: np.ndarray
    """Lid gap / eye width from the face tracker."""
    head_yaw_deg: np.ndarray
    head_pitch_deg: np.ndarray
    iris: np.ndarray
    """N x 2: the tracker's iris point from the middle of the eye corners, / eye width. x towards the nose, y down."""

    def save(self, path: Path) -> None:
        np.savez_compressed(path, images=self.images, name=np.array(self.name), openness=self.openness,
                            head_yaw_deg=self.head_yaw_deg, head_pitch_deg=self.head_pitch_deg, iris=self.iris)  # fmt: skip

    @classmethod
    def load(cls, path: Path) -> ReferenceEyes:
        with np.load(path) as d:
            return cls(d["images"], d["name"].tolist(), d["openness"], d["head_yaw_deg"], d["head_pitch_deg"], d["iris"])


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    from .models import require_face_landmarker

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(require_face_landmarker(""))),
        running_mode=vision.RunningMode.IMAGE,  # single photos, no tracking between them
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def eye_views(landmarks: np.ndarray, matrix: np.ndarray, size: tuple[int, int], vfov_deg: float):
    """The two normalised eye views of a face in a photo whose camera is not known.

    The face tracker gives the eye size in the image; that fixes the scale. ``vfov_deg`` is
    an assumed field of view. It changes only the small perspective correction.
    """
    f_tracker = (size[1] / 2) / np.tan(np.radians(MEDIAPIPE_VFOV_DEG) / 2)
    focal = (size[1] / 2) / np.tan(np.radians(vfov_deg) / 2)
    eyes = []
    for k, (eye, idx) in enumerate(zip(EYES, (IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE))):
        a, b = eye["corners"]
        canon = (np.array(CANONICAL_CM[a]) + np.array(CANONICAL_CM[b])) / 2
        depth_tracker = -(matrix[2, :3] @ canon + matrix[2, 3]) * 10.0
        state = eye_state(landmarks, size, idx)
        eyes.append((eyenorm.backproject(state.center_px, depth_tracker * focal / f_tracker, focal, size), state))
    views = [eyenorm.eye_view(eyes[k][0], eyes[1 - k][0], focal, size, is_image_left=(k == 0)) for k in (0, 1)]
    return views, [e[1] for e in eyes]


def _eye_rect(landmarks: np.ndarray, size: tuple[float, float]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the image rectangle around the two eyes, with one eye width of margin."""
    pts = landmarks[[33, 133, 362, 263], :2] * np.array(size, float)
    margin = float(np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[3] - pts[2])) / 2
    x0, y0 = np.floor(pts.min(axis=0) - margin).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0) + margin).astype(int)
    return max(x0, 0), max(y0, 0), min(x1, int(size[0])), min(y1, int(size[1]))


def build(limit: int | None = None, seed: int = 0, webcam: bool = True, log=print) -> ReferenceEyes:
    """Normalised eye images of the reference photos. With ``webcam`` they get the webcam simulation.

    A photo is sharp and clean. A webcam frame is not. Without the simulation, a classifier
    would separate photos from synthetic webcam images by their noise alone. So each photo
    goes the same way as a synthetic render: a smaller image (as a webcam at 0.4 to 0.9 m
    sees the eye), then tone, blur, noise and JPEG, and then the normalisation.
    """
    import cv2

    mp, landmarker = _landmarker()
    rng = np.random.default_rng(seed)
    images, names, openness, yaw, pitch, iris = [], [], [], [], [], []
    for n, (name, img) in enumerate(photos(limit)):
        result = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        if (n + 1) % 500 == 0:
            log(f"  reference photos: {n + 1}")
        if not result.face_landmarks or result.facial_transformation_matrixes is None or not len(result.facial_transformation_matrixes):
            continue
        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        full = (img.shape[1], img.shape[0])
        scale = float(rng.uniform(0.3, 0.75)) if webcam else 1.0  # a portrait has about 4 px/mm, a webcam 1.2 to 3
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if webcam else img
        size = (full[0] * scale, full[1] * scale)
        x0, y0, x1, y1 = _eye_rect(landmarks, size)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        frame = small[y0:y1, x0:x1]
        if webcam:
            linear = (frame.astype(np.float32) / 255.0) ** 2.2
            frame = CameraSim.random(rng).apply(Tone.random(rng).apply(linear), rng, grid_offset=(x0, y0))
        views, states = eye_views(landmarks, matrix, size, PORTRAIT_VFOV_DEG)
        forward = matrix[:3, 2]
        for k in (0, 1):
            grey = views[k].image(frame, origin=(x0, y0))
            images.append(grey[:, ::-1] if k == 1 else grey)
            names.append(name)
            openness.append(states[k].openness)
            iris.append((states[k].h if k == 0 else -states[k].h, states[k].v))
            yaw.append(float(np.degrees(np.arctan2(forward[0], forward[2]))))
            pitch.append(float(np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0)))))
    landmarker.close()
    return ReferenceEyes(np.stack(images), names, np.array(openness), np.array(yaw), np.array(pitch), np.array(iris))


def cached(limit: int | None = None, log=print) -> ReferenceEyes:
    path = reference_dir() / f"eyes-{limit or 'all'}.npz"
    if path.exists():
        return ReferenceEyes.load(path)
    ref = build(limit, log=log)
    ref.save(path)
    return ref


def glasses_labels(names: list[str], log=print) -> dict[str, str]:
    """"NoGlasses", "ReadingGlasses" (ordinary glasses), "Sunglasses" or "SwimmingGoggles" for each photo.

    The labels come from the FFHQ features data set. Only the glasses label is kept.
    """
    from concurrent.futures import ThreadPoolExecutor

    path = reference_dir() / "glasses.json"
    known: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    missing = sorted({n for n in names if n not in known})
    if not missing:
        return known

    context = ssl_context()

    def one(name: str) -> tuple[str, str | None]:
        request = urllib.request.Request(FEATURES_URL.format(id=Path(name).stem), headers={"User-Agent": "handsfree"})
        try:
            with urllib.request.urlopen(request, context=context, timeout=60) as r:
                faces = json.loads(r.read())
            return name, str(faces[0]["faceAttributes"]["glasses"]) if faces else "unknown"
        except urllib.error.HTTPError as e:
            return name, "unknown" if e.code == 404 else None
        except (OSError, ValueError, KeyError):
            return name, None  # not saved: the next call tries again

    log(f"downloading the glasses labels of {len(missing)} photos …")
    with ThreadPoolExecutor(8) as pool:
        known.update((n, g) for n, g in pool.map(one, missing) if g is not None)
    path.write_text(json.dumps(known))
    return known


def portrait_gaze(ref: ReferenceEyes, model: str = "") -> np.ndarray:
    """The model's gaze angles for the two eyes of each photo: P x 2 x 2 (photo, eye, (yaw, pitch)) in degrees.

    A person who looks into the camera has (0, 0) for each eye. The angles are in the model's
    frame: yaw is positive towards the nose for both eyes.
    """
    from .benchmark import EyeImageModel

    return EyeImageModel(model=model).angles(ref.images).astype(float).reshape(-1, 2, 2)


def person_gaze(eyes: np.ndarray) -> np.ndarray:
    """The mean of a person's two eyes (P x 2: yaw, pitch). Yaw is positive towards the image's right side."""
    return np.stack([(eyes[:, 0, 0] - eyes[:, 1, 0]) / 2, eyes[:, :, 1].mean(axis=1)], axis=1)


def _spread(x: np.ndarray) -> float:
    """Half the range that holds the middle 68% of the values. For a normal distribution it is the sigma."""
    lo, hi = np.percentile(x, [16, 84])
    return float(hi - lo) / 2


def format_portrait_gaze(ref: ReferenceEyes, eyes: np.ndarray, glasses: dict[str, str]) -> str:
    gaze = person_gaze(eyes)
    names = ref.name[::2]
    label = np.array([glasses.get(n, "unknown") for n in names])
    head = np.abs(ref.head_yaw_deg[::2])
    lines = [
        "The persons look into the camera (most of them). So the correct answer is yaw 0, pitch 0.",
        "median = the model's constant error for the average person. spread = person-to-person difference",
        "(half of the middle 68%). The spread includes the persons who do not look into the camera.",
        "",
        f"{'group':28s} {'persons':>7s} {'median yaw':>11s} {'median pitch':>13s} {'spread yaw':>11s} {'spread pitch':>13s} {'median angle':>13s}",
    ]

    def row(title: str, mask: np.ndarray) -> None:
        if mask.sum() < 20:
            return
        g = gaze[mask]
        angle = np.degrees(np.arccos(np.clip(np.cos(np.radians(g[:, 0])) * np.cos(np.radians(g[:, 1])), -1, 1)))
        lines.append(
            f"{title:28s} {int(mask.sum()):7d} {np.median(g[:, 0]):11.1f} {np.median(g[:, 1]):13.1f}"
            f" {_spread(g[:, 0]):11.1f} {_spread(g[:, 1]):13.1f} {np.median(angle):13.1f}"
        )

    usable = label != "Sunglasses"
    row("all (no sunglasses)", usable)
    row("no glasses", label == "NoGlasses")
    row("glasses", label == "ReadingGlasses")
    row("head turn below 10 deg", usable & (head < 10))
    row("head turn 10 to 25 deg", usable & (head >= 10) & (head < 25))
    row("head turn above 25 deg", usable & (head >= 25))
    row("no glasses, head below 10", (label == "NoGlasses") & (head < 10))
    row("glasses, head below 10", (label == "ReadingGlasses") & (head < 10))
    front = eyes[usable & (head < 10)]
    if len(front) >= 20:
        # The two eyes of a person look at the same point. So their difference is model error only.
        lines += [
            "",
            "Each eye alone (head turn below 10 deg). The two eyes look at the same point,",
            "so the difference between them is an error of the model, not of the person's gaze.",
            f"  yaw towards the nose, median:   image-left eye {np.median(front[:, 0, 0]):+.1f}   image-right eye {np.median(front[:, 1, 0]):+.1f}",
            f"  difference between the eyes, spread:   yaw {_spread(front[:, 0, 0] + front[:, 1, 0]):.1f}   pitch {_spread(front[:, 0, 1] - front[:, 1, 1]):.1f}",
        ]
    return "\n".join(lines)


def _standard(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float32)
    x -= x.mean(axis=(1, 2), keepdims=True)
    x /= np.maximum(x.std(axis=(1, 2), keepdims=True), 1.0)
    return x


def _tell_apart(x: np.ndarray, y: np.ndarray, test: np.ndarray, epochs: int, seed: int) -> float:
    """Train a small classifier on the images that are not ``test``. Returns its accuracy on ``test``."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)

    def conv(cin: int, cout: int) -> list:
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)]

    model = nn.Sequential(*conv(1, 16), *conv(16, 32), *conv(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1))
    xt, yt = torch.from_numpy(x[~test, None]), torch.from_numpy(y[~test].astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for i in range(0, len(order), 128):
            idx = order[i : i + 128]
            loss = F.binary_cross_entropy_with_logits(model(xt[idx])[:, 0], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x[test, None]))[:, 0].numpy() > 0
    return float((pred == (y[test] > 0.5)).mean())


def realism(real: np.ndarray, real_group: np.ndarray, synth: np.ndarray, synth_group: np.ndarray,
            grid: tuple[int, int] = (4, 4), epochs: int = 6, seed: int = 0, log=print) -> dict:  # fmt: skip
    """How well a classifier tells real eye images from synthetic ones. 0.5 = it cannot; 1.0 = always.

    The test images are of persons that the classifier did not see. ``cells`` in the result
    has one value for each part of the image (rows from the top, columns from the temple
    side to the nose side), so it shows where the synthetic eyes differ.
    """
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth))
    pick_r, pick_s = rng.choice(len(real), n, replace=False), rng.choice(len(synth), n, replace=False)
    x = _standard(np.concatenate([real[pick_r], synth[pick_s]]))
    y = np.concatenate([np.ones(n), np.zeros(n)])
    test = np.zeros(2 * n, bool)
    for offset, group in ((0, np.asarray(real_group)[pick_r]), (n, np.asarray(synth_group)[pick_s])):
        ids = np.unique(group)
        held = set(rng.choice(ids, max(1, len(ids) // 5), replace=False).tolist())
        test[offset : offset + n] = [g in held for g in group.tolist()]
    out = {"images_per_class": n, "whole": _tell_apart(x, y, test, epochs, seed)}
    log(f"whole image: {out['whole']:.2f}")
    cols, rows = grid
    w, h = eyenorm.SIZE[0] // cols, eyenorm.SIZE[1] // rows
    cells = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            cells[r, c] = _tell_apart(np.ascontiguousarray(x[:, r * h : (r + 1) * h, c * w : (c + 1) * w]), y, test, epochs, seed)
        log("  row %d: %s" % (r, "  ".join(f"{v:.2f}" for v in cells[r])))
    out["cells"] = cells
    return out


def run(args) -> int:
    if args.reference_command == "fetch":
        fetch()
        print(f"photo files in {reference_dir()}")
        return 0
    if not list(reference_dir().glob("*.tar")):
        print("no reference photos. Download them with: handsfree reference fetch", file=sys.stderr)
        return 1
    ref = cached()
    if args.reference_command == "gaze":
        print(f"{len(ref.images) // 2} photos with a face")
        print(format_portrait_gaze(ref, portrait_gaze(ref, args.model), glasses_labels(ref.name)))
        return 0
    if args.reference_command == "realism":
        from .gazetrain import build_examples

        data = build_examples(args.set)
        real_group = np.unique(np.array(ref.name), return_inverse=True)[1]
        print("classifier accuracy, real photos against synthetic images (0.5 = cannot tell them apart)")
        print("columns: temple side -> nose side; rows: top -> bottom")
        realism(ref.images, real_group, data["images"], data["group"])
        return 0
    return 1
