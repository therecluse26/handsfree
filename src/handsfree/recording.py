"""Labelled recordings for the gaze benchmark.

A recording holds, for each camera frame in a recording window:

- the target the user looked at (the label),
- the face tracker output (landmarks, face transform, gesture scores) and
  the gaze features,
- a crop of each eye, as raw camera pixels with no resampling,
- one small face crop (JPEG).

Recordings are the real test data for every gaze estimator, and they give
the real noise, blur and compression values of the camera. They contain
images of the user's eyes and face. They are written under
``~/.local/share/handsfree/recordings/`` and nothing sends them anywhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .sessions import data_dir
from .types import Target

EYE_CROP = (192, 128)
"""(width, height) in camera pixels. Raw pixels, centred on the eye."""
FACE_SIZE = 224
FACE_MARGIN = 1.3
FACE_JPEG_QUALITY = 92


def recordings_dir() -> Path:
    return data_dir() / "recordings"


def crop(image: np.ndarray, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut ``width`` x ``height`` pixels centred on (cx, cy). Areas outside the image are black.

    Returns the crop and the image position (x0, y0) of its top-left pixel.
    """
    x0, y0 = int(round(cx)) - width // 2, int(round(cy)) - height // 2
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    h, w = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + width, w), min(y0 + height, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out, (x0, y0)


def face_box(landmarks: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int]:
    """A square around the face: (x0, y0, side) in camera pixels."""
    pts = landmarks[:, :2] * np.array(image_size, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = int(round(float((hi - lo).max()) * FACE_MARGIN))
    cx, cy = (lo + hi) / 2.0
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


class FrameRecorder:
    """Collects frames in memory during a session, then writes them to a directory."""

    def __init__(self, targets: list[Target]) -> None:
        import cv2

        self._cv2 = cv2
        self.targets = targets
        self._rows: list[dict] = []
        self._eyes: list[np.ndarray] = []
        self._faces: list[bytes] = []
        self.names: tuple[str, ...] = ()
        self.blendshape_names: tuple[str, ...] = ()
        self.image_size: tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, result, target_index: int, since: float) -> None:
        """Add one frame. ``result`` is a PipelineResult with a face."""
        face, gaze, image = result.face, result.gaze, result.frame.image
        if face is None or gaze is None:
            return
        if not self.names:
            self.names = gaze.names
            self.blendshape_names = tuple(sorted(face.blendshapes))
            self.image_size = face.image_size
        eyes, origins = zip(*(crop(image, *e.center_px, *EYE_CROP) for e in gaze.eyes))
        x0, y0, side = face_box(face.landmarks, face.image_size)
        face_img, _ = crop(image, x0 + side / 2, y0 + side / 2, side, side)
        small = self._cv2.resize(face_img, (FACE_SIZE, FACE_SIZE), interpolation=self._cv2.INTER_AREA)
        ok, jpeg = self._cv2.imencode(".jpg", small, [self._cv2.IMWRITE_JPEG_QUALITY, FACE_JPEG_QUALITY])
        t = self.targets[target_index]
        self._eyes.append(np.stack(eyes))
        self._faces.append(jpeg.tobytes() if ok else b"")
        self._rows.append(
            {
                "t_capture": result.frame.t_capture,
                "since": since,
                "target_index": target_index,
                "target_xy": (t.x, t.y),
                "features": gaze.vector,
                "blinking": gaze.blinking,
                "landmarks": face.landmarks,
                "matrix": face.matrix if face.matrix is not None else np.full((4, 4), np.nan),
                "blendshapes": [face.blendshapes.get(n, 0.0) for n in self.blendshape_names],
                "eye_origin": origins,
                "eye_center": [e.center_px for e in gaze.eyes],
                "eye_width": [e.width_px for e in gaze.eyes],
                "face_box": (x0, y0, side),
            }
        )

    def save(self, directory: Path, meta: dict) -> Path:
        """Write the recording. Returns the directory."""
        if not self._rows:
            raise ValueError("the recording has no frames")
        directory.mkdir(parents=True, exist_ok=True)
        col = lambda k, dtype: np.array([r[k] for r in self._rows], dtype=dtype)  # noqa: E731
        np.savez_compressed(
            directory / "frames.npz",
            t_capture=col("t_capture", float),
            since=col("since", float),
            target_index=col("target_index", int),
            target_xy=col("target_xy", float),
            features=col("features", float),
            blinking=col("blinking", bool),
            landmarks=col("landmarks", np.float32),
            matrix=col("matrix", np.float32),
            blendshapes=col("blendshapes", np.float32),
            eye_origin=col("eye_origin", int),
            eye_center=col("eye_center", np.float32),
            eye_width=col("eye_width", np.float32),
            face_box=col("face_box", int),
            names=np.array(self.names),
            blendshape_names=np.array(self.blendshape_names),
            target_name=np.array([t.name for t in self.targets]),
            target_kind=np.array([t.kind for t in self.targets]),
        )
        np.savez_compressed(directory / "eyes.npz", eyes=np.stack(self._eyes))
        sizes = np.array([len(f) for f in self._faces], dtype=np.int64)
        np.savez(
            directory / "faces.npz",
            jpeg=np.frombuffer(b"".join(self._faces), dtype=np.uint8),
            offset=np.concatenate([[0], np.cumsum(sizes)]),
        )
        full = {
            **meta,
            "frames": len(self._rows),
            "image_size": list(self.image_size),
            "eye_crop": list(EYE_CROP),
            "face_size": FACE_SIZE,
        }
        (directory / "meta.json").write_text(json.dumps(full, indent=1))
        return directory


@dataclass(frozen=True, eq=False)
class Recording:
    """A saved recording. ``eyes`` and ``faces`` load from disk when first used."""

    directory: Path
    meta: dict
    frames: dict[str, np.ndarray]

    @classmethod
    def load(cls, directory: Path) -> Recording:
        with np.load(directory / "frames.npz", allow_pickle=False) as d:
            frames = {k: d[k] for k in d.files}
        return cls(directory, json.loads((directory / "meta.json").read_text()), frames)

    def __len__(self) -> int:
        return len(self.frames["t_capture"])

    def eyes(self) -> np.ndarray:
        """frames x 2 x height x width x 3, uint8 BGR. Index 0 is the eye on the image-left side."""
        with np.load(self.directory / "eyes.npz", allow_pickle=False) as d:
            return d["eyes"]

    def face(self, i: int) -> np.ndarray:
        """The face crop of frame ``i``: FACE_SIZE x FACE_SIZE x 3, uint8 BGR."""
        import cv2

        with np.load(self.directory / "faces.npz", allow_pickle=False) as d:
            a, b = d["offset"][i], d["offset"][i + 1]
            return cv2.imdecode(d["jpeg"][a:b], cv2.IMREAD_COLOR)


def list_recordings(directory: Path | None = None) -> list[Path]:
    """Saved recordings, oldest first."""
    return sorted(p for p in (directory or recordings_dir()).glob("record-*") if (p / "meta.json").exists())


def directory_size_mb(directory: Path) -> float:
    return sum(f.stat().st_size for f in directory.iterdir() if f.is_file()) / 1e6


def record_targets(rng: np.random.Generator | None = None) -> list[Target]:
    """A warm-up target, then a 5 x 5 grid in random order."""
    rng = rng or np.random.default_rng()
    grid = [(float(x), float(y)) for y in np.linspace(0.06, 0.94, 5) for x in np.linspace(0.06, 0.94, 5)]
    order = rng.permutation(len(grid))
    return [Target("warm-up", 0.5, 0.5, kind="warmup"), *(Target(f"{grid[i][0]:g},{grid[i][1]:g}", *grid[i]) for i in order)]


def camera_controls(device: str) -> str:
    """The camera's exposure and gain settings, as text. Best effort."""
    tool = shutil.which("v4l2-ctl")
    if not tool:
        return ""
    try:
        return subprocess.run([tool, "-d", device, "--list-ctrls"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def intro(directory: Path) -> str:
    return (
        "Recording\n\n"
        "26 targets appear, one at a time, in random order.\n"
        "Sit as you normally sit. Look at the centre dot of each target until it moves.\n\n"
        "This records images of your eyes and face.\n"
        f"They are stored on this computer only, in\n{directory}\n\n"
        "Space: start        Esc: cancel"
    )


def run_record(cfg) -> int:
    from .face import LOW_LIGHT
    from .targets import run_target_session

    rc = cfg.recording
    cfg = replace(cfg, camera=replace(cfg.camera, width=rc.width, height=rc.height))
    targets = record_targets()
    recorder = FrameRecorder(targets)
    session = run_target_session(
        cfg, "record", targets, intro(recordings_dir()), rc.settle_s, rc.record_s, on_frame=recorder.add
    )
    if session is None:
        print("recording cancelled; nothing was saved")
        return 1
    if not len(recorder):
        print("recording failed: no face was visible")
        return 1
    session.save()
    directory = recordings_dir() / session.file_name.removesuffix(".npz")
    print("saving …")
    recorder.save(
        directory,
        {
            "created": session.created,
            "screen_px": list(session.screen_px),
            "screen_mm": list(session.screen_mm),
            "brightness": session.brightness,
            "camera": {"device": cfg.camera.device, "fps": cfg.camera.fps, "fourcc": cfg.camera.fourcc},
            "camera_controls": camera_controls(cfg.camera.device),
        },
    )
    rec = Recording.load(directory)
    f = rec.frames
    fixation = np.array([targets[i].kind == "fixation" for i in f["target_index"]])
    per_target = np.bincount(f["target_index"][fixation], minlength=len(targets))[1:]
    low = "   LOW LIGHT: add light on your face, then record again" if session.brightness < LOW_LIGHT else ""
    print(f"camera image   {rec.meta['image_size'][0]}x{rec.meta['image_size'][1]}")
    print(f"frames         {len(rec)}  ({per_target.min()} to {per_target.max()} for each of 25 targets)")
    print(f"eyes closed    {f['blinking'].mean():.0%} of the frames")
    print(f"eye width      {np.median(f['eye_width']):.0f} px (median)")
    print(f"face brightness {session.brightness:.0f}/255{low}")
    print(f"size on disk   {directory_size_mb(directory):.0f} MB")
    print(f"saved          {directory}")
    return 0
