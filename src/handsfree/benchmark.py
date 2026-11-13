"""Zero-calibration gaze benchmark.

A benchmark scores gaze estimators on saved recordings (``handsfree record``)
with no per-user calibration. Each estimator gets only the camera frames, the
face tracker output and the device geometry: screen size, camera position and
camera field of view. It returns one screen point per frame. The score
compares that point with the target the user looked at.

Coordinates
-----------
Camera space follows the face tracker: x to the right of the un-mirrored
camera image, y up, z towards the user's face is negative (the user sits at
z < 0). The camera lens is in the screen plane (z = 0), above the centre of
the picture's top edge. Screen points are millimetres from the top-left corner
of the picture, as the user sees it: x to the right, y down. The user's right
is the camera image's left, so screen x grows with camera -x.

Units: the tracker's face transform is in centimetres. This module uses
millimetres for all screen and camera-space values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .recording import Recording, list_recordings

MEDIAPIPE_VFOV_DEG = 63.0
"""The vertical field of view that the MediaPipe face transform assumes."""

# Canonical face model of the MediaPipe face tracker, in cm (from
# geometry_pipeline_metadata_landmarks.binarypb inside face_landmarker.task).
CANONICAL_CM = {
    33: (-4.4518, 2.6566, 3.1718),
    133: (-1.8642, 2.5900, 3.7588),
    362: (1.8642, 2.5900, 3.7588),
    263: (4.4518, 2.6566, 3.1718),
}
EYES = (
    {"corners": (33, 133), "iris": 468},  # image-left eye (the user's right eye)
    {"corners": (362, 263), "iris": 473},  # image-right eye (the user's left eye)
)


@dataclass(frozen=True)
class DeviceGeometry:
    """Screen and camera of one device."""

    screen_mm: tuple[float, float]
    screen_px: tuple[int, int]
    camera_above_mm: float = 10.0
    """Camera lens centre above the top edge of the screen picture."""
    vertical_fov_deg: float = 0.0
    """Real vertical field of view of the camera. 0 means: unknown, use the tracker's assumption."""

    @property
    def px_per_mm(self) -> np.ndarray:
        return np.array(self.screen_px, float) / np.array(self.screen_mm, float)

    @property
    def depth_scale(self) -> float:
        """Real distance / tracker distance. The tracker assumes MEDIAPIPE_VFOV_DEG."""
        if self.vertical_fov_deg <= 0:
            return 1.0
        return float(np.tan(np.radians(MEDIAPIPE_VFOV_DEG / 2)) / np.tan(np.radians(self.vertical_fov_deg / 2)))

    def to_camera(self, screen_mm: np.ndarray) -> np.ndarray:
        """Screen points (N x 2, mm from top-left) to camera space (N x 3, mm)."""
        s = np.asarray(screen_mm, float)
        return np.stack([self.screen_mm[0] / 2 - s[:, 0], -(self.camera_above_mm + s[:, 1]), np.zeros(len(s))], 1)

    def to_screen(self, camera_mm: np.ndarray) -> np.ndarray:
        """Camera-space points in the screen plane (N x 3, mm) to screen points (N x 2, mm)."""
        p = np.asarray(camera_mm, float)
        return np.stack([self.screen_mm[0] / 2 - p[:, 0], -p[:, 1] - self.camera_above_mm], 1)

    @classmethod
    def for_recording(cls, rec: Recording, camera_above_mm: float = 10.0, vertical_fov_deg: float = 0.0):
        return cls(
            tuple(rec.meta["screen_mm"]), tuple(rec.meta["screen_px"]), camera_above_mm, vertical_fov_deg
        )


@dataclass(frozen=True)
class Prediction:
    screen_mm: np.ndarray
    """N x 2. NaN where the estimator has no result."""
    eye_mm: np.ndarray
    """N x 3, camera space. The point between the eyes that the gaze starts from."""


class Estimator(Protocol):
    name: str
    description: str

    def predict(self, rec: Recording, geom: DeviceGeometry) -> Prediction: ...


def _rays(px: np.ndarray, size: tuple[int, int], f: float) -> np.ndarray:
    """Unit rays through image pixels (N x 2) for a pinhole camera with focal length f (px)."""
    w, h = size
    d = np.stack([(px[:, 0] - w / 2) / f, -(px[:, 1] - h / 2) / f, -np.ones(len(px))], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass(frozen=True)
class GeometricEyeModel:
    """A 3D eyeball model on the face tracker's landmarks. It uses population averages only.

    For each eye: the eyeball centre is a fixed point in the canonical face,
    behind the midpoint of the eye corners. The iris centre is where the
    camera ray through the iris landmark meets a sphere around the eyeball
    centre. The gaze ray runs from the eyeball centre through the iris
    centre. The result is the mean of the two eyes' screen points.
    """

    back_cm: float = 0.8
    """Eyeball centre behind the midpoint of the eye corners, along the face's z axis."""
    radius_cm: float = 1.05
    """Eyeball centre to the visible pupil centre."""
    name: str = "geometric"
    description: str = "3D eyeball on MediaPipe iris landmarks, population averages"

    def predict(self, rec: Recording, geom: DeviceGeometry) -> Prediction:
        f = rec.frames
        size = tuple(rec.meta["image_size"])
        f_tracker = (size[1] / 2) / np.tan(np.radians(MEDIAPIPE_VFOV_DEG / 2))
        mat = f["matrix"].astype(float)
        lm = f["landmarks"].astype(float)
        scale = np.array([10.0, 10.0, 10.0 * geom.depth_scale])  # cm -> mm, with the real depth
        points, origins = [], []
        for eye in EYES:
            a, b = eye["corners"]
            centre_canon = (np.array(CANONICAL_CM[a]) + np.array(CANONICAL_CM[b])) / 2 - [0.0, 0.0, self.back_cm]
            centre = np.einsum("nij,j->ni", mat[:, :3, :3], centre_canon) + mat[:, :3, 3]
            ray = _rays(lm[:, eye["iris"], :2] * size, size, f_tracker)
            # Nearest intersection of the ray t * ray with the sphere |x - centre| = radius.
            b_ = np.einsum("ni,ni->n", ray, centre)
            c_ = np.einsum("ni,ni->n", centre, centre) - self.radius_cm**2
            t = b_ - np.sqrt(np.clip(b_ * b_ - c_, 0.0, None))
            gaze = ray * t[:, None] - centre
            gaze /= np.linalg.norm(gaze, axis=1, keepdims=True)
            start = centre * scale
            with np.errstate(divide="ignore", invalid="ignore"):
                s = -start[:, 2] / gaze[:, 2]
            hit = start + gaze * s[:, None]
            hit[~(s > 0)] = np.nan  # the ray points away from the screen
            points.append(geom.to_screen(hit))
            origins.append(start)
        return Prediction(np.mean(points, axis=0), np.mean(origins, axis=0))


@dataclass(frozen=True)
class EyeImageModel:
    """The eye-image gaze model (see gazetrain.py). It reads the eye pixels and is trained on synthetic data only."""

    name: str = "eye-cnn"
    description: str = "CNNs on normalised eye images, trained on synthetic data"
    model: str = ""
    """Path of the ONNX file. Empty means the default path."""

    def path(self):
        from .gazetrain import model_path
        from pathlib import Path

        return Path(self.model) if self.model else model_path()

    def angles(self, images: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
        """Gaze angles (N x 2 degrees) for normalised eye images (N x 64 x 96 uint8, N even: pairs of eyes).

        ``head`` is the head direction of each image (N x 2 degrees, gazedata). A model needs it
        when its .json file has "head": true.

        The model's .json file can name more models in the same folder:
        "ensemble": {"yaw": [files], "pitch": [files]} gives the yaw as the mean of the first
        list and the pitch as the mean of the second. "pitch_from" names one model for the pitch.
        With neither, this model gives both angles.
        """
        info, folder = _model_info(self.path()), self.path().parent
        ensemble = info.get("ensemble") or {}
        yaw = [folder / name for name in ensemble.get("yaw", [])] or [self.path()]
        pitch = [folder / name for name in ensemble.get("pitch", [])]
        if not pitch:
            other = info.get("pitch_from")
            pitch = [folder / other] if other and (folder / other).exists() else [self.path()]
        out = {path: _run_eye_model(path, images, head) for path in dict.fromkeys(yaw + pitch)}
        return np.stack([np.mean([out[p][:, 0] for p in yaw], axis=0), np.mean([out[p][:, 1] for p in pitch], axis=0)], axis=1)

    def predict(self, rec: Recording, geom: DeviceGeometry) -> Prediction:
        from .gazedata import recording_examples

        ex = recording_examples(rec, geom)
        angles = self.angles(ex.images, ex.head).astype(float)
        angles[:, 0] *= np.where(ex.is_right, -1.0, 1.0)  # undo the mirror: only the yaw changes sign
        hits, origins = [], []
        for view, a in zip(ex.views, angles):
            d = view.direction(a)
            s = -view.eye_mm[2] / d[2] if d[2] > 1e-6 else np.nan  # the screen is the plane z = 0
            hits.append(view.eye_mm + d * s)
            origins.append(view.eye_mm)
        hits = np.array(hits).reshape(len(rec), 2, 3)
        origins = np.array(origins).reshape(len(rec), 2, 3)
        return Prediction(geom.to_screen(hits.mean(axis=1)), origins.mean(axis=1))


def _model_info(path) -> dict:
    """The .json file that ``handsfree train`` writes next to a model. Empty when there is none."""
    import json

    info = path.with_suffix(".json")
    return json.loads(info.read_text()) if info.exists() else {}


def _run_eye_model(path, images: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
    """Run one ONNX eye model on normalised eye images. It reads the window that its .json file records."""
    import cv2

    from . import eyenorm
    from .gazetrain import ANGLE_SCALE, HEAD_SCALE

    info = _model_info(path)
    window = info.get("window")
    net = cv2.dnn.readNetFromONNX(str(path))
    x = eyenorm.to_model_input(images, window=tuple(window) if window else None)
    if info.get("head"):
        if head is None:
            raise RuntimeError(f"the model {path.name} needs the head direction")
        h = np.asarray(head, np.float32) / HEAD_SCALE
    out = np.empty((len(x), 2), np.float32)
    for i in range(0, len(x), 2):  # the model takes the two eyes of one frame
        net.setInput(np.ascontiguousarray(x[i : i + 2]), "eyes")
        if info.get("head"):
            net.setInput(np.ascontiguousarray(h[i : i + 2]), "head")
        out[i : i + 2] = net.forward()
    return out * ANGLE_SCALE


ESTIMATORS: dict[str, Estimator] = {e.name: e for e in (GeometricEyeModel(), EyeImageModel())}


@dataclass(frozen=True)
class Score:
    frames: int
    valid: float
    """Fraction of the scored frames with a prediction."""
    mean_deg: float
    median_deg: float
    mean_px: float
    p95_px: float
    target_mean_px: float
    """Error of the mean prediction for each target. Removes frame-to-frame jitter."""
    jitter_px: float
    """Mean spread of the predictions for one target (standard deviation, both axes)."""
    bias_px: tuple[float, float]
    """Mean (prediction - target), x right, y down."""
    debiased_px: float
    """Diagnostic only: the error after the recording's own bias is removed (a 1-point calibration)."""
    gain: tuple[float, float]
    """Predicted movement / true movement for each axis. 1.0 is correct."""


def score(rec: Recording, pred: Prediction, geom: DeviceGeometry) -> Score:
    f = rec.frames
    kinds = f["target_kind"][f["target_index"]]
    use = (kinds == "fixation") & ~f["blinking"]
    valid = use & np.isfinite(pred.screen_mm).all(axis=1)
    target = f["target_xy"][valid] * np.array(geom.screen_mm)
    p = pred.screen_mm[valid]
    ppm = geom.px_per_mm
    err_px = (p - target) * ppm
    dist = np.linalg.norm(err_px, axis=1)

    eye = pred.eye_mm[valid]
    a = geom.to_camera(target) - eye
    b = geom.to_camera(p) - eye
    cos = np.einsum("ni,ni->n", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    ti = f["target_index"][valid]
    groups = [ti == t for t in np.unique(ti)]
    per_target = np.array([err_px[g].mean(axis=0) for g in groups])
    jitter = np.mean([np.sqrt((err_px[g].var(axis=0)).sum()) for g in groups])
    bias = err_px.mean(axis=0)
    gain = tuple(float(np.polyfit(target[:, j], p[:, j], 1)[0]) for j in (0, 1))
    return Score(
        frames=int(use.sum()),
        valid=float(valid.sum() / max(use.sum(), 1)),
        mean_deg=float(deg.mean()),
        median_deg=float(np.median(deg)),
        mean_px=float(dist.mean()),
        p95_px=float(np.percentile(dist, 95)),
        target_mean_px=float(np.linalg.norm(per_target, axis=1).mean()),
        jitter_px=float(jitter),
        bias_px=(float(bias[0]), float(bias[1])),
        debiased_px=float(np.linalg.norm(err_px - bias, axis=1).mean()),
        gain=gain,  # type: ignore[arg-type]
    )


@dataclass
class BenchmarkResult:
    estimator: str
    rows: list[tuple[str, Score]] = field(default_factory=list)


def run_estimators(
    recordings: list[Recording], estimators: list[Estimator], camera_above_mm: float, vertical_fov_deg: float
) -> list[BenchmarkResult]:
    out = []
    for est in estimators:
        res = BenchmarkResult(est.name)
        for rec in recordings:
            geom = DeviceGeometry.for_recording(rec, camera_above_mm, vertical_fov_deg)
            res.rows.append((rec.directory.name, score(rec, est.predict(rec, geom), geom)))
        out.append(res)
    return out


def format_results(results: list[BenchmarkResult]) -> str:
    head = (
        f"{'recording':<24}{'error':>8}{'median':>8}{'error':>7}{'p95':>7}{'target':>8}"
        f"{'jitter':>8}{'bias x,y':>13}{'gain x,y':>12}"
    )
    unit = f"{'':<24}{'deg':>8}{'deg':>8}{'px':>7}{'px':>7}{'px':>8}{'px':>8}{'px':>13}{'':>12}"
    lines = []
    for res in results:
        est = ESTIMATORS.get(res.estimator)
        lines += [f"estimator: {res.estimator}" + (f"  ({est.description})" if est else ""), head, unit]
        for name, s in res.rows:
            lines.append(
                f"{name:<24}{s.mean_deg:>8.1f}{s.median_deg:>8.1f}{s.mean_px:>7.0f}{s.p95_px:>7.0f}"
                f"{s.target_mean_px:>8.0f}{s.jitter_px:>8.0f}{s.bias_px[0]:>+7.0f},{s.bias_px[1]:>+5.0f}"
                f"{s.gain[0]:>7.2f},{s.gain[1]:>4.2f}"
            )
        if len(res.rows) > 1:
            m = lambda k: np.mean([getattr(s, k) for _, s in res.rows])  # noqa: E731
            lines.append(
                f"{'mean':<24}{m('mean_deg'):>8.1f}{m('median_deg'):>8.1f}{m('mean_px'):>7.0f}{m('p95_px'):>7.0f}"
                f"{m('target_mean_px'):>8.0f}{m('jitter_px'):>8.0f}"
            )
        lines.append("")
    lines.append(
        "error: per frame, no calibration. target: error of the mean point for each target.\n"
        "jitter: spread for one target. gain: predicted / true movement, 1.00 is correct."
    )
    return "\n".join(lines)


def run_benchmark(cfg, names: list[str] | None = None) -> int:
    paths = list_recordings()
    if not paths:
        print("no recordings found. Make one with: handsfree record")
        return 1
    unknown = [n for n in names or [] if n not in ESTIMATORS]
    if unknown:
        print(f"unknown estimator(s): {', '.join(unknown)}. Known: {', '.join(ESTIMATORS)}")
        return 1
    estimators = [ESTIMATORS[n] for n in names] if names else list(ESTIMATORS.values())
    for est in list(estimators):
        if isinstance(est, EyeImageModel) and not est.path().exists():
            if names:
                print(f"no trained model at {est.path()}. Train one with: handsfree train")
                return 1
            estimators.remove(est)  # not trained yet: leave it out of the default list
    recs = [Recording.load(p) for p in paths]
    print(f"{len(recs)} recordings, camera {cfg.screen.camera_above_mm:g} mm above the screen\n")
    results = run_estimators(recs, estimators, cfg.screen.camera_above_mm, cfg.camera.vertical_fov_deg)
    print(format_results(results))
    return 0
