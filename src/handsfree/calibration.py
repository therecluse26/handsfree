"""Calibration: fit a map from gaze features to a screen point.

The calibration session has two parts:

1. Fixation targets. The user looks at 13 known points. Each target gives
   one training row: the median feature vector.
2. Head phase (experimental, off by default). The user looks at the centre
   and moves the head slowly. The eye signals and the head pose change, but
   the gaze point does not. The idea is that the model learns to compensate
   for head position and tilt. In tests so far it made the result worse.

Several candidate models are fitted. Each one gets a cross-check error: a
group of rows is left out, the model is fitted on the other rows, and the
error on the left-out rows is measured. The model with the lowest
cross-check error wins, unless the config names a model.

A model is a weighted ridge regression on standardised features. The eye
signals have linear or quadratic terms. The head pose has linear terms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sessions import TargetSession, data_dir
from .types import GazeFeatures, Target

IRIS = ("gaze_x", "gaze_y")
BLENDSHAPE = ("bs_x", "bs_y")
HEAD = ("head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z")
RIDGES = (1e-3, 1e-2, 1e-1, 1.0)
MIN_SAMPLES = 10
CHUNK_S = 0.25
"""The head phase is cut into pieces of this length. Each piece gives one training row."""
MIN_CHUNK_SAMPLES = 5
HEAD_ROW_WEIGHT = 0.25


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]
    """Eye signals."""
    degree: int
    """1: linear terms of the eye signals. 2: also squares and products."""
    head: tuple[str, ...] = ()
    """Head pose signals. They get linear terms."""


CANDIDATES = (
    ModelSpec("iris-linear", IRIS, 1),
    ModelSpec("iris-quadratic", IRIS, 2),
    ModelSpec("blendshape-linear", BLENDSHAPE, 1),
    ModelSpec("blendshape-quadratic", BLENDSHAPE, 2),
    ModelSpec("combined-linear", IRIS + BLENDSHAPE, 1),
    ModelSpec("combined-quadratic", IRIS + BLENDSHAPE, 2),
    ModelSpec("iris-linear+head", IRIS, 1, HEAD),
    ModelSpec("blendshape-linear+head", BLENDSHAPE, 1, HEAD),
    ModelSpec("combined-linear+head", IRIS + BLENDSHAPE, 1, HEAD),
    ModelSpec("combined-quadratic+head", IRIS + BLENDSHAPE, 2, HEAD),
)

_LOOK = "Keep your eyes on the dot.\n"
HEAD_TARGETS = [
    Target("head-turn", 0.5, 0.5, "head", 6.0, _LOOK + "Slowly turn your head a little left, then a little right."),
    Target("head-nod", 0.5, 0.5, "head", 6.0, _LOOK + "Slowly nod your head a little up, then a little down."),
    Target("head-shift", 0.5, 0.5, "head", 6.0, _LOOK + "Keep your head upright. Slowly move it a little left and right,\nthen a little closer and farther."),
]

FIXATION_TARGETS = [
    Target(f"{x:g},{y:g}", x, y)
    for x, y in (
        (0.5, 0.5),
        (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
        (0.725, 0.275), (0.275, 0.275),
        (0.05, 0.5), (0.95, 0.5),
        (0.725, 0.725), (0.275, 0.725),
        (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    )
]  # fmt: skip

WARMUP = Target("warm-up", 0.5, 0.5, kind="warmup")


def calibration_targets(head_phase: bool = True) -> list[Target]:
    return [WARMUP, *FIXATION_TARGETS, *(HEAD_TARGETS if head_phase else [])]


def intro(head_phase: bool) -> str:
    text = (
        "Calibration\n\n"
        "Part 1: 14 targets appear, one at a time.\n"
        "Sit as you normally sit. Follow each target with your eyes.\n"
        "Look at the centre dot of the target until it moves.\n"
    )
    if head_phase:
        text += (
            "\nPart 2: look at one dot and move your head slowly.\n"
            "The screen tells you how. You get time to read each instruction.\n"
            "Make small movements, as when you shift in your chair.\n"
        )
    return text + "\nSpace: start        Esc: cancel"


def _expand(z: np.ndarray, degree: int) -> np.ndarray:
    k = z.shape[1]
    cols = [np.ones(len(z))] + [z[:, i] for i in range(k)]
    if degree >= 2:
        cols += [z[:, i] * z[:, j] for i in range(k) for j in range(i, k)]
    return np.column_stack(cols)


def _design(spec: ModelSpec, x: np.ndarray, names: tuple[str, ...], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x[:, [names.index(f) for f in (*spec.features, *spec.head)]] - mean) / scale
    k = len(spec.features)
    a = _expand(z[:, :k], spec.degree)
    return np.column_stack([a, z[:, k:]]) if spec.head else a


@dataclass(frozen=True, eq=False)
class GazeModel:
    spec: ModelSpec
    ridge: float
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    """terms x 2. Output is a screen point, both values 0..1."""

    def predict(self, vectors: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        """``vectors`` is n x len(names). Returns n x 2 screen points (not clipped)."""
        v = np.atleast_2d(np.asarray(vectors, dtype=float))
        return _design(self.spec, v, names, self.mean, self.scale) @ self.weights

    def to_dict(self) -> dict:
        return {
            "name": self.spec.name,
            "features": list(self.spec.features),
            "degree": self.spec.degree,
            "head": list(self.spec.head),
            "ridge": self.ridge,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GazeModel:
        return cls(
            ModelSpec(d["name"], tuple(d["features"]), int(d["degree"]), tuple(d.get("head", ()))),
            float(d["ridge"]),
            np.array(d["mean"], dtype=float),
            np.array(d["scale"], dtype=float),
            np.array(d["weights"], dtype=float),
        )


def fit(
    spec: ModelSpec,
    x: np.ndarray,
    names: tuple[str, ...],
    y: np.ndarray,
    ridge: float,
    weight: np.ndarray | None = None,
) -> GazeModel:
    """``x`` is m x len(names) feature vectors, ``y`` is m x 2 screen points (0..1)."""
    f = x[:, [names.index(n) for n in (*spec.features, *spec.head)]]
    mean = f.mean(axis=0)
    scale = f.std(axis=0)
    scale[scale < 1e-9] = 1.0
    a = _design(spec, x, names, mean, scale)
    w = np.ones(len(x)) if weight is None else weight
    aw = a * w[:, None]
    reg = ridge * np.eye(a.shape[1])
    reg[0, 0] = 0.0  # no penalty on the constant term
    weights = np.linalg.solve(a.T @ aw + reg, aw.T @ y)
    return GazeModel(spec, ridge, mean, scale, weights)


def error_px(pred: np.ndarray, truth: np.ndarray, screen_px: tuple[int, int]) -> np.ndarray:
    return np.linalg.norm((pred - truth) * np.array(screen_px, dtype=float), axis=1)


@dataclass(frozen=True, eq=False)
class Rows:
    """Training rows from one calibration session."""

    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    """Rows with the same group number are left out together in the cross-check."""
    weight: np.ndarray
    head: np.ndarray
    """True for rows from the head phase."""

    def usable_by(self, spec: ModelSpec) -> np.ndarray:
        """An eye-only model must not train on head-phase rows: there the eyes move, but the target does not."""
        return np.ones(len(self.x), dtype=bool) if spec.head else ~self.head


def training_rows(session: TargetSession, use_head: bool = True) -> Rows:
    """``use_head=False`` ignores a recorded head phase."""
    x, y, group, weight, head = [], [], [], [], []
    g = 0
    for t, s, times in zip(session.targets, session.samples, session.times):
        if t.kind == "fixation":
            if len(s) < MIN_SAMPLES:
                raise ValueError(f"target {t.name} has only {len(s)} samples; the face or eyes were not visible")
            x.append(np.median(s, axis=0))
            y.append((t.x, t.y))
            group.append(g)
            weight.append(1.0)
            head.append(False)
            g += 1
        elif t.kind == "head" and use_head and len(s):
            chunk = np.floor(times / CHUNK_S).astype(int)
            rows = [np.median(s[chunk == c], axis=0) for c in np.unique(chunk) if (chunk == c).sum() >= MIN_CHUNK_SAMPLES]
            if len(rows) < 4:
                continue
            half = len(rows) // 2
            for i, row in enumerate(rows):
                x.append(row)
                y.append((t.x, t.y))
                group.append(g + (i >= half))  # two groups for each head movement
                weight.append(HEAD_ROW_WEIGHT)
                head.append(True)
            g += 2
    if not x:
        raise ValueError("the session has no usable targets")
    return Rows(np.array(x), np.array(y), np.array(group), np.array(weight), np.array(head, dtype=bool))


def cross_check(
    spec: ModelSpec, rows: Rows, names: tuple[str, ...], screen_px: tuple[int, int], ridge: float
) -> tuple[float, float | None]:
    """Mean error in px on left-out groups: (fixation targets, head phase or None)."""
    usable = rows.usable_by(spec)
    still, moving = [], []
    for g in np.unique(rows.group):
        test = rows.group == g
        train = usable & ~test
        model = fit(spec, rows.x[train], names, rows.y[train], ridge, rows.weight[train])
        err = float(error_px(model.predict(rows.x[test], names), rows.y[test], screen_px).mean())
        (moving if rows.head[test][0] else still).append(err)
    return float(np.mean(still)), (float(np.mean(moving)) if moving else None)


@dataclass(frozen=True, slots=True)
class ModelScore:
    spec: ModelSpec
    ridge: float
    still_px: float
    """Cross-check error on the fixation targets."""
    head_px: float | None
    """Cross-check error on the head phase. None if the session has no head phase."""

    @property
    def cross_check_px(self) -> float:
        return self.still_px if self.head_px is None else (self.still_px + self.head_px) / 2.0


def score_model(spec: ModelSpec, rows: Rows, names: tuple[str, ...], screen_px: tuple[int, int]) -> ModelScore | None:
    """The spec with its best ridge value. None if the session cannot train this spec."""
    if not all(f in names for f in (*spec.features, *spec.head)) or (spec.head and not rows.head.any()):
        return None
    scores = [ModelScore(spec, r, *cross_check(spec, rows, names, screen_px, r)) for r in RIDGES]
    return min(scores, key=lambda s: s.cross_check_px)


def rank_models(rows: Rows, names: tuple[str, ...], screen_px: tuple[int, int]) -> list[ModelScore]:
    """All trainable candidates, lowest cross-check error first."""
    scores = [s for s in (score_model(spec, rows, names, screen_px) for spec in CANDIDATES) if s]
    return sorted(scores, key=lambda s: s.cross_check_px)


def fit_rows(score: ModelScore, rows: Rows, names: tuple[str, ...]) -> GazeModel:
    use = rows.usable_by(score.spec)
    return fit(score.spec, rows.x[use], names, rows.y[use], score.ridge, rows.weight[use])


def calibration_path() -> Path:
    return data_dir() / "calibration.json"


@dataclass(frozen=True, eq=False)
class Calibration:
    model: GazeModel
    scores: list[ModelScore]
    screen_px: tuple[int, int]
    created: str
    brightness: float
    session: str
    """File name of the recorded session that this calibration came from."""

    VERSION = 2

    @property
    def score(self) -> ModelScore:
        return next(s for s in self.scores if s.spec.name == self.model.spec.name)

    def save(self, path: Path | None = None) -> Path:
        path = path or calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "created": self.created,
            "screen_px": list(self.screen_px),
            "brightness": self.brightness,
            "session": self.session,
            "model": self.model.to_dict(),
            "scores": [
                {"name": s.spec.name, "ridge": s.ridge, "still_px": s.still_px, "head_px": s.head_px}
                for s in self.scores
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> Calibration:
        d = json.loads((path or calibration_path()).read_text())
        if d.get("version") != cls.VERSION:
            raise ValueError("the calibration file is from an older version; run: handsfree calibrate")
        specs = {c.name: c for c in CANDIDATES}
        return cls(
            model=GazeModel.from_dict(d["model"]),
            scores=[
                ModelScore(specs[s["name"]], s["ridge"], s["still_px"], s["head_px"])
                for s in d["scores"]
                if s["name"] in specs
            ],
            screen_px=(int(d["screen_px"][0]), int(d["screen_px"][1])),
            created=d["created"],
            brightness=d["brightness"],
            session=d["session"],
        )

    @classmethod
    def load_or_none(cls, path: Path | None = None) -> Calibration | None:
        try:
            return cls.load(path)
        except (OSError, ValueError, KeyError):
            return None


def build(session: TargetSession, model: str = "auto", use_head: bool = True) -> Calibration:
    """Fit a calibration from a recorded calibration session."""
    rows = training_rows(session, use_head)
    scores = rank_models(rows, session.names, session.screen_px)
    if not scores:
        raise ValueError("no model fits the recorded features")
    chosen = scores[0] if model == "auto" else next((s for s in scores if s.spec.name == model), None)
    if chosen is None:
        raise ValueError(f"model {model!r} needs data that this session does not have")
    return Calibration(
        model=fit_rows(chosen, rows, session.names),
        scores=scores,
        screen_px=session.screen_px,
        created=session.created,
        brightness=session.brightness,
        session=session.file_name,
    )


def format_summary(cal: Calibration) -> str:
    has_head = any(s.head_px is not None for s in cal.scores)
    out = ["Cross-check error of each model in px (lower is better)"]
    out.append(f"{'model':<26}{'head still':>11}" + (f"{'head moves':>12}{'both':>7}" if has_head else ""))
    for s in cal.scores:
        line = f"{s.spec.name:<26}{s.still_px:>11.0f}"
        if has_head:
            line += f"{s.head_px:>12.0f}{s.cross_check_px:>7.0f}"
        out.append(line + ("   <- in use" if s.spec.name == cal.model.spec.name else ""))
    return "\n".join(out)


class CalibratedMapper:
    """A GazeMapper (see mapping.py) that uses a fitted GazeModel."""

    def __init__(self, model: GazeModel) -> None:
        self.model = model

    def map(self, features: GazeFeatures) -> tuple[float, float] | None:
        if features.blinking:
            return None
        x, y = self.model.predict(features.vector, features.names)[0]
        return (min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y))))


def run_calibrate(cfg, refit: bool = False) -> int:
    """Record a calibration session and fit it. ``refit`` fits the latest recording again, with no camera."""
    from .face import LOW_LIGHT
    from .sessions import list_sessions

    c = cfg.calibration
    if refit:
        recordings = list_sessions("calibrate")
        if not recordings:
            print("no recorded calibration session found. Run: handsfree calibrate")
            return 1
        recorded = recordings[-1]
        session = TargetSession.load(recorded)
    else:
        from .targets import run_target_session

        session = run_target_session(
            cfg,
            "calibrate",
            calibration_targets(c.head_phase),
            intro(c.head_phase),
            c.settle_s,
            c.record_s,
            prompt_read_s=c.prompt_read_s,
        )
        if session is None:
            print("calibration cancelled; the stored calibration is not changed")
            return 1
        recorded = session.save()
    try:
        cal = build(session, c.model, use_head=c.head_phase)
    except ValueError as e:
        print(f"calibration failed: {e}")
        return 1
    path = cal.save()
    low = "   LOW LIGHT: add light on your face, then repeat" if session.brightness < LOW_LIGHT else ""
    print(f"screen {session.screen_px[0]}x{session.screen_px[1]} px, face brightness {session.brightness:.0f}/255{low}\n")
    print(format_summary(cal))
    print(f"\nsession: {recorded}\nsaved:   {path}\nnext:    handsfree accuracy")
    return 0
