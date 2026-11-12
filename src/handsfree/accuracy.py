"""Accuracy test: the user looks at targets, we measure the calibration error.

For each target:

- offset: distance from the target to the median estimated gaze point. This
  is the accuracy (systematic error).
- spread: RMS distance of the single estimates from their median, without
  outliers. This is the precision (jitter before any smoothing).
- outliers: share of estimates far from the median (a look away, or a
  tracker fault).

The 12 test targets are at different places than the 13 calibration targets.
Each run is saved as a session and appended to accuracy.jsonl, so runs and
models can be compared later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .calibration import CANDIDATES, WARMUP, Calibration, GazeModel, fit_rows, score_model, training_rows
from .sessions import TargetSession, data_dir, list_sessions
from .types import Target

ACCURACY_TARGETS = [
    WARMUP,
    *(
        Target(f"{x:g},{y:g}", x, y)
        for row, y in enumerate((0.2, 0.5, 0.8))
        for x in ((0.125, 0.375, 0.625, 0.875) if row % 2 == 0 else (0.875, 0.625, 0.375, 0.125))
    ),
]
MIN_SAMPLES = 10

INTRO = (
    "Accuracy test\n\n"
    "13 targets appear, one at a time.\n"
    "Sit comfortably. You do not need to hold your head rigid.\n"
    "Look at the centre dot of each target until it moves.\n\n"
    "Space: start        Esc: cancel"
)


@dataclass(frozen=True, slots=True)
class TargetResult:
    target: Target
    dx_px: float
    """Estimate minus target. Positive: the estimate is right of the target."""
    dy_px: float
    """Positive: the estimate is below the target."""
    spread_px: float
    outliers: float
    """0..1."""
    samples: int

    @property
    def offset_px(self) -> float:
        return math.hypot(self.dx_px, self.dy_px)


def evaluate(model: GazeModel, session: TargetSession) -> list[TargetResult]:
    size = np.array(session.screen_px, dtype=float)
    results = []
    for t, s in zip(session.targets, session.samples):
        if t.kind != "fixation":
            continue
        if len(s) < MIN_SAMPLES:
            raise ValueError(f"target {t.name} has only {len(s)} samples; the face or eyes were not visible")
        pts = model.predict(s, session.names) * size
        med = np.median(pts, axis=0)
        d = med - np.array([t.x, t.y]) * size
        dist = np.linalg.norm(pts - med, axis=1)
        inlier = dist <= max(3.0 * 1.4826 * float(np.median(dist)), 1.0)
        spread = float(np.sqrt(np.mean(dist[inlier] ** 2)))
        results.append(TargetResult(t, float(d[0]), float(d[1]), spread, float(1.0 - inlier.mean()), len(s)))
    return results


def px_per_degree(screen_px_w: int, screen_mm_w: float, distance_cm: float) -> float | None:
    """Screen pixels for 1 degree of gaze angle, near the screen centre."""
    if screen_mm_w <= 0 or distance_cm <= 0:
        return None
    return screen_px_w / (screen_mm_w / 10.0) * distance_cm * math.tan(math.radians(1.0))


def summarise(results: list[TargetResult]) -> dict[str, float]:
    offsets = np.array([r.offset_px for r in results])
    dx = np.array([r.dx_px for r in results])
    dy = np.array([r.dy_px for r in results])
    return {
        "mean_offset_px": float(offsets.mean()),
        "median_offset_px": float(np.median(offsets)),
        "worst_offset_px": float(offsets.max()),
        "mean_abs_dx_px": float(np.abs(dx).mean()),
        "mean_abs_dy_px": float(np.abs(dy).mean()),
        "mean_spread_px": float(np.mean([r.spread_px for r in results])),
        "bias_x_px": float(dx.mean()),
        "bias_y_px": float(dy.mean()),
        "offset_without_bias_px": float(np.hypot(dx - dx.mean(), dy - dy.mean()).mean()),
    }


def format_report(results: list[TargetResult], ppd: float | None, distance_cm: float, model_name: str) -> str:
    def deg(px: float) -> str:
        return f"{px / ppd:>6.1f}°" if ppd else f"{'n/a':>7}"

    out = [f"model {model_name}"]
    if ppd:
        out.append(f"1° = {ppd:.0f} px at {distance_cm:.0f} cm (the distance is an estimate)")
    out += ["", f"{'target':<12}{'offset px':>10}{'':>7}{'dx':>7}{'dy':>7}{'spread px':>11}{'outliers':>10}"]
    for r in results:
        out.append(
            f"{r.target.name:<12}{r.offset_px:>10.0f}{deg(r.offset_px)}{r.dx_px:>+7.0f}{r.dy_px:>+7.0f}"
            f"{r.spread_px:>11.0f}{r.outliers:>10.0%}"
        )
    s = summarise(results)
    out += [
        "",
        f"accuracy  mean offset   {s['mean_offset_px']:>5.0f} px {deg(s['mean_offset_px'])}",
        f"          median offset {s['median_offset_px']:>5.0f} px {deg(s['median_offset_px'])}",
        f"          worst offset  {s['worst_offset_px']:>5.0f} px {deg(s['worst_offset_px'])}",
        f"          mean |dx| {s['mean_abs_dx_px']:.0f} px, mean |dy| {s['mean_abs_dy_px']:.0f} px",
        f"precision mean spread   {s['mean_spread_px']:>5.0f} px {deg(s['mean_spread_px'])}",
        f"bias      mean dx {s['bias_x_px']:+.0f} px, mean dy {s['bias_y_px']:+.0f} px",
        f"          mean offset without the bias: {s['offset_without_bias_px']:.0f} px",
        "          (a large bias means that the head compensation is not good enough)",
    ]
    return "\n".join(out)


def viewing_distance_cm(session: TargetSession, configured: float) -> float:
    if configured > 0 or "head_z" not in session.names:
        return configured
    k = session.names.index("head_z")
    z = np.concatenate([s[:, k] for t, s in zip(session.targets, session.samples) if t.kind == "fixation"])
    return float(abs(np.median(z))) if len(z) else 0.0


def run_accuracy(cfg) -> int:
    from .targets import run_target_session

    cal = Calibration.load_or_none()
    if cal is None:
        print("no usable calibration found. Run: handsfree calibrate")
        return 1
    session = run_target_session(
        cfg, "accuracy", ACCURACY_TARGETS, INTRO, cfg.calibration.settle_s, cfg.calibration.record_s
    )
    if session is None:
        print("accuracy test cancelled")
        return 1
    recorded = session.save()
    try:
        results = evaluate(cal.model, session)
    except ValueError as e:
        print(f"accuracy test failed: {e}")
        return 1
    distance = viewing_distance_cm(session, cfg.screen.viewing_distance_cm)
    ppd = px_per_degree(session.screen_px[0], session.screen_mm[0], distance)
    print(format_report(results, ppd, distance, cal.model.spec.name))
    record = {
        "time": session.created,
        "model": cal.model.spec.name,
        "calibrated": cal.created,
        "session": session.file_name,
        "screen_px": list(session.screen_px),
        "px_per_degree": ppd,
        "distance_cm": distance,
        "brightness": session.brightness,
        **summarise(results),
    }
    log = data_dir() / "accuracy.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nsession: {recorded}\nlogged:  {log}")
    return 0


def format_evaluation(calibration: TargetSession, tests: list[TargetSession]) -> str:
    """Fit every model on one calibration session and test it on accuracy sessions."""
    rows = training_rows(calibration)
    out = [
        f"calibration session {calibration.file_name}",
        f"accuracy sessions   {len(tests)}" + ("" if tests else "  (none recorded after this calibration)"),
        "",
        "All values in px. 'cross-check' uses the calibration session only.",
        f"{'':<26}{'cross-check':>17}{'accuracy sessions':>32}",
        f"{'model':<26}{'still':>8}{'moves':>9}{'offset':>11}{'|dx|':>7}{'|dy|':>7}{'no bias':>9}",
    ]
    lines = []
    for spec in CANDIDATES:
        score = score_model(spec, rows, calibration.names, calibration.screen_px)
        if score is None:
            continue
        model = fit_rows(score, rows, calibration.names)
        moves = f"{score.head_px:>9.0f}" if score.head_px is not None else f"{'n/a':>9}"
        line = f"{spec.name:<26}{score.still_px:>8.0f}{moves}"
        key = score.cross_check_px
        if tests:
            s = summarise([r for t in tests for r in evaluate(model, t)])
            key = s["mean_offset_px"]
            line += (
                f"{s['mean_offset_px']:>11.0f}{s['mean_abs_dx_px']:>7.0f}{s['mean_abs_dy_px']:>7.0f}"
                f"{s['offset_without_bias_px']:>9.0f}"
            )
        lines.append((key, line))
    return "\n".join(out + [line for _, line in sorted(lines)])


def run_evaluate() -> int:
    calibrations = list_sessions("calibrate")
    if not calibrations:
        print("no recorded calibration session found. Run: handsfree calibrate")
        return 1
    latest = calibrations[-1]
    tests = [TargetSession.load(p) for p in list_sessions("accuracy") if p.name.split("-", 1)[1] > latest.name.split("-", 1)[1]]
    try:
        print(format_evaluation(TargetSession.load(latest), tests))
    except ValueError as e:
        print(f"evaluation failed: {e}")
        return 1
    return 0
