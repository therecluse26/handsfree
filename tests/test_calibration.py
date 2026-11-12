import numpy as np
import pytest

from handsfree import config
from handsfree.accuracy import (
    ACCURACY_TARGETS,
    evaluate,
    format_evaluation,
    format_report,
    px_per_degree,
    summarise,
    viewing_distance_cm,
)
from handsfree.calibration import (
    CANDIDATES,
    FIXATION_TARGETS,
    CalibratedMapper,
    Calibration,
    build,
    calibration_targets,
    fit,
    format_summary,
    training_rows,
)
from handsfree.gaze import FEATURE_NAMES
from handsfree.sessions import TargetSession, list_sessions
from handsfree.types import GazeFeatures

SCREEN = (2048, 1280)
N = FEATURE_NAMES.index


def simulate(targets, kind="calibrate", seed=0, noise=1.0, head=(0.0, 0.0, 0.0, 0.0), created="2026-11-12T01:00:00-05:00"):
    """A session with the signal size, cross-talk and noise that a real probe run showed.

    The eye signals depend on the target position relative to the head, so a
    head movement changes them. ``head`` is a constant (yaw°, pitch°, x cm, y cm).
    """
    rng = np.random.default_rng(seed)
    samples, times = [], []
    for t in targets:
        n = int((t.record_s or 1.0) * 60)
        tt = np.arange(n) / 60.0
        yaw, pitch, hx, hy = (np.full(n, v) for v in head)
        wave = 5.0 * np.sin(2 * np.pi * tt / 3.0)
        if t.name == "head-turn":
            yaw = yaw + wave
        elif t.name == "head-nod":
            pitch = pitch + wave
        elif t.name == "head-shift":
            hx, hy = hx + 0.6 * wave, hy + 0.4 * np.cos(2 * np.pi * tt / 3.0)
        # where the eyes must point, relative to the head
        x = t.x - 0.5 - 0.020 * yaw - 0.030 * hx
        y = t.y - 0.5 + 0.030 * pitch + 0.045 * hy
        s = np.zeros((n, len(FEATURE_NAMES)))
        s[:, N("gaze_x")] = -0.022 + 0.100 * x + rng.normal(0, 0.0021 * noise, n)
        s[:, N("gaze_y")] = -0.070 + 0.039 * y - 0.017 * x + rng.normal(0, 0.0017 * noise, n)
        s[:, N("bs_x")] = -0.257 + 0.60 * x - 0.08 * x * x + rng.normal(0, 0.0085 * noise, n)
        s[:, N("bs_y")] = 0.151 + 0.38 * y - 0.095 * x + rng.normal(0, 0.0086 * noise, n)
        for name, v, sd in (("head_yaw", yaw, 0.08), ("head_pitch", pitch, 0.13), ("head_x", hx, 0.02), ("head_y", hy, 0.02)):
            s[:, N(name)] = v + rng.normal(0, sd * noise, n)
        s[:, N("head_z")] = -55.0 + rng.normal(0, 0.05 * noise, n)
        samples.append(s)
        times.append(tt)
    return TargetSession(kind, created, list(targets), samples, times, FEATURE_NAMES, 90.0, SCREEN, (344.6, 215.4))


def test_rows_from_a_session():
    rows = training_rows(simulate(calibration_targets()))
    assert (~rows.head).sum() == 13  # the warm-up target is not used
    assert rows.head.sum() == 3 * 24  # 3 head movements x 6 s / 0.25 s
    assert len(np.unique(rows.group)) == 13 + 6
    assert set(rows.weight[rows.head]) == {0.25}


def test_head_models_win_when_the_session_has_a_head_phase():
    cal = build(simulate(calibration_targets()))
    assert len(cal.scores) == len(CANDIDATES)
    assert cal.model.spec.head and cal.score.cross_check_px < 40
    eye_only = next(s for s in cal.scores if s.spec.name == "blendshape-linear")
    assert eye_only.head_px > 3 * cal.score.head_px
    text = format_summary(cal)
    assert "head moves" in text and "<- in use" in text


def test_without_head_phase_only_eye_models_are_trained():
    cal = build(simulate(calibration_targets(head_phase=False)))
    assert len(cal.scores) == 6 and all(s.head_px is None for s in cal.scores)
    assert "head moves" not in format_summary(cal)
    with pytest.raises(ValueError, match="needs data"):
        build(simulate(calibration_targets(head_phase=False)), model="combined-linear+head")


def test_a_recorded_head_phase_can_be_ignored():
    cal = build(simulate(calibration_targets()), use_head=False)
    assert len(cal.scores) == 6 and not cal.model.spec.head


def test_head_model_removes_the_bias_of_a_moved_head():
    session = simulate(calibration_targets())
    moved = simulate(ACCURACY_TARGETS, "accuracy", seed=1, head=(3.0, -2.0, 1.5, -1.0))
    with_head = summarise(evaluate(build(session).model, moved))
    eye_only = summarise(evaluate(build(session, "blendshape-linear").model, moved))
    assert abs(eye_only["bias_x_px"]) > 150 and abs(eye_only["bias_y_px"]) > 100
    assert abs(with_head["bias_x_px"]) < 40 and abs(with_head["bias_y_px"]) < 40
    assert with_head["mean_offset_px"] < 60 < eye_only["mean_offset_px"]


def test_accuracy_report():
    cal = build(simulate(calibration_targets()))
    session = simulate(ACCURACY_TARGETS, "accuracy", seed=1)
    session.samples[3][:6, :] += 5.0  # a short look away at one target
    res = evaluate(cal.model, session)
    assert len(res) == 12  # the warm-up target is not scored
    assert res[2].outliers == pytest.approx(0.1) and res[2].spread_px < 80
    assert viewing_distance_cm(session, 0.0) == pytest.approx(55.0, abs=0.1)
    assert viewing_distance_cm(session, 60.0) == 60.0
    text = format_report(res, px_per_degree(2048, 344.6, 55.0), 55.0, cal.model.spec.name)
    assert "mean offset" in text and "°" in text and max(map(len, text.splitlines())) <= 80


def test_too_few_samples():
    session = simulate(calibration_targets())
    session.samples[4] = session.samples[4][:2]
    with pytest.raises(ValueError, match="only 2 samples"):
        build(session)


def test_exact_linear_data_is_recovered():
    y = np.array([(t.x, t.y) for t in FIXATION_TARGETS])
    x = np.zeros((len(y), len(FEATURE_NAMES)))
    x[:, N("bs_x")] = 0.3 + 0.5 * y[:, 0]
    x[:, N("bs_y")] = -0.1 + 0.4 * y[:, 1] + 0.1 * y[:, 0]
    model = fit(CANDIDATES[2], x, FEATURE_NAMES, y, ridge=1e-9)
    assert model.predict(x, FEATURE_NAMES) == pytest.approx(y, abs=1e-6)


def test_save_load_and_mapper(tmp_path):
    cal = build(simulate(calibration_targets()))
    back = Calibration.load(cal.save(tmp_path / "sub" / "calibration.json"))
    assert back.model.spec == cal.model.spec and back.screen_px == SCREEN and back.session == cal.session
    assert [s.spec.name for s in back.scores] == [s.spec.name for s in cal.scores]
    session = simulate(ACCURACY_TARGETS, "accuracy", noise=0.0)
    v, t = session.samples[1][0], ACCURACY_TARGETS[1]
    assert back.model.predict(v, FEATURE_NAMES) == pytest.approx(cal.model.predict(v, FEATURE_NAMES))
    mapper = CalibratedMapper(back.model)
    x, y = mapper.map(GazeFeatures(0.0, v, FEATURE_NAMES, (None, None), blinking=False))
    assert abs(x - t.x) * SCREEN[0] < 40 and abs(y - t.y) * SCREEN[1] < 40
    assert mapper.map(GazeFeatures(0.0, v, FEATURE_NAMES, (None, None), blinking=True)) is None
    far = v.copy()
    far[N("bs_x")] = far[N("gaze_x")] = 50.0
    assert 0.0 <= mapper.map(GazeFeatures(0.0, far, FEATURE_NAMES, (None, None), blinking=False))[0] <= 1.0
    assert Calibration.load_or_none(tmp_path / "absent.json") is None
    (tmp_path / "old.json").write_text('{"version": 1}')
    assert Calibration.load_or_none(tmp_path / "old.json") is None


def test_session_round_trip_and_evaluation(tmp_path):
    cal_session = simulate(calibration_targets())
    acc = simulate(ACCURACY_TARGETS, "accuracy", seed=2, head=(2.0, 1.0, 0.0, 0.5), created="2026-11-12T01:05:00-05:00")
    p1, p2 = cal_session.save(tmp_path), acc.save(tmp_path)
    assert p1.name == "calibrate-20261112-010000.npz"
    assert list_sessions("accuracy", tmp_path) == [p2]
    back = TargetSession.load(p1)
    assert [t.kind for t in back.targets] == [t.kind for t in cal_session.targets]
    assert all(np.array_equal(a, b) for a, b in zip(back.samples, cal_session.samples))
    assert all(np.array_equal(a, b) for a, b in zip(back.times, cal_session.times))
    assert back.names == FEATURE_NAMES and back.screen_px == SCREEN and back.screen_mm == (344.6, 215.4)
    text = format_evaluation(back, [TargetSession.load(p2)])
    lines = text.splitlines()
    assert "+head" in lines[6]  # sorted by the error on the accuracy session
    assert len(lines) == 6 + len(CANDIDATES) and max(map(len, lines)) <= 80
    assert "none recorded" in format_evaluation(back, [])


def test_px_per_degree():
    # 2048 px over 34.46 cm = 59.43 px/cm; 1 degree at 55 cm = 0.960 cm
    assert px_per_degree(2048, 344.6, 55.0) == pytest.approx(57.06, abs=0.05)
    assert px_per_degree(2048, 0.0, 55.0) is None


def test_targets_and_config():
    fix = {(t.x, t.y) for t in FIXATION_TARGETS}
    acc = {(t.x, t.y) for t in ACCURACY_TARGETS if t.kind == "fixation"}
    assert len(fix) == 13 and len(acc) == 12 and not fix & acc
    assert [t.kind for t in calibration_targets()][:2] == ["warmup", "fixation"]
    assert config.from_dict({"calibration": {"model": "combined-linear+head"}}).calibration.model.endswith("+head")
    with pytest.raises(config.ConfigError, match="calibration.model"):
        config.from_dict({"calibration": {"model": "magic"}})
