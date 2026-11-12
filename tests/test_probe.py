import numpy as np
import pytest

from handsfree.gaze import FEATURE_NAMES, blendshape_gaze
from handsfree.probe import PROBE_TARGETS, SPAN, analyse, format_report


def fake_samples(gain_x=0.25, gain_y=0.10, noise=0.005, n=80, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in PROBE_TARGETS:
        s = rng.normal(0.0, noise, size=(n, len(FEATURE_NAMES)))
        s[:, FEATURE_NAMES.index("gaze_x")] += gain_x * (t.x - 0.5)
        s[:, FEATURE_NAMES.index("gaze_y")] += gain_y * (t.y - 0.5)
        out.append(s)
    return out


def test_analyse_recovers_signal_and_noise():
    xs, ys = analyse(PROBE_TARGETS, fake_samples(), FEATURE_NAMES)
    gx = next(s for s in xs if s.name == "gaze_x")
    gy = next(s for s in ys if s.name == "gaze_y")
    assert gx.signal == pytest.approx(0.25 * SPAN, abs=0.004)
    assert gy.signal == pytest.approx(0.10 * SPAN, abs=0.004)
    assert gx.noise == pytest.approx(0.005, rel=0.3)
    # noise on screen = noise / signal * span * screen width
    assert gx.noise_px(2000) == pytest.approx(gx.noise / gx.signal * SPAN * 2000)
    # A signal that does not respond has a ratio near zero.
    assert next(s for s in xs if s.name == "bs_x").ratio < 1.0


def test_report_text_and_gains():
    text = format_report(PROBE_TARGETS, fake_samples(), FEATURE_NAMES, (2048, 1280), 30.0)
    assert "LOW LIGHT" in text
    assert "gain_x = 4.0" in text and "gain_y = 10." in text
    assert text.count("centre") >= 3


def test_too_few_samples():
    s = fake_samples()
    s[3] = s[3][:3]  # index 0 is the warm-up target
    with pytest.raises(ValueError, match="right"):
        analyse(PROBE_TARGETS, s, FEATURE_NAMES)


def test_blendshape_gaze_signs():
    # The user looks to their right: left eye turns in, right eye turns out.
    assert blendshape_gaze({"eyeLookInLeft": 0.6, "eyeLookOutRight": 0.4}) == pytest.approx((0.5, 0.0))
    assert blendshape_gaze({"eyeLookDownLeft": 0.2, "eyeLookDownRight": 0.4})[1] == pytest.approx(0.3)
    assert blendshape_gaze({}) == (0.0, 0.0)
