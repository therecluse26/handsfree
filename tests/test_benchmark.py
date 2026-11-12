from pathlib import Path

import numpy as np
import pytest

from handsfree.benchmark import (
    CANONICAL_CM,
    EYES,
    MEDIAPIPE_VFOV_DEG,
    DeviceGeometry,
    GeometricEyeModel,
    Prediction,
    format_results,
    run_estimators,
    score,
)
from handsfree.recording import Recording, record_targets

SIZE = (1920, 1080)
F_TRACKER = (SIZE[1] / 2) / np.tan(np.radians(MEDIAPIPE_VFOV_DEG / 2))


def _rotation_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _project(p_cm: np.ndarray) -> np.ndarray:
    """Camera space (cm) to normalised image coordinates, with the tracker's camera."""
    u = SIZE[0] / 2 + F_TRACKER * p_cm[0] / -p_cm[2]
    v = SIZE[1] / 2 - F_TRACKER * p_cm[1] / -p_cm[2]
    return np.array([u / SIZE[0], v / SIZE[1]])


def synthetic_recording(geom: DeviceGeometry, model: GeometricEyeModel, frames_per_target: int = 3) -> Recording:
    """A recording whose iris landmarks come from exact eyeballs that look at the targets."""
    targets = record_targets(np.random.default_rng(0))
    mat = np.eye(4)
    mat[:3, :3] = _rotation_y(-4.0)
    mat[:3, 3] = (4.4, -10.0, -40.0)
    rows = []
    for ti, t in enumerate(targets):
        target_cm = geom.to_camera(np.array([[t.x * geom.screen_mm[0], t.y * geom.screen_mm[1]]]))[0] / 10.0
        lm = np.zeros((478, 3))
        for eye in EYES:
            a, b = eye["corners"]
            for c in (a, b):
                lm[c, :2] = _project(mat[:3, :3] @ np.array(CANONICAL_CM[c]) + mat[:3, 3])
            centre_canon = (np.array(CANONICAL_CM[a]) + np.array(CANONICAL_CM[b])) / 2 - [0, 0, model.back_cm]
            centre = mat[:3, :3] @ centre_canon + mat[:3, 3]
            gaze = target_cm - centre
            iris = centre + model.radius_cm * gaze / np.linalg.norm(gaze)
            lm[eye["iris"], :2] = _project(iris)
        rows += [(ti, (t.x, t.y), lm)] * frames_per_target
    frames = {
        "target_index": np.array([r[0] for r in rows]),
        "target_xy": np.array([r[1] for r in rows]),
        "landmarks": np.array([r[2] for r in rows], np.float32),
        "matrix": np.repeat(mat[None].astype(np.float32), len(rows), axis=0),
        "blinking": np.zeros(len(rows), bool),
        "target_kind": np.array([t.kind for t in targets]),
    }
    meta = {"image_size": list(SIZE), "screen_mm": list(geom.screen_mm), "screen_px": list(geom.screen_px)}
    return Recording(Path("record-synthetic"), meta, frames)


GEOM = DeviceGeometry((345.0, 215.0), (2048, 1280), camera_above_mm=10.0)


def test_screen_and_camera_coordinates_round_trip():
    pts = np.array([[0.0, 0.0], [345.0, 215.0], [100.0, 50.0]])
    cam = GEOM.to_camera(pts)
    assert cam[0].tolist() == [172.5, -10.0, 0.0]  # top-left of the screen is on the camera's +x side
    assert np.allclose(GEOM.to_screen(cam), pts)


def test_depth_scale():
    assert GEOM.depth_scale == 1.0
    assert DeviceGeometry((1, 1), (1, 1), vertical_fov_deg=MEDIAPIPE_VFOV_DEG).depth_scale == pytest.approx(1.0)
    assert DeviceGeometry((1, 1), (1, 1), vertical_fov_deg=50.0).depth_scale > 1.0  # narrower view: further away


def test_geometric_model_recovers_exact_eyeballs():
    model = GeometricEyeModel()
    rec = synthetic_recording(GEOM, model)
    pred = model.predict(rec, GEOM)
    target = rec.frames["target_xy"] * GEOM.screen_mm
    assert np.abs(pred.screen_mm - target).max() < 1.0  # mm
    s = score(rec, pred, GEOM)
    assert s.mean_deg < 0.2 and s.mean_px < 6
    assert s.gain == pytest.approx((1.0, 1.0), abs=0.01)
    assert s.frames == 75 and s.valid == 1.0


def test_score_reports_bias_gain_and_jitter():
    rec = synthetic_recording(GEOM, GeometricEyeModel())
    target = rec.frames["target_xy"] * GEOM.screen_mm
    centre = np.array(GEOM.screen_mm) / 2
    shifted = centre + (target - centre) * 0.5 + [10.0, 0.0]  # half the movement, 10 mm to the right
    eye = np.tile([44.0, -100.0, -400.0], (len(target), 1))
    s = score(rec, Prediction(shifted, eye), GEOM)
    assert s.gain == pytest.approx((0.5, 0.5))
    assert s.bias_px[0] == pytest.approx(10.0 * GEOM.px_per_mm[0])
    assert s.jitter_px == pytest.approx(0.0, abs=1e-9)
    nan = shifted.copy()
    nan[:30] = np.nan
    assert score(rec, Prediction(nan, eye), GEOM).valid < 1.0


def test_wrong_eyeball_constants_give_an_error():
    rec = synthetic_recording(GEOM, GeometricEyeModel(radius_cm=1.05))
    s = score(rec, GeometricEyeModel(radius_cm=1.3).predict(rec, GEOM), GEOM)
    assert s.gain[0] < 0.95  # a larger eyeball turns the same iris shift into a smaller angle


def test_format_results_lists_each_recording():
    rec = synthetic_recording(GEOM, GeometricEyeModel())
    text = format_results(run_estimators([rec, rec], [GeometricEyeModel()], 10.0, 0.0))
    assert "estimator: geometric" in text and text.count("record-synthetic") == 2 and "mean" in text
