import math

import numpy as np
import pytest

from handsfree.gaze import IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE, LandmarkGazeEstimator, eye_state
from handsfree.geometry import head_pose_from_matrix
from handsfree.mapping import UncalibratedMapper
from handsfree.types import FaceObservation, HeadPose

SIZE = (640, 480)


def make_face(iris_dx=0.0, iris_dy=0.0, lid_gap=12.0, roll_deg=0.0, head=None):
    """Synthetic face: two level eyes 40 px wide. Offsets are in pixels, in the eye frame."""
    lm = np.zeros((478, 3), dtype=np.float32)
    rot = math.radians(roll_deg)
    c, s = math.cos(rot), math.sin(rot)

    def put(i, x, y, origin):
        px = origin[0] + c * x - s * y
        py = origin[1] + s * x + c * y
        lm[i, :2] = (px / SIZE[0], py / SIZE[1])

    for idx, origin in ((IMAGE_LEFT_EYE, (250.0, 200.0)), (IMAGE_RIGHT_EYE, (390.0, 200.0))):
        put(idx.corner_a, -20, 0, origin)
        put(idx.corner_b, 20, 0, origin)
        put(idx.upper, 0, -lid_gap / 2, origin)
        put(idx.lower, 0, lid_gap / 2, origin)
        put(idx.iris, iris_dx, iris_dy, origin)
    return FaceObservation(t_capture=1.0, image_size=SIZE, landmarks=lm, blendshapes={}, head=head)


def test_centred_iris_gives_zero():
    g = LandmarkGazeEstimator().estimate(make_face())
    assert g.get("gaze_x") == pytest.approx(0.0, abs=1e-5)
    assert g.get("gaze_y") == pytest.approx(0.0, abs=1e-5)
    assert not g.blinking
    assert g.eyes[0].openness == pytest.approx(0.3, abs=1e-4)
    assert g.eyes[0].width_px == pytest.approx(40.0, abs=1e-3)


def test_sign_convention_is_the_users_view():
    # Iris moves LEFT in the image -> the user looks to the RIGHT of the screen.
    g = LandmarkGazeEstimator().estimate(make_face(iris_dx=-4.0, iris_dy=2.0))
    assert g.get("gaze_x") == pytest.approx(0.1, abs=1e-4)
    assert g.get("gaze_y") == pytest.approx(0.05, abs=1e-4)  # down is positive


def test_head_roll_does_not_change_the_eye_features():
    a = LandmarkGazeEstimator().estimate(make_face(iris_dx=3.0, iris_dy=-1.0))
    b = LandmarkGazeEstimator().estimate(make_face(iris_dx=3.0, iris_dy=-1.0, roll_deg=25.0))
    assert b.get("gaze_x") == pytest.approx(a.get("gaze_x"), abs=1e-4)
    assert b.get("gaze_y") == pytest.approx(a.get("gaze_y"), abs=1e-4)


def test_blink():
    assert LandmarkGazeEstimator().estimate(make_face(lid_gap=2.0)).blinking


def test_head_pose_goes_into_the_feature_vector():
    head = HeadPose(yaw=5.0, pitch=-3.0, roll=1.0, position=(1.0, 2.0, -55.0))
    g = LandmarkGazeEstimator().estimate(make_face(head=head))
    assert [g.get(n) for n in ("head_yaw", "head_pitch", "head_roll", "head_z")] == [5.0, -3.0, 1.0, -55.0]
    assert len(g.vector) == len(g.names)


def test_eye_state_pixels():
    e = eye_state(make_face(iris_dx=5.0).landmarks, SIZE, IMAGE_RIGHT_EYE)
    assert e.center_px == pytest.approx((390.0, 200.0), abs=1e-2)
    assert e.iris_px == pytest.approx((395.0, 200.0), abs=1e-2)


def _rot(yaw, pitch, roll):
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, -10, 5), (-35, 25, -15)])
def test_head_pose_from_matrix(angles):
    m = np.eye(4)
    m[:3, :3] = _rot(*angles) * 1.7  # uniform scale must not matter
    m[:3, 3] = (1.0, -2.0, -60.0)
    hp = head_pose_from_matrix(m)
    assert (hp.yaw, hp.pitch, hp.roll) == pytest.approx(angles, abs=1e-6)
    assert hp.position == (1.0, -2.0, -60.0)


def test_uncalibrated_mapper():
    est = LandmarkGazeEstimator()
    m = UncalibratedMapper(gain_x=4.0, gain_y=6.0)
    # The first open-eye sample is the centre reference.
    assert m.map(est.estimate(make_face(iris_dx=1.0))) == pytest.approx((0.5, 0.5))
    x, y = m.map(est.estimate(make_face(iris_dx=-1.0, iris_dy=1.0)))
    assert x == pytest.approx(0.5 + 4.0 * 0.05, abs=1e-4)
    assert y == pytest.approx(0.5 + 6.0 * 0.025, abs=1e-4)
    assert m.map(est.estimate(make_face(lid_gap=1.0))) is None  # blink
    assert m.map(est.estimate(make_face(iris_dx=-20.0))) == pytest.approx((1.0, 0.5))  # clipped
    m.recenter()
    assert m.map(est.estimate(make_face(iris_dx=-3.0))) == pytest.approx((0.5, 0.5))
