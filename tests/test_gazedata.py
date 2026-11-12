import json
from pathlib import Path

import numpy as np
import pytest

from handsfree import eyenorm
from handsfree.benchmark import CANONICAL_CM, EYES, MEDIAPIPE_VFOV_DEG, DeviceGeometry, EyeImageModel
from handsfree.gazedata import eye_positions, recording_examples, synth_examples
from handsfree.recording import EYE_CROP, Recording
from handsfree.synth import SynthSet

cv2 = pytest.importorskip("cv2")

W, H = 1920, 1080
FOCAL = (H / 2) / np.tan(np.radians(44.7) / 2)


def project(p):
    return np.array([W / 2 + FOCAL * p[0] / -p[2], H / 2 - FOCAL * p[1] / -p[2]])


def make_synth(tmp_path) -> SynthSet:
    eyes = [np.array([-31.0, -100.0, -550.0]), np.array([32.0, -98.0, -552.0])]
    target = np.array([100.0, -150.0, 0.0])
    x0, y0, w, h = 704, 600, 512, 192
    img = np.full((h, w, 3), 0.05, np.float32)
    for e in eyes:  # a bright dot at each eye point
        u, v = project(e)
        cv2.circle(img, (int(round(u)) - x0, int(round(v)) - y0), 5, (2.0, 2.0, 2.0), -1)
    np.savez_compressed(tmp_path / "s0001_p0000_00.npz", rgb=img.astype(np.float16))
    label = {
        "image": "s0001_p0000_00.npz",
        "rect": [x0, y0, w, h],
        "camera": {"width": W, "height": H, "focal_px": FOCAL, "vertical_fov_deg": 44.7},
        "crop_centres_px": [project(e).tolist() for e in eyes],
        "eyes": [{"tracker_centre_cam_mm": e.tolist(), "sight_cam": ((target - e) / np.linalg.norm(target - e)).tolist()} for e in eyes],
    }
    (tmp_path / "labels-0001.jsonl").write_text(json.dumps(label) + "\n")
    return SynthSet.load(tmp_path)


def test_synthetic_examples_are_centred_and_mirrored(tmp_path):
    ex = synth_examples(make_synth(tmp_path), [0], np.random.default_rng(0), augment=False)
    assert ex.images.shape == (2, eyenorm.SIZE[1], eyenorm.SIZE[0]) and ex.is_right.tolist() == [False, True]
    for im in ex.images:  # the bright dot is at the centre of both images
        ys, xs = np.nonzero(im > 200)
        assert abs(xs.mean() - 47.5) < 1.5 and abs(ys.mean() - 31.5) < 1.5
    # Both eyes look at one target on the image's right side: the yaw of the mirrored eye has the other sign.
    assert ex.angles[0, 0] > 5 and ex.angles[1, 0] < -5
    assert ex.angles[0, 1] == pytest.approx(ex.angles[1, 1], abs=0.5)
    assert ex.views[1].angles(np.array(make_synth(tmp_path).labels[0]["eyes"][1]["sight_cam"]))[0] == pytest.approx(-ex.angles[1, 0], abs=1e-4)


def test_augmented_copies_differ(tmp_path):
    ex = synth_examples(make_synth(tmp_path), [0], np.random.default_rng(1), augment=True, copies=3)
    assert len(ex.images) == 6 and len(set(ex.group.tolist())) == 1
    assert np.abs(ex.images[0].astype(int) - ex.images[2].astype(int)).mean() > 1
    assert np.abs(ex.angles[0] - ex.angles[2]).max() < 3  # the label changes only a little with the tracker error


def make_recording(vfov=44.7) -> tuple[Recording, DeviceGeometry]:
    geom = DeviceGeometry((345.0, 215.0), (2048, 1280), 10.0, vfov)
    # A face transform in the tracker's units (cm, depth for a 63 degree camera).
    mat = np.eye(4)
    mat[:3, 3] = (2.0, -8.0, -40.0)
    n = 3
    centres, origins = np.zeros((n, 2, 2), np.float32), np.zeros((n, 2, 2), int)
    f_tracker = (H / 2) / np.tan(np.radians(MEDIAPIPE_VFOV_DEG) / 2)
    for k, eye in enumerate(EYES):
        a, b = eye["corners"]
        p = (np.array(CANONICAL_CM[a]) + np.array(CANONICAL_CM[b])) / 2 + mat[:3, 3]
        centres[:, k] = [W / 2 + f_tracker * p[0] / -p[2], H / 2 - f_tracker * p[1] / -p[2]]
        origins[:, k] = np.round(centres[0, k]).astype(int) - [EYE_CROP[0] // 2, EYE_CROP[1] // 2]
    frames = {
        "matrix": np.repeat(mat[None].astype(np.float32), n, 0),
        "eye_center": centres,
        "eye_origin": origins,
        "target_xy": np.array([[0.5, 0.5], [0.06, 0.06], [0.94, 0.94]]),
        "t_capture": np.arange(n, dtype=float),
    }
    meta = {"image_size": [W, H], "screen_mm": [345.0, 215.0], "screen_px": [2048, 1280]}

    class Fake(Recording):
        def eyes(self):
            return np.full((n, 2, EYE_CROP[1], EYE_CROP[0], 3), 90, np.uint8)

    return Fake(Path("record-fake"), meta, frames), geom


def test_eye_positions_use_the_real_field_of_view():
    rec, geom = make_recording()
    eyes, focal = eye_positions(rec, geom)
    assert focal == pytest.approx(FOCAL)
    assert eyes[0, 0, 2] == pytest.approx(-(40.0 - 3.465) * 10 * geom.depth_scale, rel=0.01)  # further away than the tracker says
    # The eye point projects to the tracker's pixel with the real camera.
    assert project(eyes[0, 0]) == pytest.approx(rec.frames["eye_center"][0, 0], abs=0.01)
    assert eyes[0, 1, 0] - eyes[0, 0, 0] == pytest.approx(63.2, abs=1.0)  # the canonical distance between the eyes


def test_recording_examples_have_the_target_direction():
    rec, geom = make_recording()
    ex = recording_examples(rec, geom)
    assert ex.images.shape == (6, eyenorm.SIZE[1], eyenorm.SIZE[0]) and ex.group.tolist() == [0, 0, 1, 1, 2, 2]
    # Frame 1 looks at the top-left of the screen, frame 2 at the bottom-right (the user's view).
    # The user's left is the image's right, so the yaw of the image-left eye is positive there.
    assert ex.angles[2, 0] > ex.angles[0, 0] > ex.angles[4, 0]
    assert ex.angles[2, 1] > ex.angles[0, 1] > ex.angles[4, 1]


def test_a_perfect_model_finds_the_targets_on_the_screen():
    rec, geom = make_recording()
    truth = recording_examples(rec, geom).angles

    class Perfect(EyeImageModel):
        def angles(self, images):
            return truth.copy()

    pred = Perfect().predict(rec, geom)
    target = rec.frames["target_xy"] * geom.screen_mm
    assert np.abs(pred.screen_mm - target).max() < 0.01  # mm
