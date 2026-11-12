import numpy as np
import pytest

from handsfree.gaze import FEATURE_NAMES, LandmarkGazeEstimator
from handsfree.pipeline import PipelineResult
from handsfree.recording import (
    EYE_CROP,
    FACE_SIZE,
    FrameRecorder,
    Recording,
    crop,
    face_box,
    list_recordings,
    record_targets,
)
from handsfree.types import Frame

from test_gaze import make_face


def test_crop_inside_and_at_the_border():
    img = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    c, origin = crop(img, 15, 10, 8, 6)
    assert origin == (11, 7) and np.array_equal(c, img[7:13, 11:19])
    c, origin = crop(img, 1, 1, 8, 6)  # leaves the image at the top-left
    assert origin == (-3, -2)
    assert not c[:2].any() and not c[:, :3].any() and np.array_equal(c[2:, 3:], img[0:4, 0:5])
    c, _ = crop(img, 500, 500, 8, 6)  # fully outside
    assert c.shape == (6, 8, 3) and not c.any()


def test_face_box_is_a_square_around_the_landmarks():
    lm = np.zeros((4, 3), np.float32)
    lm[:, :2] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5)]
    x0, y0, side = face_box(lm, (400, 400))
    assert side == 260 and (x0, y0) == (200 - 130, 150 - 130)


def test_record_targets():
    t = record_targets(np.random.default_rng(1))
    assert t[0].kind == "warmup" and len(t) == 26
    pts = {(p.x, p.y) for p in t[1:]}
    assert len(pts) == 25 and min(x for x, _ in pts) == pytest.approx(0.06) and max(y for _, y in pts) == pytest.approx(0.94)
    assert [p.name for p in t] != [p.name for p in record_targets(np.random.default_rng(2))]


def _result(seed: int, iris_dx: float) -> PipelineResult:
    image = np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    face = make_face(iris_dx=iris_dx)
    face.blendshapes.update({"browInnerUp": 0.25, "jawOpen": 0.5})
    return PipelineResult(Frame(image, seed, float(seed), float(seed)), face, LandmarkGazeEstimator().estimate(face), {})


def test_recorder_round_trip(tmp_path):
    targets = record_targets(np.random.default_rng(0))
    rec = FrameRecorder(targets)
    results = [_result(i, iris_dx=i - 1.0) for i in range(3)]
    for i, r in enumerate(results):
        rec.add(r, target_index=i + 1, since=0.1 * i)
    rec.add(PipelineResult(results[0].frame, None, None, {}), 1, 0.0)  # no face: not recorded
    assert len(rec) == 3
    directory = rec.save(tmp_path / "record-20261112-020000", {"created": "x", "screen_px": [2048, 1280]})
    assert list_recordings(tmp_path) == [directory]

    back = Recording.load(directory)
    f = back.frames
    assert len(back) == 3 and back.meta["frames"] == 3 and back.meta["image_size"] == [640, 480]
    assert tuple(f["names"]) == FEATURE_NAMES and list(f["blendshape_names"]) == ["browInnerUp", "jawOpen"]
    assert f["blendshapes"][0].tolist() == [0.25, 0.5]
    assert f["landmarks"].shape == (3, 478, 3) and np.isnan(f["matrix"]).all()
    assert f["target_xy"][1].tolist() == [targets[2].x, targets[2].y]
    assert np.allclose(f["features"][2], results[2].gaze.vector)

    eyes = back.eyes()
    assert eyes.shape == (3, 2, EYE_CROP[1], EYE_CROP[0], 3) and eyes.dtype == np.uint8
    # The crops hold the raw camera pixels, and the stored origin locates them in the image.
    for k in (0, 1):
        x0, y0 = f["eye_origin"][1][k]
        assert np.array_equal(eyes[1, k], results[1].frame.image[y0 : y0 + EYE_CROP[1], x0 : x0 + EYE_CROP[0]])
        cx, cy = f["eye_center"][1][k]
        assert abs(x0 + EYE_CROP[0] / 2 - cx) <= 1 and abs(y0 + EYE_CROP[1] / 2 - cy) <= 1
    assert back.face(2).shape == (FACE_SIZE, FACE_SIZE, 3)


def test_empty_recording_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        FrameRecorder(record_targets()).save(tmp_path / "r", {})
