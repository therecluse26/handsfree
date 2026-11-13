import io
import tarfile

import cv2
import numpy as np
import pytest

from handsfree import eyenorm, reference
from handsfree.reference import ReferenceEyes


def _eyes(photos: int) -> ReferenceEyes:
    n = photos * 2
    return ReferenceEyes(
        images=np.zeros((n, eyenorm.SIZE[1], eyenorm.SIZE[0]), np.uint8),
        name=[f"{i // 2:05d}.webp" for i in range(n)],
        openness=np.full(n, 0.3),
        head_yaw_deg=np.zeros(n),
        head_pitch_deg=np.zeros(n),
        iris=np.zeros((n, 2)),
        points=np.zeros((n, 5, 2)),
        head=np.zeros((n, 2)),
    )


def test_photos_are_read_from_the_tar_file_and_nothing_is_unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "reference_dir", lambda: tmp_path)
    ok, data = cv2.imencode(".png", np.full((8, 8, 3), 200, np.uint8))
    with tarfile.open(tmp_path / "00000.tar", "w") as tar:
        for name, payload in (("00001.png", data.tobytes()), ("notes.txt", b"not an image"), ("00002.png", b"broken")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    found = list(reference.photos())
    assert [name for name, _ in found] == ["00001.png"] and found[0][1].shape == (8, 8, 3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00000.tar"]


def test_person_gaze_is_the_mean_of_the_two_eyes():
    # Model frame: yaw is positive towards the nose. Two eyes that both turn 5 degrees to the nose cancel.
    eyes = np.array([[[5.0, 2.0], [5.0, 4.0]], [[8.0, 0.0], [-2.0, 0.0]]])
    assert reference.person_gaze(eyes) == pytest.approx(np.array([[0.0, 3.0], [5.0, 0.0]]))


def test_portrait_table_splits_by_glasses_and_shows_each_eye():
    ref = _eyes(60)
    rng = np.random.default_rng(0)
    eyes = rng.normal(0.0, 1.0, (60, 2, 2))
    eyes[:, :, 0] += 7.0  # each eye reads 7 degrees towards the nose
    eyes[30:, :, 1] += 4.0  # the persons with glasses read 4 degrees up
    glasses = {name: ("NoGlasses" if i < 30 else "ReadingGlasses") for i, name in enumerate(ref.name[::2])}
    text = reference.format_portrait_gaze(ref, eyes, glasses)
    rows = {line[:28].strip(): line[28:].split() for line in text.splitlines() if line[:2].strip() and line[28:].strip()}
    assert rows["no glasses"][0] == "30" and abs(float(rows["no glasses"][2])) < 1.0
    assert 3.0 < float(rows["glasses"][2]) < 5.0
    assert abs(float(rows["all (no sunglasses)"][1])) < 1.0  # the nose-side error cancels in the mean of the eyes
    assert "image-left eye +7." in text or "image-left eye +6." in text


def test_eye_rect_holds_both_eyes_with_a_margin():
    landmarks = np.zeros((478, 3), np.float32)
    for index, (x, y) in {33: (0.30, 0.40), 133: (0.42, 0.41), 362: (0.58, 0.41), 263: (0.70, 0.40)}.items():
        landmarks[index, :2] = (x, y)
    x0, y0, x1, y1 = reference._eye_rect(landmarks, (1000.0, 1000.0))
    assert x0 < 300 - 100 and x1 > 700 + 100 and y0 < 400 - 100 and y1 > 410 + 100
    assert reference._eye_rect(landmarks, (100.0, 100.0))[0] >= 0


def test_realism_meter_finds_the_cell_that_differs():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    real = rng.integers(90, 160, (400, 64, 96)).astype(np.uint8)
    synth = rng.integers(90, 160, (400, 64, 96)).astype(np.uint8)
    synth[:, 0:16, 72:96] //= 2  # the top cell on the nose side is darker in the synthetic images
    groups = np.repeat(np.arange(100), 4)
    out = reference.realism(real, groups, synth, groups, epochs=3, log=lambda *a: None)
    assert out["cells"].shape == (4, 4)
    assert out["cells"][0, 3] > 0.9 and out["cells"][3, 0] < 0.7


def test_tones_read_the_parts_of_a_drawn_eye():
    img = np.full((64, 96), 120, np.uint8)  # skin
    cv2.ellipse(img, (48, 30), (28, 8), 0, 0, 360, 200, -1)  # the white of the eye
    cv2.circle(img, (48, 30), 9, 30, -1)  # the iris
    points = np.array([[[48.0, 30.0], [20.0, 30.0], [76.0, 30.0], [48.0, 22.0], [48.0, 38.0]]])
    ref = _eyes(1)
    ref = ReferenceEyes(img[None], ["a"], ref.openness[:1], ref.head_yaw_deg[:1], ref.head_pitch_deg[:1], ref.iris[:1], points, ref.head[:1])
    t = reference.tones(ref)[0]
    parts = dict(zip(reference.TONE_PARTS, t))
    assert parts["iris"] == pytest.approx(30 / 120, abs=0.02)
    assert parts["white, temple side"] == pytest.approx(200 / 120, abs=0.05)
    assert parts["white, nose side"] == pytest.approx(200 / 120, abs=0.05)
    assert parts["upper lid skin"] == pytest.approx(1.0, abs=0.05)
    assert "iris" in reference.format_tones({"drawn": ReferenceEyes(np.repeat(img[None], 12, 0), ["a"] * 12, *[np.zeros(12)] * 3, np.zeros((12, 2)), np.repeat(points, 12, 0), np.zeros((12, 2)))})
