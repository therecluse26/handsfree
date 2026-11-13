import zipfile

import numpy as np
import pytest

from handsfree import eyenorm, labelled
from handsfree.labelled import LabelledEyes

FX, FY, CX, CY = 960.0, 950.0, 650.0, 350.0


def annotation(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n):
        centre = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(450, 700)])
        u, v = FX * centre[0] / centre[2] + CX, FY * centre[1] / centre[2] + CY
        points = np.array([(u + dx, v + dy) for dx, dy in ((-40, -10), (-15, -10), (15, -10), (40, -10), (-20, 20), (20, 20))])
        cells = [f"day01/{i:04d}.jpg", 640, 400, *points.ravel(), *np.zeros(6), *centre, 10.0, 120.0, 0.0, "left"]
        lines.append(" ".join(str(c) for c in cells))
    return "\n".join(lines) + "\n"


def make_zip(tmp_path):
    path = tmp_path / "MPIIFaceGaze.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("MPIIFaceGaze/p00/p00.txt", annotation(50))
        z.writestr("MPIIFaceGaze/p01/p01.txt", annotation(40, seed=1))
        z.writestr("MPIIFaceGaze/readme.txt", "not an annotation")
    return path


def test_rows_are_read_from_the_zip_file_for_each_person(tmp_path):
    with zipfile.ZipFile(make_zip(tmp_path)) as z:
        rows = labelled.rows(z)
    assert list(rows) == ["p00", "p01"] and len(rows["p00"]) == 50 and len(rows["p01"]) == 40
    member, numbers = rows["p00"][3]
    assert member == "MPIIFaceGaze/p00/day01/0003.jpg" and numbers.shape == (26,)
    assert numbers[23:26].tolist() == [10.0, 120.0, 0.0]  # the gaze target
    assert not list(tmp_path.glob("MPIIFaceGaze/*"))  # nothing is unpacked


def test_the_camera_comes_from_the_annotation(tmp_path):
    with zipfile.ZipFile(make_zip(tmp_path)) as z:
        assert labelled.camera(labelled.rows(z)["p00"]) == pytest.approx((FX, FY, CX, CY), rel=1e-6)


def test_frame_gaze_is_the_mean_of_the_two_eyes():
    # The image-right eye is mirrored: its yaw has the other sign.
    assert labelled.frame_gaze(np.array([[4.0, -10.0], [-6.0, -12.0]])) == pytest.approx(np.array([[5.0, -11.0]]))


def test_the_table_shows_bias_and_gain_for_each_person():
    rng = np.random.default_rng(0)
    frames = 200
    true = np.stack([rng.uniform(-15, 15, frames), rng.uniform(-25, 0, frames)], axis=1)
    both = np.repeat(true, 2, axis=0)
    both[1::2, 0] *= -1  # the mirrored eye
    n = 2 * frames
    lab = LabelledEyes(np.zeros((n, eyenorm.SIZE[1], eyenorm.SIZE[0]), np.uint8), np.repeat([0, 1], frames), both, np.zeros((n, 2)), np.ones(n))
    pred = both.copy()
    pred[: frames, 1] = 0.5 * both[: frames, 1] + 1.0  # person 0: half of the vertical movement
    text = labelled.format_gaze(lab, pred)
    p0, p1 = (next(line for line in text.splitlines() if line.startswith(p)).split() for p in ("p00", "p01"))
    # columns: person, frames, true pitch, error, bias yaw, bias pitch, gain yaw, gain pitch, rest, distance
    assert float(p0[7]) == pytest.approx(0.50) and float(p0[6]) == pytest.approx(1.00)
    assert float(p0[5]) > 3 and float(p0[2]) < -5  # the bias is upwards, because the targets are below the camera
    assert float(p1[3]) == pytest.approx(0.0, abs=0.05) and float(p1[7]) == pytest.approx(1.00)
