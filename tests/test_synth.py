import json

import numpy as np
import pytest

from handsfree.camsim import CameraSim
from handsfree.recording import EYE_CROP
from handsfree.synth import SynthSet, contact_sheet

cv2 = pytest.importorskip("cv2")


def make_set(tmp_path, n=3):
    rng = np.random.default_rng(0)
    for seed in (1, 2):
        lines = []
        for i in range(n):
            w, h, x0, y0 = 480, 192, 704, 400
            img = rng.integers(40, 90, (h, w, 3), dtype=np.uint8)
            img[60:70, 150:160] = 255  # a mark at camera pixel (854..863, 460..469)
            name = f"s{seed:04d}_{i}.png"
            cv2.imwrite(str(tmp_path / name), img)
            lines.append(json.dumps({"image": name, "rect": [x0, y0, w, h], "crop_centres_px": [[859.0, 465.0], [1030.0, 470.0]]}))
        (tmp_path / f"labels-{seed:04d}.jsonl").write_text("\n".join(lines) + "\n")
    return SynthSet.load(tmp_path)


def test_load_reads_every_label_file(tmp_path):
    s = make_set(tmp_path)
    assert len(s) == 6 and s.image(0).shape == (192, 480, 3)


def test_eye_crops_are_centred_on_the_labels(tmp_path):
    s = make_set(tmp_path)
    crops, origins = s.eye_crops(0)
    assert crops.shape == (2, EYE_CROP[1], EYE_CROP[0], 3) and crops.dtype == np.uint8
    assert origins[0].tolist() == [859 - 96, 465 - 64]
    ys, xs = np.nonzero(crops[0][..., 0] == 255)
    assert abs(xs.mean() - 95.5) < 1 and abs(ys.mean() - 63.5) < 1  # the mark is at the crop centre


def test_jitter_moves_the_crop_and_camera_simulation_changes_pixels(tmp_path):
    s = make_set(tmp_path)
    rng = np.random.default_rng(3)
    _, origins = s.eye_crops(0, rng=rng, jitter_px=8.0)
    assert origins[0].tolist() != [859 - 96, 465 - 64] and np.abs(origins[0] - [859 - 96, 465 - 64]).max() <= 8
    clean, _ = s.eye_crops(0)
    noisy, _ = s.eye_crops(0, CameraSim(), np.random.default_rng(0))
    assert noisy.shape == clean.shape and 0.5 < np.abs(noisy.astype(int) - clean).mean() < 40


def test_contact_sheet(tmp_path):
    sheet = contact_sheet(make_set(tmp_path), 5, sim=True)
    assert sheet.shape == (2 * EYE_CROP[1], 3 * 2 * EYE_CROP[0], 3)
