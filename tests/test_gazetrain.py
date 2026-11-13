import numpy as np
import pytest

torch = pytest.importorskip("torch")

from handsfree import eyenorm  # noqa: E402
from handsfree.gazetrain import angular_error_deg, jitter, make_model, predict  # noqa: E402


def test_model_gives_two_angles_for_each_eye_image():
    images = np.random.default_rng(0).integers(0, 255, (5, eyenorm.SIZE[1], eyenorm.SIZE[0]), dtype=np.uint8)
    out = predict(make_model(), images)
    assert out.shape == (5, 2) and np.isfinite(out).all()


def test_jitter_moves_each_image_by_a_different_amount():
    torch.manual_seed(0)
    x = torch.zeros(8, 1, 64, 96)
    x[:, :, 30:34, 46:50] = 1.0  # a mark at the centre
    y = jitter(x, max_shift_px=9.0, max_scale=0.1)
    assert y.shape == x.shape
    cols = [float((img[0].sum(0) * torch.arange(96)).sum() / img.sum()) for img in y]
    rows = [float((img[0].sum(1) * torch.arange(64)).sum() / img.sum()) for img in y]
    assert max(cols) - min(cols) > 4 and max(rows) - min(rows) > 4  # the marks are at different places
    assert all(abs(c - 47.5) < 11 for c in cols) and all(abs(r - 31.5) < 11 for r in rows)


def test_angular_error():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, -20.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 10.0]])
    assert angular_error_deg(a, b) == pytest.approx([0.0, 10.0, 30.0], abs=1e-6)


def test_the_pitch_can_come_from_a_second_model(tmp_path):
    import json

    from handsfree.benchmark import EyeImageModel
    from handsfree.gazetrain import ANGLE_SCALE, export

    for name, answer in (("a.onnx", (1.0, 2.0)), ("b.onnx", (3.0, 4.0))):
        model = make_model()
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.copy_(torch.tensor(answer))  # the model always gives this answer
        export(model, tmp_path / name)
    images = np.random.default_rng(0).integers(0, 255, (4, eyenorm.SIZE[1], eyenorm.SIZE[0]), dtype=np.uint8)
    alone = EyeImageModel(model=str(tmp_path / "a.onnx")).angles(images)
    assert alone == pytest.approx(np.tile([1.0 * ANGLE_SCALE, 2.0 * ANGLE_SCALE], (4, 1)), abs=1e-3)
    (tmp_path / "a.json").write_text(json.dumps({"window": list(eyenorm.WINDOW), "pitch_from": "b.onnx"}))
    pair = EyeImageModel(model=str(tmp_path / "a.onnx")).angles(images)
    assert pair == pytest.approx(np.tile([1.0 * ANGLE_SCALE, 4.0 * ANGLE_SCALE], (4, 1)), abs=1e-3)
