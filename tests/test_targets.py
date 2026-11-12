import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from PySide6.QtWidgets import QApplication  # noqa: E402

from handsfree.gaze import FEATURE_NAMES, LandmarkGazeEstimator  # noqa: E402
from handsfree.pipeline import Pipeline  # noqa: E402
from handsfree.targets import TargetSequenceWindow  # noqa: E402
from handsfree.types import Frame, Target  # noqa: E402

from test_gaze import make_face  # noqa: E402
from test_pipeline import FakeTracker  # noqa: E402


class EndlessSource:
    def __init__(self):
        self.seq = 0

    def read(self):
        import time

        time.sleep(0.004)
        self.seq += 1
        t = time.monotonic()
        return Frame(np.full((480, 640, 3), 90, np.uint8), self.seq, t, t)

    def close(self):
        pass


def test_sequence_records_samples_for_each_target():
    app = QApplication.instance() or QApplication([])
    pipeline = Pipeline(EndlessSource(), FakeTracker(make_face(iris_dx=2.0)), LandmarkGazeEstimator())
    targets = [Target("a", 0.1, 0.5), Target("b", 0.9, 0.5, "head", 0.3, "Move your head.")]
    seen = []
    w = TargetSequenceWindow(
        pipeline,
        targets,
        "intro",
        settle_s=0.05,
        record_s=0.15,
        autostart=True,
        prompt_read_s=0.4,
        on_frame=lambda r, i, since: seen.append((i, since)),
    )
    t0 = time.monotonic()
    w.resize(800, 500)
    w.show()
    from PySide6.QtCore import QTimer

    guard = QTimer()  # stops the test if the window never closes
    guard.setSingleShot(True)
    guard.timeout.connect(app.quit)
    guard.start(5000)
    app.exec()  # returns when the window closes itself
    guard.stop()
    assert w.completed
    assert time.monotonic() - t0 >= 0.05 + 0.15 + 0.4 + 0.3  # the prompt target waits for the read time
    assert w.feature_names == FEATURE_NAMES
    assert all(len(s) >= 5 for s in w.samples)
    assert len(w.samples[1]) > 1.4 * len(w.samples[0])  # the head target records for 0.3 s, not 0.15 s
    assert [len(t) for t in w.times] == [len(s) for s in w.samples]
    assert [sum(i == k for i, _ in seen) for k in (0, 1)] == [len(s) for s in w.samples]  # the frame hook saw each sample
    assert 0 <= w.times[1][0] < 0.1 and 0.2 < w.times[1][-1] <= 0.35
    assert w.brightness and abs(w.brightness[0] - 90) < 1


def test_calibrate_then_accuracy_commands(tmp_path, monkeypatch, capsys):
    """The two commands, end to end, with a fake camera and a fake face."""
    import dataclasses
    import functools

    from handsfree import accuracy, calibration, targets
    from handsfree.config import CalibrationConfig, Config

    QApplication.instance() or QApplication([])
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(
        targets.Pipeline,
        "from_config",
        classmethod(lambda cls, cfg: Pipeline(EndlessSource(), FakeTracker(make_face()), LandmarkGazeEstimator())),
    )
    monkeypatch.setattr(targets, "run_target_session", functools.partial(targets.run_target_session, autostart=True))
    monkeypatch.setattr(calibration, "HEAD_TARGETS", [dataclasses.replace(t, record_s=1.2) for t in calibration.HEAD_TARGETS])
    cfg = dataclasses.replace(Config(), calibration=CalibrationConfig(settle_s=0.02, record_s=0.25, head_phase=True, prompt_read_s=0.1))

    assert accuracy.run_accuracy(cfg) == 1  # no calibration yet
    assert calibration.run_calibrate(cfg) == 0
    assert (tmp_path / "handsfree" / "calibration.json").exists()
    assert accuracy.run_accuracy(cfg) == 0
    assert accuracy.run_evaluate() == 0
    assert calibration.run_calibrate(Config(), refit=True) == 0  # head phase off: eye-only models
    assert not calibration.Calibration.load().model.spec.head
    out = capsys.readouterr().out
    assert "no usable calibration found" in out and "<- in use" in out and "mean offset" in out
    assert "accuracy sessions   1" in out
    assert len((tmp_path / "handsfree" / "accuracy.jsonl").read_text().splitlines()) == 1
    assert len(list((tmp_path / "handsfree" / "sessions").glob("*.npz"))) == 2


def test_record_command(tmp_path, monkeypatch, capsys):
    """`handsfree record`, end to end, with a fake camera and a fake face."""
    import dataclasses
    import functools

    from handsfree import recording, targets
    from handsfree.config import Config, RecordingConfig

    QApplication.instance() or QApplication([])
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    seen_cfg = []

    def fake_pipeline(cls, cfg):
        seen_cfg.append(cfg.camera)
        return Pipeline(EndlessSource(), FakeTracker(make_face()), LandmarkGazeEstimator())

    monkeypatch.setattr(targets.Pipeline, "from_config", classmethod(fake_pipeline))
    monkeypatch.setattr(targets, "run_target_session", functools.partial(targets.run_target_session, autostart=True))
    cfg = dataclasses.replace(Config(), recording=RecordingConfig(settle_s=0.02, record_s=0.12))
    assert recording.run_record(cfg) == 0
    assert (seen_cfg[0].width, seen_cfg[0].height) == (1920, 1080)  # the recording asks for the full resolution
    (directory,) = recording.list_recordings()
    rec = recording.Recording.load(directory)
    assert len(rec) > 25 * 3 and set(rec.frames["target_index"]) == set(range(26))
    assert rec.eyes().shape[1:] == (2, 128, 192, 3)
    assert "size on disk" in capsys.readouterr().out
    assert len(list((tmp_path / "handsfree" / "sessions").glob("record-*.npz"))) == 1
