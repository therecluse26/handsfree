import numpy as np

from handsfree.gaze import LandmarkGazeEstimator
from handsfree.pipeline import Pipeline
from handsfree.types import Frame

from test_gaze import make_face


class FakeSource:
    def __init__(self, n):
        self.n, self.closed = n, False

    def read(self):
        if self.n == 0:
            return None
        self.n -= 1
        return Frame(np.zeros((480, 640, 3), np.uint8), seq=1, t_capture=1.0, t_received=1.01)

    def close(self):
        self.closed = True


class FakeTracker:
    def __init__(self, face):
        self.face, self.closed = face, False

    def process(self, frame):
        return self.face

    def close(self):
        self.closed = True


def test_step_with_face_and_without():
    src = FakeSource(1)
    p = Pipeline(src, FakeTracker(make_face()), LandmarkGazeEstimator())
    r = p.step()
    assert r.gaze is not None and list(r.marks) == ["capture", "received", "start", "face", "gaze"]
    assert p.step() is None  # the source has no more frames

    p2 = Pipeline(FakeSource(1), FakeTracker(None), LandmarkGazeEstimator())
    r2 = p2.step()
    assert r2.face is None and r2.gaze is None
    p.close()
    assert src.closed and p.tracker.closed
