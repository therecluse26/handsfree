import pytest

from handsfree.latency import LatencyTracker, RateMeter


def test_stages_and_total():
    lt = LatencyTracker()
    for i in range(10):
        lt.record({"capture": i, "received": i + 0.010, "start": i + 0.010, "face": i + 0.018, "gaze": i + 0.019})
    s = {x.name: x for x in lt.summary()}
    assert list(s) == ["capture→received", "received→start", "start→face", "face→gaze", "total"]
    assert s["capture→received"].median_ms == pytest.approx(10.0)
    assert s["start→face"].median_ms == pytest.approx(8.0)
    assert s["total"].median_ms == pytest.approx(19.0)
    assert s["total"].count == 10
    assert "face model" in lt.format_table()


def test_window_and_empty():
    lt = LatencyTracker(window=3)
    assert lt.format_table() == "no frames measured yet"
    for ms in (1, 2, 3, 100, 100, 100):
        lt.record({"a": 0.0, "b": ms / 1000})
    assert lt.summary()[0].median_ms == pytest.approx(100.0)


def test_rate_meter():
    m = RateMeter()
    assert m.rate == 0.0
    for i in range(31):
        m.tick(i / 30)
    assert m.rate == pytest.approx(30.0)
