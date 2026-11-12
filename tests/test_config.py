import tomllib

import pytest

from handsfree import config
from handsfree.config import Config, ConfigError


def test_defaults_are_valid():
    assert Config().validate().camera.fps == 60


def test_partial_override_keeps_other_defaults():
    cfg = config.from_dict({"camera": {"fps": 30}, "preview": {"gain_x": 2}})
    assert cfg.camera.fps == 30
    assert cfg.camera.width == 640
    assert cfg.preview.gain_x == 2.0 and isinstance(cfg.preview.gain_x, float)


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="camera.fpss"):
        config.from_dict({"camera": {"fpss": 30}})
    with pytest.raises(ConfigError, match="cammera"):
        config.from_dict({"cammera": {}})


@pytest.mark.parametrize(
    "data",
    [
        {"camera": {"fps": "fast"}},
        {"camera": {"fps": True}},
        {"camera": {"fps": 0}},
        {"camera": {"fourcc": "MJPEG"}},
        {"face": {"delegate": "npu"}},
        {"preview": {"mirror": 1}},
        {"camera": 5},
    ],
)
def test_bad_values_are_errors(data):
    with pytest.raises(ConfigError):
        config.from_dict(data)


def test_toml_round_trip():
    cfg = config.from_dict({"camera": {"device": "/dev/video2", "fps": 30}, "preview": {"mirror": False}})
    text = config.to_toml(cfg)
    assert "# V4L2 device path" in text
    assert config.from_dict(tomllib.loads(text)) == cfg


def test_load_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.load() == Config()
    with pytest.raises(ConfigError, match="not found"):
        config.load(tmp_path / "nope.toml")
    p = tmp_path / "handsfree" / "config.toml"
    p.parent.mkdir()
    p.write_text("[gaze]\nblink_openness = 0.2\n")
    assert config.load().gaze.blink_openness == 0.2
