"""User configuration, loaded from one TOML file.

The default path is ``$XDG_CONFIG_HOME/handsfree/config.toml``
(usually ``~/.config/handsfree/config.toml``). Every key is optional; a
missing key keeps its default. Unknown keys and wrong types are errors, so a
typo never gets ignored silently.
"""

from __future__ import annotations

import os
import tomllib
import typing
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    device: str = "/dev/video0"
    """V4L2 device path, or a camera index such as "0"."""
    width: int = 640
    height: int = 480
    fps: int = 60
    fourcc: str = "MJPG"
    """Pixel format. MJPG gives the highest frame rates on most webcams."""
    fixed_framerate: bool = True
    """Stop auto-exposure from lowering the frame rate in dim light."""
    vertical_fov_deg: float = 0.0
    """Real vertical field of view of the camera, in degrees. 0 means unknown: use the face tracker's assumption (63)."""


@dataclass(frozen=True)
class FaceConfig:
    model_path: str = ""
    """Path to face_landmarker.task. Empty means the model cache (see `handsfree fetch-models`)."""
    delegate: str = "cpu"
    """"cpu" or "gpu"."""
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class GazeConfig:
    estimator: str = "landmark"
    """Gaze estimator. Only "landmark" exists for now."""
    blink_openness: float = 0.12
    """An eye with lid gap / eye width below this value counts as closed."""


@dataclass(frozen=True)
class PreviewConfig:
    mirror: bool = True
    """Show the camera image like a mirror."""
    gain_x: float = 10.0
    """Uncalibrated view only: screen widths per eye width of horizontal iris movement."""
    gain_y: float = 26.0
    """Uncalibrated view only: screen heights per eye width of vertical iris movement."""
    trail_length: int = 20
    """Number of past gaze points drawn behind the current one."""


@dataclass(frozen=True)
class CalibrationConfig:
    model: str = "auto"
    """"auto" picks the model with the lowest cross-check error. Or give a model name, for example "blendshape-linear"."""
    settle_s: float = 1.0
    """Seconds to wait after a target appears, before recording starts."""
    record_s: float = 1.0
    """Seconds of recording for each target."""
    prompt_read_s: float = 5.0
    """Seconds to read an on-screen instruction before its recording starts."""
    head_phase: bool = False
    """Experimental. After the targets, record head movements and train models that use head pose. In tests so far this made the result worse."""


@dataclass(frozen=True)
class RecordingConfig:
    width: int = 1920
    """Camera width for `handsfree record`. Recordings use the full camera resolution."""
    height: int = 1080
    settle_s: float = 1.0
    """Seconds to wait after a target appears, before recording starts."""
    record_s: float = 1.0
    """Seconds of recording for each target."""


@dataclass(frozen=True)
class ScreenConfig:
    viewing_distance_cm: float = 0.0
    """Eye-to-screen distance, used to convert pixels to degrees. 0 means: use the head tracker's estimate."""
    camera_above_mm: float = 10.0
    """Distance from the camera lens centre down to the top edge of the screen picture."""


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    def validate(self) -> Config:
        c = self.camera
        _check(c.width > 0 and c.height > 0, "camera.width and camera.height must be > 0")
        _check(c.fps > 0, "camera.fps must be > 0")
        _check(len(c.fourcc) == 4, "camera.fourcc must have exactly 4 characters")
        _check(self.face.delegate in ("cpu", "gpu"), 'face.delegate must be "cpu" or "gpu"')
        for name in ("min_detection_confidence", "min_presence_confidence", "min_tracking_confidence"):
            v = getattr(self.face, name)
            _check(0.0 <= v <= 1.0, f"face.{name} must be between 0 and 1")
        _check(self.gaze.estimator in ("landmark",), 'gaze.estimator must be "landmark"')
        _check(0.0 < self.gaze.blink_openness < 1.0, "gaze.blink_openness must be between 0 and 1")
        _check(self.preview.trail_length >= 0, "preview.trail_length must be >= 0")
        from .calibration import CANDIDATES

        models = ["auto", *(c.name for c in CANDIDATES)]
        _check(self.calibration.model in models, f"calibration.model must be one of: {', '.join(models)}")
        _check(self.calibration.settle_s >= 0.2, "calibration.settle_s must be >= 0.2")
        _check(self.calibration.record_s >= 0.3, "calibration.record_s must be >= 0.3")
        _check(self.calibration.prompt_read_s >= 0, "calibration.prompt_read_s must be >= 0")
        _check(self.recording.width > 0 and self.recording.height > 0, "recording.width and recording.height must be > 0")
        _check(self.recording.settle_s >= 0.2, "recording.settle_s must be >= 0.2")
        _check(self.recording.record_s >= 0.3, "recording.record_s must be >= 0.3")
        _check(self.screen.viewing_distance_cm >= 0, "screen.viewing_distance_cm must be >= 0")
        _check(self.screen.camera_above_mm >= 0, "screen.camera_above_mm must be >= 0")
        _check(0 <= c.vertical_fov_deg < 170, "camera.vertical_fov_deg must be 0 (unknown) or between 0 and 170")
        return self


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ConfigError(msg)


def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "handsfree" / "config.toml"


def load(path: Path | None = None) -> Config:
    """Load the config file. A missing file at the default path gives the defaults."""
    explicit = path is not None
    path = path or default_path()
    if not path.exists():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return Config()
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    return from_dict(data)


def from_dict(data: dict[str, Any]) -> Config:
    return _build(Config, data, "").validate()


def _build(cls: type, data: Any, prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{prefix.rstrip('.')}] must be a table")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s): {', '.join(prefix + k for k in sorted(unknown))}")
    values: dict[str, Any] = {}
    for name, raw in data.items():
        typ = hints[name]
        key = prefix + name
        if is_dataclass(typ):
            values[name] = _build(typ, raw, key + ".")
        else:
            values[name] = _coerce(typ, raw, key)
    return replace(cls(), **values)


def _coerce(typ: type, raw: Any, key: str) -> Any:
    # bool is a subclass of int in Python; do not accept it for numbers.
    if typ is bool:
        ok = isinstance(raw, bool)
    elif typ is int:
        ok = isinstance(raw, int) and not isinstance(raw, bool)
    elif typ is float:
        ok = isinstance(raw, (int, float)) and not isinstance(raw, bool)
        raw = float(raw) if ok else raw
    elif typ is str:
        ok = isinstance(raw, str)
    else:  # pragma: no cover - only reached if a new field type is added
        raise ConfigError(f"{key}: unsupported field type {typ}")
    if not ok:
        raise ConfigError(f"{key} must be of type {typ.__name__}, not {type(raw).__name__}")
    return raw


def to_toml(cfg: Config) -> str:
    """Write the config as TOML, with each field's doc comment."""
    out: list[str] = []
    for sec in fields(cfg):
        section = getattr(cfg, sec.name)
        out.append(f"[{sec.name}]")
        docs = _field_docs(type(section))
        for f in fields(section):
            if f.name in docs:
                out.append(f"# {docs[f.name]}")
            out.append(f"{f.name} = {_toml_value(getattr(section, f.name))}")
        out.append("")
    return "\n".join(out)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(v)


def _field_docs(cls: type) -> dict[str, str]:
    """Read the attribute docstrings that follow each field in the source."""
    import ast
    import inspect

    try:
        tree = ast.parse(inspect.getsource(cls))
    except (OSError, TypeError):
        return {}
    body = tree.body[0].body  # type: ignore[attr-defined]
    docs: dict[str, str] = {}
    for prev, node in zip(body, body[1:]):
        if (
            isinstance(prev, ast.AnnAssign)
            and isinstance(prev.target, ast.Name)
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docs[prev.target.id] = " ".join(node.value.value.split())
    return docs
