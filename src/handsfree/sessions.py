"""Recorded target sessions.

A session holds the gaze feature vectors recorded while the user looked at
each target. It holds numbers only, never camera images. Sessions are saved
under ``~/.local/share/handsfree/sessions/`` so that models can be compared
later without a new recording.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .types import Target


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "handsfree"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True, eq=False)
class TargetSession:
    kind: str
    """"probe", "calibrate" or "accuracy"."""
    created: str
    targets: list[Target]
    samples: list[np.ndarray]
    """``samples[i]`` is an n x len(names) array for target ``i``."""
    times: list[np.ndarray]
    """``times[i]`` holds seconds since the recording of target ``i`` began."""
    names: tuple[str, ...]
    brightness: float
    screen_px: tuple[int, int]
    """Logical pixels."""
    screen_mm: tuple[float, float]
    """Physical size as the display reports it. (0, 0) if unknown."""

    @property
    def file_name(self) -> str:
        stamp = self.created[:19].replace("-", "").replace(":", "").replace("T", "-")
        return f"{self.kind}-{stamp}.npz"

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or sessions_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.file_name
        k = len(self.names)
        np.savez_compressed(
            path,
            kind=np.array(self.kind),
            created=np.array(self.created),
            names=np.array(self.names),
            target_name=np.array([t.name for t in self.targets]),
            target_kind=np.array([t.kind for t in self.targets]),
            target_xy=np.array([(t.x, t.y) for t in self.targets], dtype=float),
            features=np.concatenate([s.reshape(-1, k) for s in self.samples]),
            time=np.concatenate(self.times),
            target_index=np.concatenate([np.full(len(s), i) for i, s in enumerate(self.samples)]),
            brightness=np.array(self.brightness),
            screen_px=np.array(self.screen_px),
            screen_mm=np.array(self.screen_mm),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> TargetSession:
        with np.load(path, allow_pickle=False) as d:
            index = d["target_index"]
            targets = [
                Target(str(n), float(x), float(y), kind=str(kd))
                for n, kd, (x, y) in zip(d["target_name"], d["target_kind"], d["target_xy"])
            ]
            return cls(
                kind=str(d["kind"]),
                created=str(d["created"]),
                targets=targets,
                samples=[d["features"][index == i] for i in range(len(targets))],
                times=[d["time"][index == i] for i in range(len(targets))],
                names=tuple(str(n) for n in d["names"]),
                brightness=float(d["brightness"]),
                screen_px=(int(d["screen_px"][0]), int(d["screen_px"][1])),
                screen_mm=(float(d["screen_mm"][0]), float(d["screen_mm"][1])),
            )


def list_sessions(kind: str, directory: Path | None = None) -> list[Path]:
    """Saved sessions of one kind, oldest first."""
    return sorted((directory or sessions_dir()).glob(f"{kind}-*.npz"))
