"""Signal probe: how strongly does each gaze signal respond to a known eye movement?

The user looks at targets near the screen edges with a still head. For each
gaze signal the report gives:

- signal: the change between two opposite targets (right minus left, or
  bottom minus top). The targets are 80 % of the screen apart.
- noise: the spread (standard deviation) while the user looks at one target.
- ratio: signal divided by noise.
- noise px: the noise, converted to screen pixels with the measured signal.
- centre shift px: the largest difference between the three centre visits.

A signal is useful only if its noise on screen is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .face import LOW_LIGHT
from .types import Target

EDGE = 0.1
SPAN = 1.0 - 2 * EDGE

PROBE_TARGETS = [
    Target("warm-up", 0.5, 0.5, kind="warmup"),
    Target("centre", 0.5, 0.5),
    Target("left", EDGE, 0.5),
    Target("right", 1 - EDGE, 0.5),
    Target("centre", 0.5, 0.5),
    Target("top", 0.5, EDGE),
    Target("bottom", 0.5, 1 - EDGE),
    Target("centre", 0.5, 0.5),
]

X_SIGNALS = ("gaze_x", "bs_x", "head_yaw")
Y_SIGNALS = ("gaze_y", "bs_y", "head_pitch")
MIN_SAMPLES = 10

INTRO = (
    "Signal probe\n\n"
    "8 targets appear, one at a time.\n"
    "Keep your head still. Follow each target with your eyes only.\n"
    "Look at the centre dot of the target until it moves.\n\n"
    "Space: start        Esc: cancel"
)


@dataclass(frozen=True, slots=True)
class SignalStats:
    name: str
    signal: float
    noise: float
    centre_shift: float
    """Largest difference between the centre visits."""

    @property
    def ratio(self) -> float:
        return abs(self.signal) / self.noise if self.noise > 0 else float("inf")

    def noise_px(self, screen_px: int) -> float:
        return self.noise / abs(self.signal) * SPAN * screen_px if self.signal else float("inf")


def analyse(
    targets: list[Target], samples: list[np.ndarray], names: tuple[str, ...]
) -> tuple[list[SignalStats], list[SignalStats]]:
    """Return (horizontal signals, vertical signals). Each ``samples[i]`` is n x len(names)."""
    used = [i for i, t in enumerate(targets) if t.kind == "fixation"]
    targets, samples = [targets[i] for i in used], [samples[i] for i in used]
    for t, s in zip(targets, samples):
        if len(s) < MIN_SAMPLES:
            raise ValueError(f"target '{t.name}' has only {len(s)} samples; the face or eyes were not visible")
    med = [np.median(s, axis=0) for s in samples]
    std = [np.std(s, axis=0) for s in samples]
    by_name: dict[str, list[int]] = {}
    for i, t in enumerate(targets):
        by_name.setdefault(t.name, []).append(i)

    def stats(feature: str, low: str, high: str) -> SignalStats:
        k = names.index(feature)
        signal = float(med[by_name[high][0]][k] - med[by_name[low][0]][k])
        noise = float(np.median([s[k] for s in std]))
        centres = [float(med[i][k]) for i in by_name.get("centre", [])]
        shift = max(centres) - min(centres) if len(centres) > 1 else 0.0
        return SignalStats(feature, signal, noise, shift)

    return (
        [stats(f, "left", "right") for f in X_SIGNALS if f in names],
        [stats(f, "top", "bottom") for f in Y_SIGNALS if f in names],
    )


def format_report(
    targets: list[Target],
    samples: list[np.ndarray],
    names: tuple[str, ...],
    screen: tuple[int, int],
    brightness: float,
) -> str:
    xs, ys = analyse(targets, samples, names)
    shown = [n for n in (*X_SIGNALS, *Y_SIGNALS) if n in names]
    order = sorted(shown, key=lambda n: (n.startswith("head"), n.startswith("bs")))
    out = [f"screen {screen[0]}x{screen[1]} px, targets {SPAN:.0%} of the screen apart"]
    low = "   LOW LIGHT: add light on your face, then repeat" if brightness < LOW_LIGHT else ""
    out += [f"face brightness {brightness:.0f}/255{low}", ""]
    out.append("Median value at each target")
    out.append(f"{'target':<8}" + "".join(f"{n:>11}" for n in order) + f"{'n':>5}")
    for t, s in zip(targets, samples):
        if t.kind != "fixation":
            continue
        m = np.median(s, axis=0)
        out.append(f"{t.name:<8}" + "".join(f"{m[names.index(n)]:>+11.3f}" for n in order) + f"{len(s):>5}")
    for title, rows, px in (
        ("Horizontal (right minus left)", xs, screen[0]),
        ("Vertical (bottom minus top)", ys, screen[1]),
    ):
        out += ["", title]
        out.append(f"{'signal':<12}{'change':>9}{'noise':>9}{'ratio':>8}{'noise px':>11}{'centre shift px':>18}")
        for s in rows:
            shift_px = s.centre_shift / abs(s.signal) * SPAN * px if s.signal else float("inf")
            out.append(
                f"{s.name:<12}{s.signal:>+9.3f}{s.noise:>9.4f}{s.ratio:>8.1f}"
                f"{s.noise_px(px):>11.0f}{shift_px:>18.0f}"
            )
    gx = next((s for s in xs if s.name == "gaze_x"), None)
    gy = next((s for s in ys if s.name == "gaze_y"), None)
    if gx and gy and gx.signal > 0 and gy.signal > 0:
        out += ["", "Preview gains that fit these numbers:", "[preview]"]
        out += [f"gain_x = {SPAN / gx.signal:.1f}", f"gain_y = {SPAN / gy.signal:.1f}"]
    return "\n".join(out)


def run_probe(cfg: Config) -> int:
    from .targets import run_target_session

    session = run_target_session(cfg, "probe", PROBE_TARGETS, INTRO, settle_s=1.0, record_s=1.5)
    if session is None:
        print("probe cancelled")
        return 1
    session.save()
    try:
        print(format_report(session.targets, session.samples, session.names, session.screen_px, session.brightness))
    except ValueError as e:
        print(f"probe failed: {e}")
        return 1
    return 0
