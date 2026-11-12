"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__, config as config_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="handsfree", description=__doc__)
    parser.add_argument("--version", action="version", version=f"handsfree {__version__}")
    parser.add_argument("--config", type=Path, help="config file (default: ~/.config/handsfree/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-models", help="download the face model once (about 4 MB)")
    p = sub.add_parser("preview", help="live camera view with the gaze estimate")
    p.add_argument("--seconds", type=float, help="close the window after this many seconds")
    sub.add_parser("probe", help="show 7 targets and measure how strongly each gaze signal responds")
    p = sub.add_parser("calibrate", help="look at 14 targets to fit the gaze-to-screen map")
    p.add_argument("--refit", action="store_true", help="fit the latest recorded session again; needs no camera")
    sub.add_parser("accuracy", help="look at 12 targets and measure the error of the calibration")
    sub.add_parser("evaluate", help="compare all models on the recorded sessions (no camera)")
    p = sub.add_parser("bench", help="measure frame rate and latency without a window")
    p.add_argument("--seconds", type=float, default=5.0)
    p = sub.add_parser("config", help="print the active config as TOML")
    p.add_argument("--defaults", action="store_true", help="print the defaults instead")

    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            cfg = config_mod.Config() if args.defaults else config_mod.load(args.config)
            print(f"# handsfree config. Default location: {config_mod.default_path()}\n")
            print(config_mod.to_toml(cfg), end="")
            return 0
        if args.command == "fetch-models":
            from .models import fetch_face_landmarker

            print(f"face model ready: {fetch_face_landmarker()}")
            return 0
        cfg = config_mod.load(args.config)
        if args.command == "preview":
            from .preview import run_preview

            return run_preview(cfg, args.seconds)
        if args.command == "probe":
            from .probe import run_probe

            return run_probe(cfg)
        if args.command == "calibrate":
            from .calibration import run_calibrate

            return run_calibrate(cfg, args.refit)
        if args.command == "accuracy":
            from .accuracy import run_accuracy

            return run_accuracy(cfg)
        if args.command == "evaluate":
            from .accuracy import run_evaluate

            return run_evaluate()
        if args.command == "bench":
            return _bench(cfg, args.seconds)
    except (config_mod.ConfigError, RuntimeError) as e:
        print(f"handsfree: {e}", file=sys.stderr)
        return 1
    return 0


def _bench(cfg: config_mod.Config, seconds: float) -> int:
    from .latency import LatencyTracker
    from .pipeline import Pipeline

    pipeline = Pipeline.from_config(cfg)
    latency = LatencyTracker(window=100_000)
    frames = faces = blinks = 0
    try:
        for _ in range(10):  # let the camera settle
            pipeline.step()
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            r = pipeline.step()
            if r is None:
                continue
            frames += 1
            latency.record(r.marks)
            if r.gaze is not None:
                faces += 1
                blinks += r.gaze.blinking
        elapsed = time.monotonic() - t0
    finally:
        pipeline.close()
    cam = pipeline.source
    print(f"camera      {getattr(cam, 'width', '?')}x{getattr(cam, 'height', '?')}, asked for {cfg.camera.fps} fps")
    print(f"frame rate  {frames / elapsed:.1f} fps ({frames} frames in {elapsed:.1f} s)")
    print(f"face found  {faces}/{frames} frames, eyes closed in {blinks}")
    print(latency.format_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
