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
    sub.add_parser("record", help="record a labelled data set: 26 targets, with eye and face images, stored locally")
    sub.add_parser("evaluate", help="compare all models on the recorded sessions (no camera)")
    p = sub.add_parser("benchmark", help="score gaze estimators on the recordings, with no calibration (no camera)")
    p.add_argument("estimators", nargs="*", help="estimator names (default: all)")
    p = sub.add_parser("train", help="train the two eye-image gaze models on a synthetic set (needs: uv sync --group train)")
    p.add_argument("--set", default="default", help="name of the synthetic set; several names with commas")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--copies", type=int, default=3, help="simulated-camera versions of each render")
    p.add_argument("--seeds", type=int, default=1, help="trainings of each model; the estimator uses their mean")
    p.add_argument("--head", action="store_true", help="the models also get the head direction from the face tracker")
    p = sub.add_parser("synth", help="synthetic training data from Blender")
    ssub = p.add_subparsers(dest="synth_command", required=True)
    ssub.add_parser("setup", help="install the Blender add-on and download the CC0 character assets (about 270 MB)")
    g = ssub.add_parser("generate", help="render labelled synthetic eye images")
    g.add_argument("--name", default="default", help="name of the set (a folder in ~/.local/share/handsfree/synth/)")
    g.add_argument("--people", type=int, default=25, help="persons for each job")
    g.add_argument("--poses", type=int, default=8, help="images for each person")
    g.add_argument("--jobs", type=int, default=4, help="Blender processes that run at the same time")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--face", action="store_true", help="also render the full face of each sample (slower)")
    g = ssub.add_parser("sheet", help="write one image with many eye crops of a set, for a visual check")
    g.add_argument("--name", default="default")
    g.add_argument("--count", type=int, default=24)
    g.add_argument("--camera", action="store_true", help="apply the camera simulation")
    p = sub.add_parser("reference", help="measurements on images of many real persons (FFHQ photos, MPIIFaceGaze; never used for training)")
    rsub = p.add_subparsers(dest="reference_command", required=True)
    g = rsub.add_parser("fetch", help="download the reference photos")
    g.add_argument("--labelled", action="store_true", help="download the labelled set instead (MPIIFaceGaze, 942 MB)")
    g = rsub.add_parser("labelled", help="error, constant error and gain of the eye-image model on 15 persons with known gaze targets")
    g.add_argument("--model", default="", help="ONNX file (default: the current model)")
    g = rsub.add_parser("gaze", help="constant gaze error of the eye-image model on persons who look into the camera")
    g.add_argument("--model", default="", help="ONNX file (default: the current model)")
    g = rsub.add_parser("tone", help="brightness of the iris, the whites and the lids against the skin: photos, recordings, synthetic sets")
    g.add_argument("--set", action="append", help="a synthetic set with face renders (synth generate --face); can be given more than once")
    g = rsub.add_parser("realism", help="where a classifier can tell synthetic eyes from real ones (needs: uv sync --group train)")
    g.add_argument("--set", default="default", help="name of the synthetic set; several names with commas")
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
        if args.command == "record":
            from .recording import run_record

            return run_record(cfg)
        if args.command == "evaluate":
            from .accuracy import run_evaluate

            return run_evaluate()
        if args.command == "train":
            from .gazetrain import run_train

            return run_train(args)
        if args.command == "synth":
            from .synth import run as run_synth

            return run_synth(args)
        if args.command == "reference":
            from .reference import run as run_reference

            return run_reference(args)
        if args.command == "benchmark":
            from .benchmark import run_benchmark

            return run_benchmark(cfg, args.estimators)
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
