# handsfree

A hands-free mouse for Linux that uses a regular webcam.

- **Gaze moves the cursor.** No eye-tracking hardware.
- **A facial gesture clicks.** The trigger method is replaceable.
- **Fully local.** No video, audio or data leaves the machine. The only network
  use is a one-time download of the face model (about 4 MB).
- **X11 and Wayland.** Cursor control goes through a kernel virtual pointer (`uinput`).

> **Status: pre-alpha, step 2 of the roadmap.** The camera, the face tracker, the
> gaze estimator, the live preview, calibration and the accuracy test work.
> The program does not move the cursor or click yet.

## Roadmap

| Step | Content | State |
|---|---|---|
| 1 | Camera input, face tracking, gaze features, live preview, latency report | done |
| 2 | Signal probe, calibration, accuracy test screen with targets | done |
| 3 | Cursor movement (`uinput`), fixation/jump filter, head fine-tune | next |
| 4 | Click trigger (facial gesture), click freeze, safety limits | |
| 5 | Pause/resume shortcut, physical-input takeover helper, drift correction | |

Not in scope yet, but the design keeps them possible: right-click, double-click,
drag, scroll, voice commands, snapping to UI elements, more trigger methods.

## Requirements

- Linux, x86_64 or aarch64
- A webcam (V4L2)
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12 or newer (`uv` downloads one if necessary)

## Setup

```sh
git clone <this repo> && cd handsfree
uv sync
uv run handsfree fetch-models    # one-time download, SHA-256 checked
uv run handsfree preview         # check that the green rings follow your irises
uv run handsfree calibrate       # look at 14 targets, about 30 seconds
uv run handsfree accuracy        # look at 12 targets, get the error in px and degrees
```

### NixOS

The PyPI wheels need shared libraries that NixOS does not put in standard
paths. The flake gives a dev shell that has them:

```sh
nix develop          # or: direnv allow
uv sync
uv run handsfree fetch-models
uv run handsfree preview
```

## Commands

| Command | Function |
|---|---|
| `handsfree preview` | Live camera view with eye overlays, enlarged eye views, the uncalibrated gaze point, gesture scores, light level and latency. |
| `handsfree probe` | Shows 7 targets and measures how strongly each gaze signal responds, and how noisy it is. Prints numbers only. |
| `handsfree calibrate` | Shows 14 targets and fits the map from gaze signals to the screen. Compares 6 models and uses the best one. `--refit` fits the latest recording again. |
| `handsfree evaluate` | Compares all models on the recorded sessions. Needs no camera. |
| `handsfree accuracy` | Shows 12 other targets and reports accuracy (offset) and precision (spread) in pixels and degrees. Appends each run to `~/.local/share/handsfree/accuracy.jsonl`. |
| `handsfree record` | Records a labelled data set for the gaze benchmark: 26 targets at 1920x1080, with eye crops (raw camera pixels) and a small face crop for each frame. Stored locally only, in `~/.local/share/handsfree/recordings/`. |
| `handsfree benchmark [NAME ...]` | Scores gaze estimators on all recordings, with no calibration. Needs no camera. |
| `handsfree bench [--seconds N]` | Measures frame rate and latency with no window. |
| `handsfree fetch-models` | Downloads the face model to `~/.cache/handsfree/models/`. |
| `handsfree config [--defaults]` | Prints the active (or default) config as TOML. |

Preview keys: `Q` or `Esc` quits. `C` re-centres the gaze point (look at the
screen centre first). `M` toggles the mirror view.

Before calibration, the gaze point in the preview is **uncalibrated**. It
shows that the signal follows your eyes, not where you look. After
`handsfree calibrate`, the preview uses the calibration.

## Calibration and accuracy

`handsfree calibrate` shows 14 targets and takes about 30 seconds. The first
target is a warm-up and is not used.

Calibration fits 6 candidate models. The eye signal is the iris position, the
model's eye-look scores ("blendshape"), or both, with linear or quadratic
terms. Each model gets a cross-check error: one target is left out of the
fit, and the error at that target is measured. The lowest error wins. To
force a model, set `calibration.model` in the config.

`handsfree calibrate --refit` fits the latest recording again, for example
after a config change. It needs no camera.

An experimental head phase (`calibration.head_phase = true`) adds 4 models that
use head pose. In tests so far it made the result worse, so it is off.

`handsfree accuracy` reports, for 12 other targets, the offset (accuracy), the
spread (jitter before smoothing) and the bias (a constant shift, which shows
poor head compensation).

Each calibration and accuracy run is recorded in
`~/.local/share/handsfree/sessions/`. A recording holds gaze feature numbers
only, never camera images. `handsfree evaluate` fits every model on the latest
calibration recording and tests it on the accuracy recordings made after it.
So models can be compared without a new recording.

The calibration is in `~/.local/share/handsfree/calibration.json`.

## Recordings and privacy

`handsfree record` is the only command that stores images. It stores crops of
your eyes and a small crop of your face, with the target you looked at, in
`~/.local/share/handsfree/recordings/`. The files stay on your computer. No
command sends them anywhere. To delete them, delete that folder.

All other commands store numbers only (gaze signals, head pose), or nothing.

## Zero-calibration benchmark

The product must work without a per-user calibration. `handsfree benchmark`
measures that. Each estimator gets the recorded frames and the device
geometry only, and predicts one screen point per frame. The device geometry
is the screen size, `screen.camera_above_mm` and `camera.vertical_fov_deg`.

The report gives, for each recording:

- **error**: mean distance to the target, per frame, in degrees and pixels.
- **target**: error of the mean point for each target. Jitter is removed.
- **jitter**: spread of the points for one target.
- **bias**: mean offset. A constant offset in all recordings points to a
  wrong device constant or a wrong model constant.
- **gain**: predicted movement divided by true movement. 1.00 is correct.

| Estimator | Method | Result on the first 3 recordings |
|---|---|---|
| `geometric` | 3D eyeball on the MediaPipe iris landmarks, population averages | 14.7° mean error; gain 0.75 horizontal, 0.39 vertical (camera field of view 44.7° vertical) |

The `geometric` result shows that the MediaPipe iris landmark moves too
little, mostly vertically. An eye-image model is the next estimator.

## Configuration

The config file is `~/.config/handsfree/config.toml`. All keys are optional.
To start from the defaults:

```sh
mkdir -p ~/.config/handsfree
uv run handsfree config --defaults > ~/.config/handsfree/config.toml
```

[`config.example.toml`](config.example.toml) lists every key with its description.
An unknown key or a wrong type stops the program with a clear message.

## Architecture

Each component has one interface (a Python `Protocol`) and can be replaced alone.

```
FrameSource ─► FaceTracker ─┬─► GazeEstimator ─► GazeMapper ─► CursorFilter ─► PointerBackend
 camera.py     face.py      │    gaze.py          mapping.py    (step 3)        (step 3)
                            └─► ClickTrigger(s) ─► Controller (steps 4-5)
```

| Module | Interface | Implementation now |
|---|---|---|
| `camera.py` | `FrameSource` | `OpenCVCamera` (V4L2, MJPG) |
| `face.py` | `FaceTracker` | `MediaPipeFaceTracker` (478 landmarks, head pose, 52 gesture scores) |
| `gaze.py` | `GazeEstimator` | `LandmarkGazeEstimator` (iris position between the eye corners + head pose) |
| `mapping.py` | `GazeMapper` | `UncalibratedMapper` (before calibration) |
| `calibration.py` | `GazeMapper` | `CalibratedMapper` (ridge regression on eye signals, 6 candidate models) |
| `accuracy.py` | | Accuracy and precision report; offline model comparison |
| `sessions.py` | | Recorded target sessions (feature numbers only) |
| `recording.py` | | Labelled recordings with eye and face images, for the benchmark |
| `targets.py` | | Fullscreen target sequence that records gaze features (probe, calibration, accuracy test) |
| `probe.py` | | Signal and noise report for each gaze signal |
| `pipeline.py` | | Joins the components and timestamps each stage |
| `latency.py` | | Rolling median / p95 / max for each stage |
| `config.py` | | TOML config with validation |

## Latency

`handsfree bench` reports the time for each stage, from the moment the camera
driver captured a frame.

Measured on a Framework Laptop 16 (Ryzen 7 7840HS, CPU only), built-in camera,
640x480 MJPG, 59.1 fps:

| Stage | Median | p95 |
|---|---|---|
| Camera driver → app | 17.1 ms | 17.3 ms |
| Face model | 7.9 ms | 9.2 ms |
| Gaze features | 0.1 ms | 0.1 ms |
| **Total, capture → gaze features** | **25.1 ms** | **26.8 ms** |

The total does not include the exposure and readout time inside the camera,
which software cannot see. Cursor output is not part of the pipeline yet.

## Development

```sh
uv run pytest
```

## Licence

MIT. See [LICENSE](LICENSE). The MediaPipe face model is Apache-2.0.
