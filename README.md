# handsfree

A hands-free mouse for Linux that uses a regular webcam.

- **Gaze moves the cursor.** No eye-tracking hardware.
- **A facial gesture clicks.** The trigger method is replaceable.
- **Fully local.** No video, audio or data leaves the machine. The only network
  use is a one-time download of the face model (about 4 MB).
- **X11 and Wayland.** Cursor control goes through a kernel virtual pointer (`uinput`).

> **Status: pre-alpha, step 1 of the roadmap.** The camera, the face tracker, the
> gaze estimator and the live preview work. The program does not move the
> cursor or click yet.

## Roadmap

| Step | Content | State |
|---|---|---|
| 1 | Camera input, face tracking, gaze features, live preview, latency report | done |
| 2 | Calibration, accuracy test screen with targets | next |
| 3 | Cursor movement (`uinput`), fixation/jump filter, head fine-tune | |
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
uv run handsfree preview
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
| `handsfree preview` | Live camera view with eye overlays, the uncalibrated gaze point, gesture scores and latency. |
| `handsfree bench [--seconds N]` | Measures frame rate and latency with no window. |
| `handsfree fetch-models` | Downloads the face model to `~/.cache/handsfree/models/`. |
| `handsfree config [--defaults]` | Prints the active (or default) config as TOML. |

Preview keys: `Q` or `Esc` quits. `C` re-centres the gaze point (look at the
screen centre first). `M` toggles the mirror view.

The gaze point in the preview is **uncalibrated**. It shows that the signal
follows your eyes. It does not show where you look on the screen. Calibration
is step 2.

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
| `mapping.py` | `GazeMapper` | `UncalibratedMapper` (preview only) |
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
