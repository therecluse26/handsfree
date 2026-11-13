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
| `handsfree synth setup` | Installs the Blender add-on and downloads the CC0 character assets (about 270 MB). Uses a separate Blender profile. |
| `handsfree synth generate` | Renders labelled synthetic eye images with Blender. `--people`, `--poses`, `--jobs`, `--seed`, `--name`. |
| `handsfree synth sheet [--camera]` | Writes one image with many eye crops of a synthetic set, for a visual check. `--camera` applies the camera simulation. |
| `handsfree train --set NAME` | Trains the two eye-image gaze models (one for yaw, one for pitch) on a synthetic set and writes them as ONNX. Needs `uv sync --group train` (PyTorch). |
| `handsfree reference fetch` | Downloads 3000 portrait photos of real persons (FFHQ, about 3 GB) for measurements. They are never used for training. |
| `handsfree reference gaze` | Measures the eye-image model on those photos: the persons look into the camera, so the correct gaze is known. |
| `handsfree reference realism --set NAME` | Trains a classifier to tell real eye images from synthetic ones, for each part of the image. Needs `uv sync --group train`. |
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

| Estimator | Method | Result on 4 recordings of one person |
|---|---|---|
| `geometric` | 3D eyeball on the MediaPipe iris landmarks, population averages | 14.1° mean error; gain 0.68 horizontal, 0.39 vertical (camera field of view 44.7° vertical) |
| `eye-cnn` | Two CNNs on normalised eye images, trained on synthetic renders only | 5.5° mean error; gain 0.94 horizontal, 0.62 vertical |

The `geometric` result shows that the MediaPipe iris landmark moves too
little, mostly vertically.

History of the `eye-cnn` result. Each model is trained on synthetic renders
only. Recordings 1 to 3 were used to compare the models. Recording 4 was made
after the comparison, in low light, so it is an independent test:

| Training set | Part of the eye image that the model reads | Recordings 1 to 3 | Recording 4 | Gain horizontal, vertical (1 to 3) |
|---|---|---|---|---|
| 5543 renders, no glasses, bright eyes | whole image (48 x 32 mm) | 8.0° | 8.6° | 0.70, 0.42 |
| the same | window, 32 x 20 mm | 8.8° | 10.5° | 0.90, 0.37 |
| 4800 renders with glasses and lid shade | whole image | 6.9° | 7.7° | 0.61, 0.65 |
| the same | window, 32 x 20 mm | 6.7° | 7.8° | 1.01, 0.37 |
| the same | window, 32 x 32 mm | 6.0° | 7.8° | 0.78, 0.36 |
| the same | yaw from the window model, pitch from the whole-image model (the current `eye-cnn`) | 5.2° | 6.6° | 1.00, 0.67 |

The window model follows horizontal movement correctly. The whole-image
model follows vertical movement better. So `eye-cnn` takes the yaw from the
first and the pitch from the second.

One measurement shows that the eye images hold more information: a model
tuned on two recordings of one person has 1.9° to 3.8° on that person's
third recording. So the open problem is to make the model correct for a new
person, not the camera or the image size.

## Measurements on many real persons

Three recordings of one person cannot show that a model works for everyone.
`handsfree reference` uses 3000 portrait photos (FFHQ) as a second test. In
a portrait the person looks into the camera. So the correct gaze is known:
(0, 0) in the normalised eye view. Each photo gets the camera simulation
first, so it looks like a webcam frame.

```sh
uv run handsfree reference fetch
uv run handsfree reference gaze
```

Results for the current `eye-cnn`, frontal heads:

- **Constant error for the average person**: yaw -0.3°, pitch -0.5°.
- **Difference from person to person**: 7.6° in yaw, 5.4° in pitch (half of
  the middle 68%). This includes the persons who do not look into the camera.
  The whole-image model has 6.6° in yaw here, so the window model is better
  at movement and worse at the constant error.
- **Each eye alone** is read about 7° towards the nose (3° with the
  whole-image model). The two eyes cancel this in the mean.
- **Head turn**: the yaw difference grows to 14.2° at 10° to 25° of head turn.
- **Glasses**: persons with glasses are read 4.1° lower than persons without.

These measurements found one cause of error in the synthetic images: the
eye opening was too bright against the cheek (0.60 of the cheek brightness;
0.36 to 0.45 in real images). The lid shade (see below) corrects it to 0.45.

The photos are licensed CC BY-NC-SA 4.0 by NVIDIA, and each photo has its
own licence. handsfree uses them to measure only. No model is trained on
them, and they are not in this repository. The glasses label of each photo
comes from the FFHQ features data set (DCGM/ffhq-features-dataset).

## Synthetic training data

The gaze model must work for everyone with no calibration. So its training
data is synthetic: Blender renders persons with exact gaze labels, and a
camera simulation makes the renders look like webcam frames. Real recordings
are the test set only.

Set-up needs Blender 4.2 or later:

```sh
uv run handsfree synth setup
uv run handsfree synth generate --name trial --people 3 --poses 4 --jobs 2
uv run handsfree synth sheet --name trial --camera
```

- **Persons** are MakeHuman characters, made with the MPFB add-on. Gender,
  age, body, skin, eyebrows and eyelashes are random.
- **Eyes** are anatomical models (`handsfree/blender/eye.py`): a 12 mm
  eyeball, a refracting cornea, a procedural iris, and the angle between the
  optical axis and the line of sight. Size, colour and pupil are random.
- **Lid shade**: the lids, lashes and brow shade a real eye. The eye
  material darkens where the lids hide the surroundings, and the room light
  is bright above and dark below.
- **Glasses** (`handsfree/blender/glasses.py`): 45% of the persons wear
  glasses. The lenses are real lens shapes with a random power, so they bend
  the light, change the size of the eye and reflect the screen and the lamps.
  Frame shape, colour and position are random.
- **Each image** has a random camera field of view (36° to 70°), distance
  (350 to 900 mm), screen size, head pose, gaze target, lid position and
  light. Head size, eye shape and the eyeball position in its socket are
  random for each person.
- **Labels** are exact, in camera space (mm): the target, each eyeball
  centre, the line of sight, the pupil position and the head pose. A check
  that renders only the iris finds it within about 0.5 px of its label.
- **Camera simulation** (`handsfree/camsim.py`): lens blur, noise that grows
  with brightness and is correlated between neighbours, sharpening and 4:2:2
  JPEG. The ranges come from measurements on real recordings.

The renders hold linear light values. Exposure, white balance and the tone
curve are applied when a set is loaded, like a camera's auto exposure. So one
render gives many different camera images.

Sets are stored in `~/.local/share/handsfree/synth/`. Speed is about 1.3
images per second on a 16-thread CPU.

Licences: the MPFB add-on is GPL-3.0. handsfree runs it inside Blender as a
separate program and does not include it. The MakeHuman system assets are
CC0. The generated images contain no real person.

## The eye-image gaze model

`handsfree train` trains a small CNN on a synthetic set. Real recordings are
never used to fit it. `handsfree benchmark` tests it on them as `eye-cnn`.

1. **Normalisation** (`eyenorm.py`). Each eye is warped to the view of a
   standard virtual camera: it looks straight at the eye, has no roll, and
   shows the eye at 2 pixels per millimetre (96 x 64 pixels). The warp uses the
   camera's focal length and the eye's distance. So the model sees the same
   kind of image for every camera, distance and position in the frame.
2. **Model** (`gazetrain.py`). In training, each image moves and changes
   size at random. Without this the model reads the iris position in the
   image frame, and each millimetre of face tracker offset gives about 3° of
   error. The input is one grey eye image with zero mean
   and unit variance. A model can read the whole image or only a window
   around the eye (`eyenorm.WINDOW`); its `.json` file records the window.
   `handsfree train` trains two models: one with the window, which gives the
   yaw, and one with the whole image, which gives the pitch. The output is the gaze direction as yaw and pitch in the
   virtual camera. Right eyes are mirrored, so one model serves both eyes.
3. **Use** (`benchmark.py`, `EyeImageModel`). The ONNX file runs in OpenCV.
   Each eye's direction is turned back into camera space and meets the screen
   plane. The result is the mean of the two eyes.

```sh
uv sync --group train
uv run handsfree synth generate --name train1 --people 200 --poses 6 --jobs 4
uv run handsfree train --set train1
uv run handsfree benchmark
```

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
