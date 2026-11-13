"""Train the eye-image gaze model on synthetic data. Needs PyTorch (`uv sync --group train`).

The model gets one normalised eye image (eyenorm) and gives the gaze angles
(yaw, pitch) in the eye's virtual camera. It trains on synthetic images only.
Real recordings are the test: they are never used to fit the model.
A model can also get the head direction (gazedata) as a second input.

The trained model is exported as ONNX, so the app runs it with OpenCV and does
not need PyTorch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import eyenorm
from .gazedata import HEAD_ERROR_DEG, Examples, synth_examples
from .sessions import data_dir
from .synth import SynthSet, synth_dir

ANGLE_SCALE = 10.0
"""The model's outputs are degrees / ANGLE_SCALE."""
HEAD_SCALE = 30.0
"""The head direction goes into the model as degrees / HEAD_SCALE."""


PITCH_MODEL = "eye-gaze-pitch.onnx"
"""The second model of the pair, in the same folder as the first. See ``run_train``."""


def model_path() -> Path:
    return data_dir() / "models" / "eye-gaze.onnx"


def build_examples(name: str, copies: int = 3, seed: int = 0, log=print) -> dict[str, np.ndarray]:
    """Normalised training examples of a synthetic set, cached in the set's folder.

    ``name`` can list several sets, separated by commas.
    """
    if "," in name:
        parts = [build_examples(n.strip(), copies, seed, log) for n in name.split(",")]
        return {k: np.concatenate([p[k] for p in parts]) if k != "samples" else sum(p[k] for p in parts) for k in parts[0]}
    directory = synth_dir() / name
    cache = directory / f"examples-{copies}-{seed}.npz"
    s = SynthSet.load(directory)
    if cache.exists():
        with np.load(cache) as d:
            if int(d["samples"]) == len(s) and "head" in d.files:
                return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    parts: list[Examples] = []
    t0 = time.time()
    for start in range(0, len(s), 200):
        parts.append(synth_examples(s, range(start, min(start + 200, len(s))), rng, augment=True, copies=copies))
        log(f"  examples: {min(start + 200, len(s))}/{len(s)} renders ({time.time() - t0:.0f} s)")
    out = {
        "images": np.concatenate([p.images for p in parts]),
        "angles": np.concatenate([p.angles for p in parts]),
        "group": np.concatenate([p.group for p in parts]),
        "head": np.concatenate([p.head for p in parts]),
        "samples": np.array(len(s)),
    }
    np.savez_compressed(cache, **out)
    return out


WIDTHS = (24, 48, 96, 160)
"""Channels of the four stages of the model."""


def make_model(widths: tuple[int, ...] = WIDTHS, head: bool = False):
    """A small CNN: 64 x 96 grey image in, (yaw, pitch) / ANGLE_SCALE out.

    The first layer halves the resolution. Layers at the full resolution made training 6 times slower.
    With ``head`` the model has a second input, the head direction / HEAD_SCALE. It joins the
    image features before the last layers.
    """
    import torch
    import torch.nn as nn

    def conv(cin: int, cout: int, stride: int = 1) -> list:
        return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

    layers, cin = [], 1
    for w in widths:  # 64 x 96 -> 32 x 48 -> 16 x 24 -> 8 x 12 -> 4 x 6
        layers += conv(cin, w, stride=2) + conv(w, w)
        cin = w
    body, features = [*layers, nn.Flatten(), nn.Dropout(0.2)], cin * 4 * 6
    if not head:
        return nn.Sequential(*body, nn.Linear(features, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))

    class WithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*body)
            self.head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(inplace=True))
            self.top = nn.Sequential(nn.Linear(features + 16, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))

        def forward(self, eyes, head):
            return self.top(torch.cat([self.body(eyes), self.head(head)], 1))

    return WithHead()


def jitter(x, max_shift_px: float = 9.0, max_scale: float = 0.1):
    """Move and resize each image of a batch at random (N x 1 x H x W tensor).

    The face tracker's eye point is not the same point on every face, and its distance has
    an error. A model that reads the iris position in the image frame gets a constant gaze
    error from that: 1 mm of offset is about 3 degrees. With this jitter the model must read
    the iris against the lids and eye corners, which is correct for every eye point.
    """
    import torch
    import torch.nn.functional as F

    n, _, h, w = x.shape
    scale = 1.0 + (torch.rand(n) * 2 - 1) * max_scale
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = scale
    theta[:, 0, 2] = (torch.rand(n) * 2 - 1) * max_shift_px * 2 / w  # the grid runs from -1 to 1
    theta[:, 1, 2] = (torch.rand(n) * 2 - 1) * max_shift_px * 2 / h
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def balance_weights(angles: np.ndarray, head: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Example weights that make the gaze and the head direction independent in a training set. Their mean is 1.

    A synthetic person turns the head part of the way to the target, so the head direction
    predicts the gaze (correlation 0.6). A model that gets the head direction learns this, and
    on a real eye it then follows the head: measured on photos and on recordings of a user
    whose head does not move. With these weights the correlation is below 0.05, and the head
    direction only tells the model how to read the eye image. ``clip`` limits a weight to
    ``clip`` times the median; with 10, the set counts as 43% of its examples.
    """
    w = np.ones(len(angles))
    for j in (0, 1):
        a = (angles[:, j] - angles[:, j].mean()) / angles[:, j].std()
        b = (head[:, j] - head[:, j].mean()) / head[:, j].std()
        r = float(np.corrcoef(a, b)[0, 1])
        w *= np.exp((r * r * (a * a + b * b) - 2 * r * a * b) / (2 * (1 - r * r)))  # normal(a) normal(b) / normal(a, b)
    w = np.minimum(w / np.median(w), clip)
    return (w / w.mean()).astype(np.float32)


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between gaze directions given as (yaw, pitch) degrees."""

    def unit(yp):
        y, p = np.radians(yp[:, 0]), np.radians(yp[:, 1])
        return np.stack([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)], 1)

    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(1), -1.0, 1.0)))


def predict(model, images: np.ndarray, batch: int = 512, window=eyenorm.WINDOW, head: np.ndarray | None = None) -> np.ndarray:
    """Gaze angles (N x 2, degrees) for normalised grey images (N x 64 x 96 uint8).

    ``head`` (N x 2, degrees) is for a model with the head direction as its second input.
    """
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = [torch.from_numpy(eyenorm.to_model_input(images[i : i + batch], window=window))]
            if head is not None:
                x.append(torch.from_numpy(np.asarray(head[i : i + batch], np.float32) / HEAD_SCALE))
            out.append(model(*x).numpy() * ANGLE_SCALE)
    return np.concatenate(out)


def train(name: str, epochs: int = 30, copies: int = 3, seed: int = 0, batch: int = 128, lr: float = 2e-3, log=None, window=eyenorm.WINDOW,
          path: Path | None = None, pitch_from: str = "", widths: tuple[int, ...] = WIDTHS, head: bool = False, balance: bool = False) -> dict:  # fmt: skip
    """Train on the synthetic set(s) ``name`` and write the model to ``path``.

    ``window`` is the part of the eye image that the model reads; None is all of it.
    ``pitch_from`` names a second model file whose pitch replaces this model's pitch in use.
    With ``head`` the model also gets the head direction. The training adds the face tracker's
    error (HEAD_ERROR_DEG) to the exact direction of the synthetic person.
    With ``balance`` the examples get ``balance_weights``.
    """
    import torch
    import torch.nn.functional as F

    log = log or (lambda *a: print(*a, flush=True))

    torch.manual_seed(seed)
    data = build_examples(name, copies, seed, log)
    groups = np.unique(data["group"])
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(groups, max(1, len(groups) // 10), replace=False).tolist())
    val = np.array([g in held_out for g in data["group"]])
    x_train = torch.from_numpy(eyenorm.to_model_input(data["images"][~val], window=window))
    y_train = torch.from_numpy(data["angles"][~val] / ANGLE_SCALE)
    if head and not np.isfinite(data["head"]).all():
        raise RuntimeError(f"a set of {name} has no head direction in its labels: generate it again or train without the head direction")
    head_error = torch.tensor(HEAD_ERROR_DEG) / HEAD_SCALE
    h_train = torch.from_numpy(data["head"][~val] / HEAD_SCALE)
    w_train = torch.from_numpy(balance_weights(data["angles"][~val], data["head"][~val])) if balance else torch.ones(len(x_train))
    h_val = data["head"][val] + np.random.default_rng(seed).normal(0.0, HEAD_ERROR_DEG, (int(val.sum()), 2)) if head else None
    log(f"training examples {len(x_train)}, validation examples {int(val.sum())} ({len(held_out)} held-out persons)")

    model = make_model(widths, head)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(x_train) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i : i + batch]
            x, y = x_train[idx], y_train[idx]
            x = [jitter(x) + 0.05 * torch.randn_like(x)]
            if head:
                x.append(h_train[idx] + torch.randn(len(idx), 2) * head_error)
            loss = (F.smooth_l1_loss(model(*x), y, reduction="none").mean(dim=1) * w_train[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        pred = predict(model, data["images"][val], window=window, head=h_val)
        err = angular_error_deg(pred, data["angles"][val])
        # Gain: how much of a true gaze movement the model follows. 1.0 is correct.
        gain = [float(np.polyfit(data["angles"][val][:, j], pred[:, j], 1)[0]) for j in (0, 1)]
        history.append({"epoch": epoch + 1, "loss": total / len(order), "val_deg": float(err.mean()), "val_gain": gain})
        log(
            f"epoch {epoch + 1:3d}/{epochs}  loss {history[-1]['loss']:.4f}  synthetic validation {err.mean():.2f} deg"
            f"  gain yaw {gain[0]:.2f} pitch {gain[1]:.2f}  ({time.time() - t0:.0f} s)"
        )
    path = export(model, path)
    info = {"set": name, "epochs": epochs, "copies": copies, "window": list(window) if window else None, "pitch_from": pitch_from, "widths": list(widths), "head": head, "balance": balance, "examples": int(len(x_train)), "val_deg": history[-1]["val_deg"], "history": history}
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))
    torch.save(model.state_dict(), path.with_suffix(".pt"))
    return {"model": model, "path": path, **info}


def export(model, path: Path | None = None) -> Path:
    """Write the model as ONNX for a batch of 2 images: the two eyes of one frame."""
    import torch

    path = path or model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = [torch.zeros(2, 1, eyenorm.SIZE[1], eyenorm.SIZE[0])]
    if hasattr(model, "head"):
        dummy.append(torch.zeros(2, 2))
    names = ["eyes", "head"][: len(dummy)]
    torch.onnx.export(model, tuple(dummy), str(path), input_names=names, output_names=["angles"], opset_version=17, dynamo=False)
    return path


def run_train(args) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is missing. Install the training packages with: uv sync --group train")
        return 1
    # Two kinds of model. Measured on real recordings: a model that reads only the window around
    # the eye follows horizontal movement correctly (gain about 1.0; 0.6 with the whole image). A
    # model that reads the whole eye image follows vertical movement better.
    #
    # Several models of each kind (--seeds). Two trainings on the same set differ by 2 to 5 degrees
    # in their constant error on one real person. The mean of several models removes a part of that.
    names: dict[str, list[str]] = {"yaw": [], "pitch": []}
    for seed in range(args.seeds):
        suffix = "" if seed == 0 else f"-s{seed}"
        for kind, window, base in (("yaw", eyenorm.WINDOW, model_path()), ("pitch", None, model_path().with_name(PITCH_MODEL))):
            path = base.with_name(base.stem + suffix + ".onnx")
            result = train(args.set, epochs=args.epochs, copies=args.copies, seed=seed, window=window, path=path, pitch_from=PITCH_MODEL if kind == "yaw" else "", head=args.head, balance=args.head)
            names[kind].append(path.name)
            print(f"{kind} model written: {result['path']}  (synthetic validation {result['val_deg']:.2f} deg)")
    info_path = model_path().with_suffix(".json")
    info = json.loads(info_path.read_text())
    info["ensemble"] = names
    info_path.write_text(json.dumps(info, indent=1))
    print("test them on the real recordings with: handsfree benchmark")
    return 0
