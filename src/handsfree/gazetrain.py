"""Train the eye-image gaze model on synthetic data. Needs PyTorch (`uv sync --group train`).

The model gets one normalised eye image (eyenorm) and gives the gaze angles
(yaw, pitch) in the eye's virtual camera. It trains on synthetic images only.
Real recordings are the test: they are never used to fit the model.

The trained model is exported as ONNX, so the app runs it with OpenCV and does
not need PyTorch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import eyenorm
from .gazedata import Examples, synth_examples
from .sessions import data_dir
from .synth import SynthSet, synth_dir

ANGLE_SCALE = 10.0
"""The model's outputs are degrees / ANGLE_SCALE."""


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
            if int(d["samples"]) == len(s):
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
        "samples": np.array(len(s)),
    }
    np.savez_compressed(cache, **out)
    return out


def make_model():
    import torch.nn as nn

    def block(cin: int, cout: int) -> list:
        return [
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]  # fmt: skip

    return nn.Sequential(
        nn.Conv2d(1, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
        *block(24, 32), *block(32, 64), *block(64, 96), *block(96, 128),  # 64 x 96 -> 4 x 6
        nn.Flatten(), nn.Dropout(0.2), nn.Linear(128 * 4 * 6, 128), nn.ReLU(inplace=True), nn.Linear(128, 2),
    )  # fmt: skip


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between gaze directions given as (yaw, pitch) degrees."""

    def unit(yp):
        y, p = np.radians(yp[:, 0]), np.radians(yp[:, 1])
        return np.stack([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)], 1)

    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(1), -1.0, 1.0)))


def predict(model, images: np.ndarray, batch: int = 512) -> np.ndarray:
    """Gaze angles (N x 2, degrees) for normalised grey images (N x 64 x 96 uint8)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(eyenorm.to_model_input(images[i : i + batch]))
            out.append(model(x).numpy() * ANGLE_SCALE)
    return np.concatenate(out)


def train(name: str, epochs: int = 30, copies: int = 3, seed: int = 0, batch: int = 128, lr: float = 2e-3, log=print) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    data = build_examples(name, copies, seed, log)
    groups = np.unique(data["group"])
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(groups, max(1, len(groups) // 10), replace=False).tolist())
    val = np.array([g in held_out for g in data["group"]])
    x_train = torch.from_numpy(eyenorm.to_model_input(data["images"][~val]))
    y_train = torch.from_numpy(data["angles"][~val] / ANGLE_SCALE)
    log(f"training examples {len(x_train)}, validation examples {int(val.sum())} ({len(held_out)} held-out persons)")

    model = make_model()
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
            # The eye point from the face tracker moves a little from frame to frame: shift the image.
            dx, dy = (int(v) for v in torch.randint(-3, 4, (2,)))
            x = torch.roll(x, (dy, dx), dims=(2, 3))
            x = x + 0.05 * torch.randn_like(x)
            loss = F.smooth_l1_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        pred = predict(model, data["images"][val])
        err = angular_error_deg(pred, data["angles"][val])
        # Gain: how much of a true gaze movement the model follows. 1.0 is correct.
        gain = [float(np.polyfit(data["angles"][val][:, j], pred[:, j], 1)[0]) for j in (0, 1)]
        history.append({"epoch": epoch + 1, "loss": total / len(order), "val_deg": float(err.mean()), "val_gain": gain})
        log(
            f"epoch {epoch + 1:3d}/{epochs}  loss {history[-1]['loss']:.4f}  synthetic validation {err.mean():.2f} deg"
            f"  gain yaw {gain[0]:.2f} pitch {gain[1]:.2f}  ({time.time() - t0:.0f} s)"
        )
    path = export(model)
    info = {"set": name, "epochs": epochs, "copies": copies, "examples": int(len(x_train)), "val_deg": history[-1]["val_deg"], "history": history}
    path.with_suffix(".json").write_text(json.dumps(info, indent=1))
    torch.save(model.state_dict(), path.with_suffix(".pt"))
    return {"model": model, "path": path, **info}


def export(model, path: Path | None = None) -> Path:
    """Write the model as ONNX for a batch of 2 images: the two eyes of one frame."""
    import torch

    path = path or model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(2, 1, eyenorm.SIZE[1], eyenorm.SIZE[0])
    torch.onnx.export(model, dummy, str(path), input_names=["eyes"], output_names=["angles"], opset_version=17, dynamo=False)
    return path


def run_train(args) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is missing. Install the training packages with: uv sync --group train")
        return 1
    result = train(args.set, epochs=args.epochs, copies=args.copies)
    print(f"model written: {result['path']}  (synthetic validation {result['val_deg']:.2f} deg)")
    print("test it on the real recordings with: handsfree benchmark")
    return 0
