"""Synthetic training data: set-up, generation and loading.

The images come from Blender (see ``handsfree/blender/``). A person is a
MakeHuman character (CC0 assets, made with the MPFB add-on) with anatomical
eyeballs. Each sample has exact labels: where the person looks, where each
eyeball is, and the camera. ``SynthSet`` loads a generated set and applies the
camera simulation, so the images look like webcam frames.

Blender runs with its own profile in ``~/.cache/handsfree/blender``. The
user's normal Blender set-up does not change.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camsim import CameraSim
from .recording import EYE_CROP
from .sessions import data_dir

ASSET_URL = "https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip"
ASSET_SHA256 = "b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"
SCRIPTS = Path(__file__).parent / "blender"


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "handsfree"


def blender_profile() -> Path:
    return cache_dir() / "blender"


def assets_dir() -> Path:
    return cache_dir() / "assets" / "makehuman" / "system"


def synth_dir() -> Path:
    return data_dir() / "synth"


def _blender_env() -> dict[str, str]:
    return {**os.environ, "BLENDER_USER_RESOURCES": str(blender_profile()), "HANDSFREE_SYNTH_ASSETS": str(assets_dir())}


def _blender() -> str:
    exe = shutil.which("blender")
    if not exe:
        raise RuntimeError("Blender is not installed. Install Blender 4.2 or later, then run this command again.")
    return exe


def setup() -> None:
    """Install the MPFB add-on into the handsfree Blender profile and download the CC0 MakeHuman assets."""
    exe = _blender()
    blender_profile().mkdir(parents=True, exist_ok=True)
    if not (blender_profile() / "extensions" / "blender_org" / "mpfb").exists():
        print("installing the MPFB add-on (GPL-3.0) from extensions.blender.org …")
        for args in (["extension", "sync"], ["extension", "install", "mpfb", "--enable"]):
            subprocess.run([exe, "-b", "--online-mode", "--command", *args], env=_blender_env(), check=True, capture_output=True)
    if not (assets_dir() / "skins").exists():
        print("downloading the MakeHuman system assets (CC0, about 270 MB) …")
        assets_dir().parent.mkdir(parents=True, exist_ok=True)
        archive = assets_dir().parent / "makehuman_system_assets_cc0.zip"
        digest = hashlib.sha256()
        request = urllib.request.Request(ASSET_URL, headers={"User-Agent": "handsfree"})
        with urllib.request.urlopen(request, timeout=60) as r, open(archive, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != ASSET_SHA256:
            archive.unlink()
            raise RuntimeError("the downloaded asset pack has an unexpected checksum; it was deleted")
        with zipfile.ZipFile(archive) as z:
            z.extractall(assets_dir())
    print(f"Blender profile  {blender_profile()}")
    print(f"assets           {assets_dir()}")


def generate(name: str, people: int, poses: int, seed: int, jobs: int, samples: int = 48) -> Path:
    """Run Blender ``jobs`` times in parallel. Each job makes ``people`` persons with ``poses`` images each."""
    if not (assets_dir() / "skins").exists():
        raise RuntimeError("the synthetic-data assets are missing. Run: handsfree synth setup")
    out = synth_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for j in range(jobs):
        cmd = [_blender(), "-b", "-P", str(SCRIPTS / "generate.py"), "--", "--out", str(out)]
        cmd += ["--people", str(people), "--poses", str(poses), "--seed", str(seed + j), "--samples", str(samples)]
        log = open(out / f"blender-{seed + j:04d}.log", "w")
        procs.append((subprocess.Popen(cmd, env=_blender_env(), stdout=log, stderr=subprocess.STDOUT), log))
    failed = 0
    for proc, log in procs:
        failed += proc.wait() != 0
        log.close()
    if failed:
        raise RuntimeError(f"{failed} of {jobs} Blender jobs failed. See the blender-*.log files in {out}")
    return out


@dataclass(frozen=True)
class SynthSet:
    """A generated set: one label dictionary for each rendered image."""

    directory: Path
    labels: list[dict]

    @classmethod
    def load(cls, directory: Path) -> SynthSet:
        labels: list[dict] = []
        for path in sorted(directory.glob("labels*.jsonl")):
            labels += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return cls(directory, labels)

    def __len__(self) -> int:
        return len(self.labels)

    def image(self, i: int) -> np.ndarray:
        """The clean render of sample ``i``: the camera-image rectangle with both eyes, uint8 BGR."""
        import cv2

        img = cv2.imread(str(self.directory / self.labels[i]["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.directory / self.labels[i]["image"])
        return img

    def eye_crops(
        self, i: int, sim: CameraSim | None = None, rng: np.random.Generator | None = None, jitter_px: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two eye crops like a recording's: (2 x height x width x 3 uint8 BGR, 2 x 2 crop origins in camera pixels).

        Index 0 is the eye on the image-left side (the person's right eye). ``sim`` applies
        the camera simulation to the whole rectangle first, as a real camera does.
        ``jitter_px`` moves each crop at random, because a real face tracker does not find
        the same eye point as the labels.
        """
        from .recording import crop

        rng = rng or np.random.default_rng()
        label = self.labels[i]
        img = self.image(i)
        x0, y0 = label["rect"][:2]
        if sim is not None:
            img = sim.apply(img, rng, grid_offset=(x0, y0))
        crops, origins = [], []
        for cx, cy in label["crop_centres_px"]:
            dx, dy = rng.uniform(-jitter_px, jitter_px, 2) if jitter_px else (0.0, 0.0)
            c, (ox, oy) = crop(img, cx - x0 + dx, cy - y0 + dy, *EYE_CROP)
            crops.append(c)
            origins.append((ox + x0, oy + y0))
        return np.stack(crops), np.array(origins)


def contact_sheet(s: SynthSet, count: int, sim: bool, seed: int = 0) -> np.ndarray:
    """An image with the eye crops of the first ``count`` samples, for a visual check."""
    rng = np.random.default_rng(seed)
    tiles = [np.hstack(list(s.eye_crops(i, CameraSim.random(rng) if sim else None, rng)[0])) for i in range(min(count, len(s)))]
    cols = 3
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def run(args) -> int:
    if args.synth_command == "setup":
        setup()
        return 0
    if args.synth_command == "generate":
        print(f"generating {args.jobs * args.people * args.poses} samples with {args.jobs} Blender jobs …")
        out = generate(args.name, args.people, args.poses, args.seed, args.jobs)
        print(f"{len(SynthSet.load(out))} samples in {out}")
        return 0
    if args.synth_command == "sheet":
        import cv2

        directory = synth_dir() / args.name
        s = SynthSet.load(directory)
        if not len(s):
            print(f"no samples in {directory}", file=sys.stderr)
            return 1
        path = directory / ("sheet-camera.png" if args.camera else "sheet.png")
        cv2.imwrite(str(path), contact_sheet(s, args.count, args.camera))
        print(f"wrote {path}")
        return 0
    return 1
