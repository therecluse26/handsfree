"""Model files. They are downloaded once, then used offline."""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class ModelError(RuntimeError):
    pass


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "handsfree" / "models"


def face_landmarker_path(configured: str = "") -> Path:
    return Path(configured).expanduser() if configured else cache_dir() / "face_landmarker.task"


def require_face_landmarker(configured: str = "") -> Path:
    p = face_landmarker_path(configured)
    if not p.exists():
        raise ModelError(f"face model not found at {p}. Run: handsfree fetch-models")
    return p


def fetch_face_landmarker(force: bool = False) -> Path:
    """Download the face model to the cache and verify its SHA-256."""
    dest = cache_dir() / "face_landmarker.task"
    if dest.exists() and not force and _sha256(dest.read_bytes()) == FACE_LANDMARKER_SHA256:
        return dest
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats()["x509_ca"]:
        # Standalone Python builds do not always find the system CA bundle.
        for ca in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
            if os.path.exists(ca):
                ctx.load_verify_locations(cafile=ca)
                break
    with urllib.request.urlopen(FACE_LANDMARKER_URL, context=ctx, timeout=60) as r:
        data = r.read()
    digest = _sha256(data)
    if digest != FACE_LANDMARKER_SHA256:
        raise ModelError(f"face model checksum mismatch: got {digest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
