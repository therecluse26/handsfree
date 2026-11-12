"""Camera simulation for synthetic images.

A render is clean: no lens blur, no sensor noise, no sharpening, no JPEG
blocks. A gaze model that learns from clean renders fails on real camera
images. This module makes a render look like a webcam frame. It applies, in
the order of a real camera:

    lens blur -> sensor noise -> sharpening -> JPEG (4:2:2)

The value ranges in ``CameraSim.random`` come from measurements on real
recordings (``handsfree record``) of a Framework Laptop 16 webcam (OV2740,
MJPG 1920x1080) in low light:

- Noise: sigma 1.9 to 2.8 levels in dark areas (0-31), 4.3 to 5.2 at 96-127.
  The noise is correlated: 0.57 at 1 px, 0.05 to 0.20 at 2 px.
- Edges: 10-90% rise of 2.0 to 2.75 px (median), overshoot about 0.2 of the
  edge contrast (the camera sharpens).
- JPEG: 4:2:2, quality about 61, near-constant frame size (rate control).

The ranges are wider than the measurements, so a model also learns cameras
that are better or worse than this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSim:
    blur_sigma: float = 0.8
    """Lens and demosaic blur: Gaussian sigma in pixels."""
    noise_floor: float = 2.7
    """Noise variance at black, in squared 8-bit levels (read noise)."""
    noise_slope: float = 0.18
    """Noise variance added per level of brightness (shot noise)."""
    noise_blur_sigma: float = 0.67
    """Blur of the noise field in pixels. It gives the noise its correlation between neighbours."""
    chroma_noise: float = 0.6
    """Colour noise relative to the brightness noise."""
    sharpen: float = 0.6
    """Amount of unsharp masking (the camera's sharpening). 0 is off."""
    sharpen_sigma: float = 1.0
    jpeg_quality: int = 61
    """JPEG quality on the libjpeg scale. 0 means: no JPEG step."""

    @classmethod
    def random(cls, rng: np.random.Generator) -> CameraSim:
        return cls(
            blur_sigma=float(rng.uniform(0.4, 1.4)),
            noise_floor=float(rng.uniform(0.5, 8.0)),
            noise_slope=float(rng.uniform(0.03, 0.35)),
            noise_blur_sigma=float(rng.uniform(0.4, 0.9)),
            chroma_noise=float(rng.uniform(0.2, 1.0)),
            sharpen=float(rng.uniform(0.0, 1.2)),
            sharpen_sigma=float(rng.uniform(0.7, 1.5)),
            jpeg_quality=int(rng.integers(40, 91)),
        )

    def apply(self, image: np.ndarray, rng: np.random.Generator, grid_offset: tuple[int, int] = (0, 0)) -> np.ndarray:
        """Simulate the camera on ``image`` (H x W x 3, uint8, BGR). Returns the same shape and type.

        ``grid_offset`` is the position of the image's top-left pixel in the full
        camera frame, modulo the JPEG block grid. It puts the JPEG blocks at the
        correct places when the image is a crop of a frame.
        """
        import cv2

        img = image.astype(np.float32)
        if self.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)
        img = self._add_noise(img, rng, cv2)
        if self.sharpen > 0:
            img = img + self.sharpen * (img - cv2.GaussianBlur(img, (0, 0), self.sharpen_sigma))
        out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if self.jpeg_quality > 0:
            out = _jpeg_422(out, self.jpeg_quality, grid_offset, cv2)
        return out

    def _add_noise(self, img: np.ndarray, rng: np.random.Generator, cv2) -> np.ndarray:
        h, w = img.shape[:2]
        luma = img @ np.array([0.114, 0.587, 0.299], np.float32)
        sigma = np.sqrt(self.noise_floor + self.noise_slope * np.clip(luma, 0, 255))

        def field() -> np.ndarray:
            n = rng.standard_normal((h, w)).astype(np.float32)
            if self.noise_blur_sigma > 0:
                n = cv2.GaussianBlur(n, (0, 0), self.noise_blur_sigma)
                n /= max(float(n.std()), 1e-6)  # the blur lowers the amplitude; restore unit variance
            return n

        brightness = field() * sigma
        colour = np.stack([field(), np.zeros((h, w), np.float32), field()], axis=-1)  # blue and red against green
        return img + brightness[..., None] + colour * (sigma * self.chroma_noise)[..., None]


def _jpeg_422(img: np.ndarray, quality: int, grid_offset: tuple[int, int], cv2) -> np.ndarray:
    """JPEG-encode and decode with 4:2:2 chroma. A 4:2:2 block unit is 16 x 8 pixels."""
    left, top = grid_offset[0] % 16, grid_offset[1] % 8
    h, w = img.shape[:2]
    right, bottom = (-(left + w)) % 16, (-(top + h)) % 8
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_422"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422]
    ok, data = cv2.imencode(".jpg", padded, params)
    if not ok:  # pragma: no cover
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(data, cv2.IMREAD_COLOR)[top : top + h, left : left + w]
