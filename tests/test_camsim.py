import numpy as np

from handsfree.camsim import CameraSim


def _flat(level: int, size=(128, 192)) -> np.ndarray:
    return np.full((*size, 3), level, np.uint8)


def _noise_sigma(sim: CameraSim, level: int) -> float:
    rng = np.random.default_rng(0)
    out = sim.apply(_flat(level), rng).astype(float)
    luma = out @ [0.114, 0.587, 0.299]
    return float(luma.std())


def test_output_has_the_input_shape_and_type():
    img = np.random.default_rng(1).integers(0, 255, (101, 77, 3), dtype=np.uint8)  # not a multiple of the block size
    out = CameraSim().apply(img, np.random.default_rng(0), grid_offset=(5, 3))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_noise_grows_with_brightness():
    sim = CameraSim(blur_sigma=0, sharpen=0, jpeg_quality=0, chroma_noise=0)
    dark, bright = _noise_sigma(sim, 16), _noise_sigma(sim, 110)
    assert abs(dark - np.sqrt(2.7 + 0.18 * 16)) < 0.4
    assert abs(bright - np.sqrt(2.7 + 0.18 * 110)) < 0.6
    assert bright > dark * 1.5


def test_noise_is_correlated_between_neighbours():
    sim = CameraSim(blur_sigma=0, sharpen=0, jpeg_quality=0, chroma_noise=0)
    out = sim.apply(_flat(80, (256, 256)), np.random.default_rng(0)).astype(float)
    d = out[..., 1] - out[..., 1].mean()
    corr1 = (d[:, 1:] * d[:, :-1]).mean() / (d * d).mean()
    corr4 = (d[:, 4:] * d[:, :-4]).mean() / (d * d).mean()
    assert 0.4 < corr1 < 0.7 and abs(corr4) < 0.1  # measured on the real camera: 0.57 and about 0


def test_blur_widens_an_edge_and_sharpening_adds_overshoot():
    img = _flat(40)
    img[:, 96:] = 160
    quiet = dict(noise_floor=0.0, noise_slope=0.0, jpeg_quality=0)
    rng = np.random.default_rng(0)
    row = lambda sim: sim.apply(img, rng)[64, :, 1].astype(float)  # noqa: E731
    sharp = row(CameraSim(blur_sigma=0, sharpen=0, **quiet))
    soft = row(CameraSim(blur_sigma=1.2, sharpen=0, **quiet))
    halo = row(CameraSim(blur_sigma=1.2, sharpen=1.0, **quiet))
    assert np.abs(np.diff(soft)).max() < np.abs(np.diff(sharp)).max() / 2
    assert soft.max() <= 160 and halo.max() > 161 and halo.min() < 39


def test_jpeg_changes_the_image_and_low_quality_changes_it_more():
    img = np.random.default_rng(2).integers(60, 120, (128, 192, 3), dtype=np.uint8)
    quiet = dict(blur_sigma=0, sharpen=0, noise_floor=0.0, noise_slope=0.0)
    rng = np.random.default_rng(0)
    err = lambda q: np.abs(CameraSim(jpeg_quality=q, **quiet).apply(img, rng).astype(int) - img).mean()  # noqa: E731
    assert err(0) == 0 and 0 < err(90) < err(40)


def test_random_parameters_are_inside_their_ranges_and_differ():
    rng = np.random.default_rng(0)
    sims = [CameraSim.random(rng) for _ in range(50)]
    assert all(0.4 <= s.blur_sigma <= 1.4 and 40 <= s.jpeg_quality <= 90 for s in sims)
    assert len({s.jpeg_quality for s in sims}) > 10
