import numpy as np
import pytest

from src.config import Config
from src.data.augment import add_noise_at_snr
from src.data.features import Featurizer


def _clip(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 16_000, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.normal(size=16_000)).astype(np.float32)


@pytest.mark.parametrize("feature_type", ["mfcc", "logmel"])
def test_output_matches_declared_shape(feature_type):
    config = Config.load(overrides=[f"features.type={feature_type}"])
    out = Featurizer(config.features)(_clip(), 16_000)
    assert out.shape == config.features.shape
    assert out.dtype == np.float32


def test_length_is_fixed_regardless_of_input_duration():
    config = Config.load()
    featurizer = Featurizer(config.features)
    short = featurizer(_clip()[:4000], 16_000)
    long = featurizer(np.tile(_clip(), 3), 16_000)
    assert short.shape == long.shape == config.features.shape


def test_per_file_normalization_makes_gain_invisible():
    """Documents why `gain` is disabled in configs/default.yaml: per-row z-scoring
    removes the constant offset a gain change produces, so the transform costs
    compute and buys no diversity."""
    featurizer = Featurizer(Config.load().features)
    y = _clip()
    assert np.abs(featurizer(y, 16_000) - featurizer(y * 4.0, 16_000)).max() < 1e-4


def test_noise_at_snr_hits_the_requested_ratio():
    y = _clip()
    noisy = add_noise_at_snr(y, 10.0, np.random.default_rng(0))
    noise = noisy - y
    measured = 10 * np.log10(np.mean(y**2) / np.mean(noise**2))
    assert measured == pytest.approx(10.0, abs=0.5)
