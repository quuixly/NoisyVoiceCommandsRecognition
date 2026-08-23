import pytest

from src.config import Config


def test_experiment_file_deep_merges_onto_default():
    config = Config.load("configs/experiments/logmel.yaml")
    assert config.features.type == "logmel"
    assert config.features.n_fft == 512  # untouched key survives the merge


def test_set_override_is_typed_not_string():
    config = Config.load(overrides=["train.lr=1e-3", "train.epochs=3", "augment.enabled=false"])
    assert config.train.lr == pytest.approx(1e-3)
    assert config.train.epochs == 3
    assert config.augment.enabled is False


def test_unknown_key_fails_loudly():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.load(overrides=["train.epocs=3"])


def test_feature_shape_tracks_type():
    assert Config.load(overrides=["features.type=mfcc", "features.n_mfcc=20"]).features.shape == (3, 20, 101)
    assert Config.load(overrides=["features.type=logmel", "features.n_mels=64"]).features.shape == (1, 64, 101)


def test_fingerprint_changes_with_any_feature_knob():
    a = Config.load().features.fingerprint()
    b = Config.load(overrides=["features.n_mels=80"]).features.fingerprint()
    assert a != b
