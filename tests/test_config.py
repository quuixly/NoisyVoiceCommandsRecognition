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


def test_every_shipped_experiment_loads_and_builds_a_model():
    """A config file that no longer merges cleanly — a renamed key, a width the model
    can't take — is only discovered when a six-hour sweep reaches it. Load and
    instantiate every one of them here instead."""
    from pathlib import Path

    from src.models import build_model

    paths = sorted(Path("configs/experiments").glob("*.yaml"))
    assert paths, "no experiment configs found"
    for path in paths:
        config = Config.load(path)
        assert build_model(config) is not None, path


def test_sweep_selection_skips_smoke_and_keeps_one_baseline():
    from src.pipeline import SWEEP_SKIP, sweep_names

    names = sweep_names([])
    assert names.count("baseline") == 1
    assert not SWEEP_SKIP & set(names)
    assert "crnn" in names and "logmel" in names
    # Naming a skipped experiment explicitly still runs it.
    assert sweep_names(["smoke"]) == ["smoke"]
