import pytest
import torch

from src.config import Config
from src.models import MODELS, build_model, count_parameters


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("feature_type", ["mfcc", "logmel"])
def test_every_model_accepts_every_feature_front_end(name, feature_type):
    config = Config.load(overrides=[f"model.name={name}", f"features.type={feature_type}"])
    model = build_model(config)
    channels, rows, frames = config.features.shape
    out = model(torch.randn(4, channels, rows, frames))
    assert out.shape == (4, len(config.data.commands))


def test_architecture_comes_from_config_not_the_source():
    small = build_model(Config.load(overrides=["model.channels=[8, 16]"]))
    big = build_model(Config.load(overrides=["model.channels=[64, 128, 256]"]))
    assert count_parameters(small) < count_parameters(big)


def test_baseline_stays_under_the_on_device_budget():
    assert count_parameters(build_model(Config.load())) < 200_000


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown model\.name"):
        build_model(Config.load(overrides=["model.name=resnet152"]))
