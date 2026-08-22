import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN

_MODELS: dict[str, type[nn.Module]] = {
    "baseline_cnn": BaselineCNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    try:
        model_cls = _MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model_name={name!r}, available: {sorted(_MODELS)}") from None
    return model_cls(**kwargs)
