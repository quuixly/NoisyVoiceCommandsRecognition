import inspect

from torch import nn

from src.config import Config
from src.models.baseline_cnn import BaselineCNN
from src.models.crnn import CRNN
from src.models.logreg import LogisticRegression

MODELS: dict[str, type[nn.Module]] = {
    "baseline_cnn": BaselineCNN,
    "crnn": CRNN,
    "logreg": LogisticRegression,
}


def build_model(config: Config) -> nn.Module:
    """Input shape comes from the feature config and class count from the command
    list, so swapping `features.type` or adding a command needs no model edit.

    Constructor arguments are filtered by signature: `model:` in the YAML is one flat
    section, and a model only receives the keys it actually declares."""
    try:
        model_cls = MODELS[config.model.name]
    except KeyError:
        raise ValueError(
            f"Unknown model.name={config.model.name!r}, available: {sorted(MODELS)}"
        ) from None

    channels, rows, frames = config.features.shape
    args = {
        "in_channels": channels,
        "in_rows": rows,
        "in_frames": frames,
        "num_classes": len(config.data.commands),
        **config.model.kwargs(),
    }
    accepted = inspect.signature(model_cls).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        args = {k: v for k, v in args.items() if k in accepted}
    return model_cls(**args)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
