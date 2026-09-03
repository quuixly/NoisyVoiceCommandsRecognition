import torch
from torch import nn


class LogisticRegression(nn.Module):
    """Flatten + one linear layer. The floor every other model has to clear: if the
    CNN isn't well ahead of this, the gain is coming from the features, not the
    architecture."""

    def __init__(self, in_channels: int, in_rows: int, num_classes: int, in_frames: int = 101, **_: object) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_channels * in_rows * in_frames, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.flatten(1))
