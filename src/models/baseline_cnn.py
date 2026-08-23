import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Conv stack + global average pool + linear head.

    Input is (B, C, rows, frames) exactly as the Featurizer produces it — for MFCC
    with deltas that is 3 channels of n_mfcc rows, which keeps the first convolution
    from blending three unrelated feature types across a row boundary; for log-mel
    it is a single channel. Global pooling rather than a flatten keeps the head at
    `channels[-1] * num_classes` weights instead of letting a `128*rows*frames`
    linear layer outweigh the entire conv stack.
    """

    def __init__(
        self,
        in_channels: int,
        in_rows: int,
        num_classes: int,
        channels: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 96, 128]
        blocks: list[nn.Module] = []
        prev = in_channels
        for width in channels:
            blocks.append(_conv_block(prev, width))
            prev = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x)
