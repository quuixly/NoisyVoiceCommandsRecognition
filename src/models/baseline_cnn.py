import torch
import torch.nn as nn

from src.constants import N_MFCC
from src.labels import LABEL_TO_INDEX


def _conv_block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    layers.append(nn.MaxPool2d(2) if pool else nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)


class BaselineCNN(nn.Module):
    """~186K params. 4 conv blocks + global-pool head (no flatten — a
    128*T -> 6 linear would alone outweigh the whole conv stack).

    Expects (B, 3*n_mfcc, T) as produced by SpeechCommandsDataset — the 3
    stacked feature planes (MFCC / delta / delta2, see
    src.feature_extractor.extract_mfcc) are split into channels here via a
    plain view(), not on disk, so no feature regeneration is needed. Splitting
    into channels — rather than treating the input as one (3*n_mfcc, T) image —
    keeps the first conv from blending three unrelated feature types across
    their row boundary.
    """

    def __init__(self, num_classes: int = len(LABEL_TO_INDEX), n_mfcc: int = N_MFCC) -> None:
        super().__init__()
        self.n_mfcc = n_mfcc
        self.features = nn.Sequential(
            _conv_block(3, 32, pool=True),
            _conv_block(32, 64, pool=True),
            _conv_block(64, 96, pool=True),
            _conv_block(96, 128, pool=False),  # AdaptiveAvgPool2d(1) here
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, rows, t = x.shape
        x = x.view(b, 3, self.n_mfcc, t)  # (B, 3*n_mfcc, T) -> (B, 3, n_mfcc, T)
        x = self.features(x)
        x = x.flatten(1)  # (B, 128, 1, 1) -> (B, 128)
        return self.head(x)
