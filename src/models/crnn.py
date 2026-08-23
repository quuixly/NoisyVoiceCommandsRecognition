import torch
from torch import nn


class CRNN(nn.Module):
    """Conv front end + bidirectional GRU over time.

    The convolutions collapse the frequency axis while keeping time resolution, then
    the GRU models the order of phonemes explicitly. Worth comparing against
    BaselineCNN when the CNN underfits — commands like "up"/"stop" differ mainly in
    how their spectra evolve, which a global average pool throws away.
    """

    def __init__(
        self,
        in_channels: int,
        in_rows: int,
        num_classes: int,
        channels: list[int] | None = None,
        rnn_hidden: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 96]
        blocks: list[nn.Module] = []
        prev, rows = in_channels, in_rows
        for width in channels:
            blocks += [
                nn.Conv2d(prev, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),  # halve frequency only — time stays intact for the GRU
            ]
            prev, rows = width, rows // 2
        self.features = nn.Sequential(*blocks)
        self.rnn = nn.GRU(prev * max(1, rows), rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                       # (B, C, rows, T)
        b, c, rows, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * rows)
        x, _ = self.rnn(x)
        return self.head(x.mean(dim=1))
