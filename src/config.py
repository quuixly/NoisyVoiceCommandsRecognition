from dataclasses import dataclass, field
from datetime import datetime

from src.constants import SPLIT_SEED


@dataclass
class TrainConfig:
    """Single source of truth for training hyperparameters. Everything a run
    needs to be reproduced lives here; it gets snapshotted into every
    checkpoint (see Trainer)."""

    model_name: str = "baseline_cnn"
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    patience: int = 7  # early stop on val_acc plateau

    seed: int = SPLIT_SEED  # reuse the 42 discipline already in constants.py
    device: str = "auto"  # cuda > mps > cpu, resolved in Trainer
    amp: bool = False  # MPS AMP is flaky — opt-in only

    overfit_batch: bool = False
    resume: bool = False  # resume run_name from checkpoints/<run_name>/last.pt
    num_workers: int = 4
