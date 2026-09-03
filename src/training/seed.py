import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds random/numpy/torch global RNGs. DataLoader shuffling and augmentation
    seed themselves separately (src/data/dataset.py), so their streams stay
    independent of model init.

    CUDA additionally gets cudnn determinism; MPS has no equivalent knob, so expect
    minor run-to-run variance there even at a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
