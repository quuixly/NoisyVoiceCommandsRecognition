import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds random/numpy/torch's global RNGs for reproducible model init and
    training. No effect on DataLoader shuffle order — get_dataloaders() seeds
    its own generator independently (src/torch_dataset.py).

    CUDA gets cudnn.deterministic for full run-to-run reproducibility. MPS has
    no equivalent knob — expect minor run-to-run variance there even with the
    same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
