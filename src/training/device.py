import torch


def resolve_device(name: str = "auto") -> torch.device:
    """`auto` prefers cuda, then Apple MPS, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
