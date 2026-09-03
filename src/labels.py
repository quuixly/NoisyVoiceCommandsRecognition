import json
from pathlib import Path

LABELS_PATH = Path("labels.json")


def label_to_index(commands: list[str]) -> dict[str, int]:
    """Index = position in `data.commands`, so the mapping is defined by the config
    and cannot drift between manifest building, training, and evaluation."""
    return {command: i for i, command in enumerate(commands)}


def index_to_label(commands: list[str]) -> dict[int, str]:
    return dict(enumerate(commands))


def save_label_map(commands: list[str], path: Path = LABELS_PATH) -> None:
    """Snapshot for a future on-device codebase that won't import this package."""
    path.write_text(json.dumps(label_to_index(commands), indent=2))
