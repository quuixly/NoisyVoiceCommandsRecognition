import json
from pathlib import Path

from src.constants import COMMANDS_TO_PROCESS, LABELS_PATH

LABEL_TO_INDEX: dict[str, int] = {cmd: i for i, cmd in enumerate(COMMANDS_TO_PROCESS)}
INDEX_TO_LABEL: dict[int, str] = {i: cmd for cmd, i in LABEL_TO_INDEX.items()}


def save_label_map(path: Path = LABELS_PATH) -> None:
    """Snapshots the label map to disk for a future standalone inference codebase
    that won't import src.constants. The Dataset itself always uses LABEL_TO_INDEX
    directly, so it can never drift out of sync with this file."""
    path.write_text(json.dumps(LABEL_TO_INDEX, indent=2))
