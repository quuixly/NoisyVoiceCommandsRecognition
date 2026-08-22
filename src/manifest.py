import csv
from pathlib import Path

from src.constants import MANIFEST_PATH

MANIFEST_COLUMNS = [
    "file_path",
    "command",
    "group_id",
    "kind",
    "transform",
    "feature_path",
    "feature_idx",
    "split",
]


def write_manifest(rows: list[dict], path: Path = MANIFEST_PATH) -> None:
    """Overwrites path wholesale. Missing columns default to ""."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))
