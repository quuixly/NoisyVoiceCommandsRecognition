from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import MANIFEST_PATH, SPLIT_SEED
from src.labels import LABEL_TO_INDEX
from src.manifest import read_manifest


class SpeechCommandsDataset(Dataset):
    """One split's rows, filtered from the manifest. Features are mmap'd from
    features/<split>.npy (see FeatureExtractor) — __getitem__ is a slice, not a file open."""

    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        split: str = "train",
        label_map: dict[str, int] = LABEL_TO_INDEX,
        rows: list[dict] | None = None,
    ) -> None:
        all_rows = rows if rows is not None else read_manifest(manifest_path)
        split_rows = [r for r in all_rows if r["split"] == split and r["feature_path"]]
        if not split_rows:
            raise ValueError(
                f"No rows for split={split!r} in {manifest_path} — "
                "did you run feature extraction + split yet?"
            )

        feature_paths = {r["feature_path"] for r in split_rows}
        if len(feature_paths) != 1:
            raise ValueError(
                f"split={split!r} rows point at {len(feature_paths)} different "
                "packed feature arrays — expected exactly one per split"
            )
        self.features = np.load(feature_paths.pop(), mmap_mode="r")

        # precomputed once — avoids per-getitem label_map lookup + dict indexing
        self.indices = [int(r["feature_idx"]) for r in split_rows]
        self.labels = [label_map[r["command"]] for r in split_rows]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        features = self.features[self.indices[idx]].copy()  # copy out of read-only mmap
        return torch.from_numpy(features).float(), self.labels[idx]


def get_dataloaders(
    manifest_path: Path = MANIFEST_PATH,
    batch_size: int = 32,
    num_workers: int = 4,
    label_map: dict[str, int] = LABEL_TO_INDEX,
    seed: int = SPLIT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parses manifest.csv once, shared across all 3 splits."""
    all_rows = read_manifest(manifest_path)

    train_ds = SpeechCommandsDataset(manifest_path, "train", label_map, rows=all_rows)
    val_ds = SpeechCommandsDataset(manifest_path, "val", label_map, rows=all_rows)
    test_ds = SpeechCommandsDataset(manifest_path, "test", label_map, rows=all_rows)

    generator = torch.Generator().manual_seed(seed)  # seeded to match SPLIT_SEED reproducibility
    loader_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=generator, **loader_kwargs
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    )
