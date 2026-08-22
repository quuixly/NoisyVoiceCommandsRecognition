import logging
import os
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

from src.constants import FEATURES_DIR, MAX_PAD_LEN, N_MFCC, PRE_EMPHASIS, TOP_DB

FEATURE_SHAPE = (3 * N_MFCC, MAX_PAD_LEN)


def extract_mfcc(
    file: Path,
    n_mfcc: int = N_MFCC,
    max_pad_len: int = MAX_PAD_LEN,
) -> np.ndarray:
    """(3*n_mfcc, max_pad_len) float32: static MFCC rows 0..n_mfcc-1, delta
    n_mfcc..2*n_mfcc-1, delta2 2*n_mfcc..3*n_mfcc-1; per-feature z-score, padded/truncated."""
    y, sr = librosa.load(file, sr=16_000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # pre-emphasis: boosts high frequencies before MFCC
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])

    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160, n_mels=40, fmin=20, fmax=sr // 2
    )
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    combined = np.vstack([mfccs, delta, delta2])

    combined = (combined - combined.mean(axis=1, keepdims=True)) / (
        combined.std(axis=1, keepdims=True) + 1e-8
    )

    if combined.shape[1] < max_pad_len:
        pad_width = max_pad_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode="constant")
    else:
        combined = combined[:, :max_pad_len]

    return combined.astype(np.float32, copy=False)  # force float32: librosa dtype not guaranteed


def _extract_worker(args: tuple[str, str, int]) -> str | None:
    """Picklable ProcessPoolExecutor worker: computes features for one file,
    writes directly into the split's preallocated memmap at idx_in_split.
    Returns traceback string on failure, None on success."""
    file_path_str, memmap_path_str, idx_in_split = args
    try:
        features = extract_mfcc(Path(file_path_str))
        arr = np.load(memmap_path_str, mmap_mode="r+")
        arr[idx_in_split] = features
        arr.flush()
        return None
    except Exception:
        return traceback.format_exc()  # full traceback — don't relabel real bugs as "bad file"


class FeatureExtractor:
    """Converts manifest rows (already split-assigned) into MFCC/delta/delta2
    features, packed one array per split (features/<split>.npy) instead of
    one .npy per sample."""

    def __init__(self, features_dir: Path = FEATURES_DIR, n_jobs: int | None = None) -> None:
        self.features_dir = features_dir
        self.n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)

    def _split_array_path(self, split: str) -> Path:
        return self.features_dir / f"{split}.npy"

    def _open_or_create_split_array(self, split: str, n_rows: int) -> tuple[Path, bool]:
        """Returns (path, already_complete) — skips re-extraction if the split
        array already exists with the exact shape/dtype this run expects."""
        path = self._split_array_path(split)
        expected_shape = (n_rows, *FEATURE_SHAPE)
        if path.exists():
            try:
                existing = np.load(path, mmap_mode="r")
                if existing.shape == expected_shape and existing.dtype == np.float32:
                    return path, True
            except Exception:
                pass  # corrupt/partial file from an interrupted run — recreate it
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=expected_shape)
        return path, False

    def extract_all(self, rows: list[dict]) -> list[dict]:
        missing_split = [r for r in rows if not r.get("split")]
        if missing_split:
            raise ValueError(
                "FeatureExtractor.extract_all requires rows with 'split' already "
                "assigned — run assign_splits() first, so features can be packed "
                "one array per split."
            )

        self.features_dir.mkdir(parents=True, exist_ok=True)

        rows_by_split: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            rows_by_split[row["split"]].append(row)

        task_rows: list[dict] = []
        task_args: list[tuple[str, str, int]] = []
        for split, split_rows in rows_by_split.items():
            path, already_complete = self._open_or_create_split_array(split, len(split_rows))
            for idx_in_split, row in enumerate(split_rows):
                row["feature_path"] = str(path)
                row["feature_idx"] = str(idx_in_split)
                if not already_complete:
                    task_rows.append(row)
                    task_args.append((row["file_path"], str(path), idx_in_split))
            if already_complete:
                logging.info(f"Split '{split}' features already packed, skipping ({len(split_rows)} rows)")

        if task_args:
            if self.n_jobs > 1:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                    results = list(pool.map(_extract_worker, task_args, chunksize=64))
            else:
                results = [_extract_worker(a) for a in task_args]
        else:
            results = []

        n_errors = 0
        for row, error in zip(task_rows, results):
            if error is not None:
                n_errors += 1
                row["feature_path"] = ""
                row["feature_idx"] = ""
                logging.error(f"Feature extraction failed for {row['file_path']}:\n{error}")

        logging.info(f"Feature extraction complete: {len(rows) - n_errors}/{len(rows)} succeeded")
        return rows
