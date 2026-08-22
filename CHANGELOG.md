# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `prepare_dataset.py`: full training-prep entry point, separate from `main.py`'s fast
  debug run — downloads data, augments ~100 sources/command, extracts features, assigns
  splits, writes `manifest.csv`.
- `src/manifest.py`: `manifest.csv` read/write (raw + augmented rows, `group_id` for
  leakage-safe splitting).
- `src/labels.py`: `LABEL_TO_INDEX`/`INDEX_TO_LABEL`, `save_label_map()` -> `labels.json`.
- `src/feature_extractor.py`: `extract_mfcc()` (MFCC + delta + delta², pre-emphasis,
  per-feature z-score, fixed-length pad/truncate), moved and expanded from the draft in
  `TODO.md`. `FeatureExtractor.extract_all()` runs it in parallel via
  `ProcessPoolExecutor`, packing output one array per split
  (`features/<split>.npy`, shape `(n, 3*N_MFCC, MAX_PAD_LEN)`) instead of one `.npy` per
  sample.
- `src/splitter.py`: `assign_splits()` — group-aware (raw + all its augmented children
  stay together), per-command-balanced train/val/test split, seeded.
- `src/torch_dataset.py`: `SpeechCommandsDataset` + `get_dataloaders()` — mmaps the
  packed per-split feature arrays, returns ready-to-train PyTorch `DataLoader`s.
- `src/logging_config.py`: shared `setup_logging()` for both entry points.
- `README.md`, this `CHANGELOG.md`.

### Changed

- `main.py` reduced to a thin orchestrator (`DatasetLoader` -> `Augmenter` ->
  `generate_mfcc`); pipeline logic now lives in `src/`.
- `Augmenter` now builds `manifest_rows` live during `augment_audio()` (one row per raw
  file, one per successfully-written augmented output) instead of requiring a later
  re-derivation from disk.
- `prepare_dataset.py` pipeline order: splits are now assigned _before_ feature
  extraction, since `FeatureExtractor` needs to know each row's split up front to pack
  features one array per split.
- `manifest.csv` is now written once at the end of `prepare_dataset()` instead of after
  every stage — both `augment_audio()` and `extract_all()` are independently resumable
  via on-disk existence checks, so the intermediate writes weren't buying anything.
- `torch_dataset.get_dataloaders()`: parses `manifest.csv` once and shares it across all
  three splits (was 3 redundant full CSV reads); labels/feature indices precomputed once
  per dataset instead of looked up per sample; `DataLoader`s now set `num_workers=4`,
  `pin_memory=torch.cuda.is_available()`, `persistent_workers=True`, `prefetch_factor=4`,
  and a seeded `torch.Generator` for reproducible shuffling (matches `SPLIT_SEED`'s
  existing reproducibility discipline).
- `FeatureExtractor`'s worker no longer creates a directory per sample or round-trips the
  input path through a side lookup table; error results now carry a full traceback
  instead of `str(e)`, so real bugs aren't silently relabeled as "bad input file".

### Fixed

- `SpeechCommandsDataset.__getitem__`: samples pulled from the read-only mmap were
  non-writable tensors (PyTorch warns and forbids in-place ops on them); fixed by
  copying the slice out of the mmap instead of relying on `np.asarray()`, which is a
  no-op on an already-`ndarray` view.

### Removed

- Per-sample `.npy` feature files (`features/<command>/<file>.npy`) — replaced by the
  packed per-split arrays above. Cuts ~130K individual file opens per epoch down to 3
  mmap'd arrays.
- Draft `extract_mfcc()`/`generate_mfcc()` code from `TODO.md` — superseded by
  `src/feature_extractor.py`.
