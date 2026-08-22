# Changelog

Notable changes to this project. Pre-release, unversioned thesis project — everything
lives under `Unreleased` until that changes.

## [Unreleased]

### Added

- Full training-prep pipeline: downloads data, augments the full source set, extracts
  MFCC/delta/delta² features, assigns leakage-safe train/val/test splits, and produces
  ready-to-train PyTorch `DataLoader`s. Separate from the existing fast debug run.
- Label map (`labels.json`) and dataset manifest (`manifest.csv`) as the shared
  source of truth between pipeline stages.

### Changed

- Feature storage switched from one file per audio sample to one packed array per
  split — cuts ~130K individual file opens per training epoch down to 3.
  `DataLoader`s tuned accordingly (parallel workers, pinned memory, reproducible
  shuffling).
- Manifest is now written once per full pipeline run instead of after every stage.

### Fixed

- Interrupted pipeline runs (crash, OOM, power loss) could leave corrupted,
  all-zero feature data that looked valid on the next run and would have trained
  silently on garbage — now detected and regenerated instead.
- A single unreadable source audio file no longer aborts an entire multi-hour
  augmentation run.
- Small dataset splits could leave the validation or test set empty, crashing only
  after augmentation and feature extraction had already finished — now caught
  immediately, with guaranteed non-empty splits.
- Removed an unused, unsafe recursive-delete code path in the dataset cache cleanup.
- Non-writable feature tensors (PyTorch warning) fixed at the source.

### Removed

- Draft feature-extraction code superseded by the real implementation.
