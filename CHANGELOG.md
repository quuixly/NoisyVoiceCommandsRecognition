# Changelog

Notable changes to this project. Pre-release, unversioned thesis project — everything
lives under `Unreleased` until that changes.

## [Unreleased]

### Added

- Configuration system: `configs/default.yaml` holds every knob in the project, with
  experiment files that override only what they change and `--set key.path=value` for
  ad-hoc changes.
- HTML run reports
- Noise-robustness evaluation: accuracy measured across a sweep of signal-to-noise
  ratios
- A log-mel feature front end as an alternative to MFCC, switchable from the config with
  no model changes.
- Automated test suite (27 tests, ~17 seconds, no dataset required) covering
  configuration, splitting, features, every model, and a full pipeline round trip.

### Changed

- Train/test splitting is now speaker-aware. 
- Split ratios are now accurate.
- Augmentation now happens during training rather than being written to disk. 
- One entry point: `main.py`. 
- Feature extraction settings corrected: the previous configuration requested as many
  output coefficients as it had frequency bands, so roughly a quarter of every extracted
  feature carried filterbank artifacts rather than speech.
- Charts and audio previews are generated from the same feature code the model consumes,
  so a plot can no longer show something different from what the network sees.
- Validation and test data are now processed deterministically, so validation curves
  reflect model progress rather than random variation in preprocessing.

### Fixed

- `--set some.key=1e-3` silently passed the text "1e-3" through to the optimizer instead
  of the number, because YAML only recognizes that form with an explicit decimal point.
- Validation and test data were being augmented like training data, making validation
  accuracy noisy and slightly pessimistic.
- The `gain` augmentation had no effect whatsoever: the feature normalization step
  removes exactly the change a volume adjustment produces (measured difference after a
  4x volume change: 0.000008). 
### Removed

- The precomputed `augmented/` audio directory and the per-file feature cache, both
  superseded by on-the-fly augmentation.
