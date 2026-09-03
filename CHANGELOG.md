# Changelog

Notable changes to this project. Pre-release, unversioned thesis project — everything
lives under `Unreleased` until that changes.

## [Unreleased]

### Added

- Configuration system: `configs/default.yaml` holds every knob in the project, with
  experiment files that override only what they change and `--set key.path=value` for
  ad-hoc changes. Unknown keys are rejected instead of silently ignored. The resolved
  configuration is saved with each run and embedded in every checkpoint, so a run can be
  reproduced or evaluated from its own artifacts.
- A single `nvcr` command with subcommands for preparing data, training, evaluating,
  reporting, comparing runs, and previewing augmentation — replacing four separate
  scripts.
- HTML run report: one self-contained file per run with headline figures, training
  curves, a confusion matrix, per-class scores, an accuracy-vs-noise curve, the model's
  most confident mistakes, and the exact configuration used. Every chart is paired with
  a table of the same numbers.
- Noise-robustness evaluation: accuracy measured across a sweep of signal-to-noise
  ratios, not just on clean audio.
- Two more model architectures for comparison (a recurrent model and a linear baseline),
  selectable by name from the config.
- A log-mel feature front end as an alternative to MFCC, switchable from the config with
  no model changes.
- Automated test suite (27 tests, ~17 seconds, no dataset required) covering
  configuration, splitting, features, every model, and a full pipeline round trip.

### Changed

- **Train/test splitting is now speaker-aware.** Previously the same speaker's recordings
  could appear in both training and test data, so reported test accuracy was partly
  measuring memorization of individual voices. Results from before this change are not
  comparable with results after it.
- Split ratios are now accurate. The previous per-command approach drifted to 77/13/10
  when asked for 70/15/15, because speakers record several different commands.
- **Augmentation now happens during training rather than being written to disk.** The
  previous approach precomputed every combination of transforms for a small subset of the
  audio; the model now sees the entire corpus (~3,800 recordings per command instead of
  100) under continuously varying distortion. Hours of preprocessing and gigabytes of
  generated audio are no longer needed.
- **One entry point again: `main.py`.** The `nvcr` console script and the
  `scripts/sweep.sh` shell script are gone; `main.py` parses arguments and dispatches,
  and `src/pipeline.py` holds each stage as a plain function. The sweep is now Python
  rather than bash — it discovers experiment files itself, forwards `--set` overrides
  instead of `EPOCHS`/`EXTRA` environment variables, and runs each experiment in its own
  child process so one crash cannot end a long sweep.
- `src/viz/` renamed to `src/visualization/` — the abbreviation said nothing the full
  word doesn't. No behaviour change.
- Experiment configs now cover the comparisons the thesis actually needs: the three
  architectures, both feature front ends, an augmentation ablation and a heavier
  augmentation setting, a small on-device candidate, and a capacity probe.
- Commands that take a run now accept a path to it as well as its name
  (`--run reports/baseline`, `--run checkpoints/baseline/best.pt`), so shell completion
  is enough.
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
  4x volume change: 0.000008). Nearly half of all previously generated audio variants
  differed from another variant only by this no-op. It is now disabled by default, with
  a test pinning the reason.

### Removed

- The precomputed `augmented/` audio directory and the per-file feature cache, both
  superseded by on-the-fly augmentation.
- The four root-level scripts (`main.py`, `prepare_dataset.py`, `train.py`,
  `evaluate.py`), replaced by the `nvcr` command.
