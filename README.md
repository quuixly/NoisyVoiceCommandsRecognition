# NoisyVoiceCommandsRecognition

Bachelor's thesis project: automatic recognition of voice commands (`up`, `down`, `left`,
`right`, `stop`, `go`) in noisy environments.

Currently a data pipeline — audio download, augmentation, MFCC feature extraction, and
`DataLoader`-ready output. Model training is not implemented yet.

## Getting started

```bash
git clone git@github.com:quuixly/NoisyVoiceCommandsRecognition.git
cd NoisyVoiceCommandsRecognition
uv sync
```

`uv` manages the virtualenv; you don't need to activate it manually as long as you run
things through `uv run`.

### Run

There are two entry points, for different purposes:

```bash
uv run python main.py             # fast debug run (~5 files/command, generates preview MFCC plots)
uv run python prepare_dataset.py  # full pipeline (~100 files/command, builds the training-ready dataset)
```

`main.py` is for quickly sanity-checking a change to the pipeline — it downloads the
dataset, augments a handful of files per command, and saves MFCC plots you can eyeball.

`prepare_dataset.py` is the real pipeline: downloads the dataset, augments ~100 source
files per command (plus every transform/combo — hundreds of variants each), extracts
MFCC+delta+delta² features, assigns train/val/test splits, and writes everything needed
to build PyTorch `DataLoader`s. This is a much slower, heavier run than `main.py`.

```bash
uv add <package-name>   # add a dependency
```

## Project structure

```
main.py               fast debug entry point
prepare_dataset.py     full training-prep entry point
src/
  constants.py         shared paths + pipeline constants (N_MFCC, MAX_PAD_LEN, ...)
  logging_config.py     shared logging setup for both entry points
  data_loader.py         DatasetLoader: downloads/caches raw audio from Kaggle
  augmenter.py            Augmenter: applies audiomentations transforms + combos
  feature_extractor.py     extract_mfcc() + FeatureExtractor: MFCC/delta/delta² -> packed arrays
  splitter.py              assign_splits(): group-aware, per-command train/val/test split
  manifest.py               read/write manifest.csv
  labels.py                 command <-> index label map
  torch_dataset.py          SpeechCommandsDataset + get_dataloaders()
  visualization.py          MFCC plots, raw-vs-augmented comparison plots
```

Everything under `src/` is documented in detail in `CLAUDE.md`, aimed at whoever (human
or AI assistant) is picking the pipeline back up.

## Pipeline stages

```
data/<command>/*.wav                         (DatasetLoader)
  -> augmented/<command>/*.wav                (Augmenter)
  -> manifest.csv (raw + augmented rows)      (Augmenter.manifest_rows)
  -> manifest.csv + split column              (assign_splits)
  -> features/<split>.npy (packed array)      (FeatureExtractor)
  -> manifest.csv + feature_path/feature_idx  (FeatureExtractor)
  -> SpeechCommandsDataset / DataLoader        (torch_dataset.py)
```

`data/`, `augmented/`, `mfcc/`, `features/`, `manifest.csv`, and `labels.json` are all
gitignored — regenerated locally by running the pipeline, not checked in.

## Status / what's missing

No model architecture or training loop yet — the pipeline stops at ready-to-iterate
`DataLoader`s (`src/torch_dataset.py`). Next step is `src/model.py` + a training script.

See `TODO.md` and `CHANGELOG.md` for more detail.
