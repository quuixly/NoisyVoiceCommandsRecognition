# NoisyVoiceCommandsRecognition

Bachelor's thesis project: automatic recognition of voice commands (`up`, `down`, `left`,
`right`, `stop`, `go`) in noisy environments.

Data pipeline (download → augment → MFCC features → `DataLoader`s) plus a training loop
for a small CNN baseline. Model export/mobile deployment not implemented yet.

## Getting started

```bash
git clone git@github.com:quuixly/NoisyVoiceCommandsRecognition.git
cd NoisyVoiceCommandsRecognition
uv sync
```

`uv` manages the virtualenv; you don't need to activate it manually as long as you run
things through `uv run`.

```bash
uv add <package-name>   # add a dependency
```

## Run

Three entry points, for different purposes:

```bash
uv run python main.py             # fast debug run (~5 files/command, generates preview MFCC plots)
uv run python prepare_dataset.py  # full pipeline (~100 files/command, builds the training-ready dataset)
uv run python train.py            # trains a model on whatever prepare_dataset.py last built
```

`main.py` is for quickly sanity-checking a change to the pipeline — it downloads the
dataset, augments a handful of files per command, and saves MFCC plots you can eyeball.
It does **not** produce `manifest.csv`/`features/`, so `train.py` can't run against it.

`prepare_dataset.py` is the real pipeline: downloads the dataset, augments ~100 source
files per command (plus every transform/combo — hundreds of variants each), extracts
MFCC+delta+delta² features, assigns train/val/test splits, and writes everything needed
to build PyTorch `DataLoader`s. Much slower/heavier than `main.py` — this is what
`train.py` actually consumes.

```bash
uv run python train.py --run-name baseline --epochs 30
```

Common flags (all have sane defaults — see `src/config.py`'s `TrainConfig` for every
field): `--model-name`, `--run-name`, `--epochs`, `--batch-size`, `--lr`, `--patience`,
`--device {auto,cuda,mps,cpu}`, `--resume` (continues `checkpoints/<run-name>/last.pt`),
`--overfit-batch` (see Testing below).

Each run writes to its own `checkpoints/<run-name>/` and `reports/<run-name>/` — running
with a different `--run-name` never overwrites a previous run.

## Testing

There's no unit test suite — the pipeline and training loop are validated by actually
running them at small scale, fast:

```bash
# 1. Tiny real dataset (seconds, not the full ~100/command run)
uv run python -c "from prepare_dataset import prepare_dataset; prepare_dataset(sources_per_command=5)"

# 2. Wiring check: can the model actually learn anything at all?
uv run python train.py --run-name smoke --overfit-batch
# expect: "Overfit-batch reached 100% train accuracy" within a few dozen steps.
# If it doesn't, something upstream (data/loss/optimizer) is broken, not the model.

# 3. A couple of real epochs on the tiny set
uv run python train.py --run-name smoke --epochs 2

# 4. Resume works and doesn't corrupt state
uv run python train.py --run-name smoke --epochs 3 --resume
# history.csv should have exactly 3 rows (appended, not duplicated), and epoch
# should continue from where it left off, not restart at 1.

# 5. Clean up the smoke run
rm -rf checkpoints/smoke reports/smoke
```

This is the same procedure used to validate the training loop itself — it's how two real
bugs got caught during development (a wiring test that was silently using the wrong
hyperparameters, and a checkpoint ordering bug that would have let `--resume` overwrite
a good `best.pt` with a worse one). Re-run it after touching `src/trainer.py`,
`src/models/`, or `src/torch_dataset.py`.

## Analyzing results

Every run under a given `--run-name` produces:

```
checkpoints/<run-name>/
  best.pt     # highest-val_acc checkpoint so far (model + optimizer + scheduler + config)
  last.pt     # most recent epoch (what --resume continues from)
reports/<run-name>/
  history.csv        # one row per epoch: epoch, train_loss, val_loss, val_acc, lr, secs
  test_metrics.json   # written by evaluate.py — see below
```

**During/after training** — plot `history.csv` (loss/accuracy curves, e.g. with pandas +
matplotlib) to check for overfitting (val loss rising while train loss keeps falling) or
an unstable LR.

**On the held-out test set**, once you have a `best.pt` you trust:

```bash
uv run python evaluate.py --run-name baseline
```

Writes `reports/<run-name>/test_metrics.json`: overall accuracy, the full confusion
matrix, and per-class precision/recall/F1. The confusion matrix is the useful part for
this thesis specifically — acoustically close pairs (`up`/`down`, `left`/`right`) mixing
up tells you which augmentations need strengthening, not just that accuracy is X%.

Checkpoints are plain `torch.load(..., weights_only=False)`-able dicts if you want to
poke at them directly — `ckpt["config"]` has the exact hyperparameters that run used.

## Project structure

```
main.py               fast debug entry point
prepare_dataset.py    full training-prep entry point
train.py               training entry point
evaluate.py             test-set evaluation entry point
src/
  constants.py         shared paths + pipeline constants (N_MFCC, MAX_PAD_LEN, ...)
  logging_config.py     shared logging setup for every entry point
  data_loader.py         DatasetLoader: downloads/caches raw audio from Kaggle
  augmenter.py            Augmenter: applies audiomentations transforms + combos
  feature_extractor.py     extract_mfcc() + FeatureExtractor: MFCC/delta/delta² -> packed arrays
  splitter.py              assign_splits(): group-aware, per-command train/val/test split
  manifest.py               read/write manifest.csv
  labels.py                 command <-> index label map
  torch_dataset.py          SpeechCommandsDataset + get_dataloaders()
  visualization.py          MFCC plots, raw-vs-augmented comparison plots
  config.py                TrainConfig — single source of training hyperparameters
  seed.py                   set_seed(): reproducible model init (CUDA fully, MPS best-effort)
  models/                   build_model(name) factory + baseline_cnn.py
  trainer.py                Trainer: epoch loop, checkpointing, early stop, resume
  metrics.py                confusion matrix, per-class precision/recall/F1
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
  -> Trainer.fit -> checkpoints/<run>/         (train.py)
  -> evaluate.py -> reports/<run>/             (evaluate.py)
```

`data/`, `augmented/`, `mfcc/`, `features/`, `manifest.csv`, `labels.json`,
`checkpoints/`, and `reports/` are all gitignored — regenerated locally by running the
pipeline, not checked in.

## Status / what's missing

Data pipeline and training loop are done (baseline CNN, ~187K params). Not implemented
yet: noise-robustness evaluation (SNR sweep, FAR/FRR — blocked on sourcing real
background-noise/negative-class audio, see `TODO.md` Phase 2) and mobile export
(ONNX + quantization, `TODO.md` Phase 3).

See `TODO.md` and `CHANGELOG.md` for more detail.
