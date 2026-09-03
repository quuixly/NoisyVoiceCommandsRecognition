# NoisyVoiceCommandsRecognition

Bachelor's thesis project: recognizing the voice commands `up`, `down`, `left`, `right`,
`stop`, `go` in noisy environments, aimed at running on a phone.

Everything — data preparation, augmentation, features, model, training, evaluation and
reporting — is driven by one YAML file and one command.

## TL;DR

**Set up once.**

```bash
git clone git@github.com:quuixly/NoisyVoiceCommandsRecognition.git
cd NoisyVoiceCommandsRecognition
uv sync
```

**Run the whole thing.** One command: downloads the audio, splits it by speaker, trains,
evaluates on the test set, sweeps the noise levels, and writes a report.

```bash
uv run python main.py run --set train.run_name=baseline
open reports/baseline/report.html
```

First time takes ~1 min to prepare the corpus plus ~9 s per epoch. Later runs reuse the
prepared corpus automatically.

**Try something different.** Every knob is a config key; nothing is hardcoded.

```bash
uv run python main.py run --set train.run_name=faster --set train.lr=1e-3 --set train.epochs=50
uv run python main.py run --set train.run_name=logmel --set features.type=logmel
uv run python main.py run --set train.run_name=crnn   --set model.name=crnn
```

**Compare what you tried.**

```bash
uv run python main.py summary                     # every run -> reports/summary.html
uv run python main.py summary baseline logmel     # or just these two
```

**Run a whole batch of experiments unattended.**

```bash
uv run python main.py sweep                          # every configs/experiments/*.yaml + the defaults
uv run python main.py sweep --set train.epochs=40    # same sweep, 40 epochs each
uv run python main.py sweep logmel crnn              # only these
```

It trains, evaluates and reports each one, keeps going if one fails, and builds the
comparison page at the end.

**Check nothing is broken** (seconds, no dataset needed):

```bash
uv run pytest tests -q
```

---

## The commands in full

Everything goes through `main.py`; there is no second entry point and no shell script.

| Command                             | What it does                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `main.py run`                       | prepare (if stale) + train + evaluate + report. **The one you usually want.** |
| `main.py sweep [names...]`          | that chain once per experiment, each in its own process, then the summary page |
| `main.py prepare`                   | download audio, split by speaker, decode waveforms into packed arrays         |
| `main.py train`                     | train only, then write the run's report                                       |
| `main.py evaluate --run <name>`     | test metrics + SNR sweep for a trained run, refresh its report                |
| `main.py report --run <name>`       | rebuild a report from existing artifacts, no compute                          |
| `main.py summary [runs...]`         | one comparison page over several runs                                         |
| `main.py compare runA runB`         | just the overlaid validation curves, as a PNG                                 |
| `main.py preview`                   | clean vs augmented waveforms and features, to sanity-check augmentation       |

Every command except `report`, `summary` and `compare` accepts `-c <experiment.yaml>` and
repeatable `--set key.path=value`. `sweep` takes `--set` too and forwards it to every
experiment.

`main.py run` skips preparation when the manifest on disk was already built from the same
`data` and `split` settings, and re-prepares when they differ — so changing
`data.max_per_command` or `split.by` can't leave you training against the previous corpus.
Force a skip with `--skip-prepare` if you know better.

## Configuring a run

`configs/default.yaml` holds every knob there is — data, split, features, augmentation,
model, training, evaluation. Nothing is hardcoded in the source. Two ways to change it:

```bash
# 1. Ad-hoc overrides. Any key, any depth.
uv run python main.py train --set train.lr=1e-3 --set model.channels=[64,128,256] --set features.type=logmel

# 2. An experiment file, deep-merged onto the defaults — name only what you change.
uv run python main.py train -c configs/experiments/logmel.yaml --set train.run_name=logmel
```

Shipped experiments: `logmel.yaml` (log-mel front end instead of MFCC), `crnn.yaml`
(Conv + BiGRU), `smoke.yaml` (60 clips per command, 2 epochs — a wiring check that runs
in seconds).

A typo in a config key is an error, not a silent no-op:

```
ValueError: Unknown config key(s) for TrainConfig: ['epocs']
```

The fully resolved config is written to `reports/<run>/config.yaml` **and** into every
checkpoint, so any run can be reproduced or evaluated from its own artifacts.

## Running experiments

The loop is: change one thing, give it a name, run it, compare.

```bash
uv run python main.py run --set train.run_name=baseline
uv run python main.py run --set train.run_name=wider --set model.channels=[64,128,256]
uv run python main.py run --set train.run_name=logmel --set features.type=logmel
uv run python main.py summary
open reports/summary.html
```

`train.run_name` is the whole isolation mechanism — it names `checkpoints/<name>/` and
`reports/<name>/`, so two experiments never touch each other's files. Omit it and you get
a timestamp, which is fine for a one-off and useless for a comparison. **Name your runs.**

When a change is worth keeping, promote it from a `--set` flag to a file in
`configs/experiments/`, naming only what differs from the defaults:

```yaml
# configs/experiments/wider.yaml
model:
  channels: [64, 128, 256]
train:
  epochs: 50
```

Then `main.py sweep` picks it up automatically along with everything else in that
directory:

```bash
uv run python main.py sweep                              # baseline + every experiment file, then the summary page
uv run python main.py sweep --set train.epochs=40        # same, but force 40 epochs everywhere
uv run python main.py sweep --set train.batch_size=128
uv run python main.py sweep logmel crnn                  # only these two
```

Each experiment becomes a run named after its config file, and runs in its own child
process — a crash or an out-of-memory in one config leaves the others untouched and gives
each run clean accelerator state. Per-run logs land in `reports/sweep_logs/<name>.log`
(and stream to your terminal as they arrive), and one failing experiment doesn't abort the
rest — it's recorded and the sweep continues, because a six-hour sweep that dies on its
second config and discards the rest is worse than one that reports a gap.

### Comparing honestly

`reports/summary.html` is a table of every run — model, features, parameters, epochs, best
validation accuracy, test accuracy, macro F1, accuracy at the worst SNR — plus overlaid
validation curves and overlaid noise-robustness curves.

It also carries a **corpus fingerprint** per run, a hash of the `data` and `split` config.
Runs that don't share one were trained and tested on different clips, so their accuracies
aren't comparable — the page says so at the top rather than quietly plotting them on one
axis:

> **These runs do not share one corpus.** 2 different data/split configurations are on
> this page…

Pass explicit run names to compare within one group.

The noise-robustness chart is usually the one that decides things: two models can tie on
clean accuracy and come apart badly once noise is added, and only the slope shows it.

### The knobs worth knowing

| Key                                 | What it does                                                          |
| ----------------------------------- | --------------------------------------------------------------------- |
| `data.max_per_command`              | `null` = every clip (~3800/command). An int caps it for fast runs.    |
| `split.by`                          | `speaker` (default) or `file`. See "Leakage" below.                   |
| `features.type`                     | `mfcc` or `logmel` — swaps the front end; models adapt automatically. |
| `augment.min_k` / `max_k`           | How many random transforms each training clip gets.                   |
| `augment.transforms.<name>.enabled` | Turn one transform on/off without touching code.                      |
| `model.name`                        | `baseline_cnn`, `crnn`, `logreg`.                                     |
| `model.channels`                    | Conv widths — architecture size is config, not source.                |
| `eval.snr_db`                       | The SNR points swept at evaluation time.                              |

## How the pipeline works

```
data/<command>/*.wav                       (main.py prepare: DatasetLoader)
  -> manifest.csv + speaker splits          (splits.assign_splits)
  -> waveforms/<split>.npy                  (WaveformStore — decode once)
  -> augment + featurize per __getitem__    (RandomAugmenter + Featurizer)
  -> Trainer.fit -> checkpoints/<run>/      (main.py train)
  -> reports/<run>/report.html              (main.py evaluate / report)
  -> reports/summary.html                   (main.py summary, across runs)
```

`main.py run` is that whole column in one command; the individual steps exist for when you
want to redo just one of them.

**Augmentation is on-the-fly.** There is no `augmented/` directory any more. Each
training item gets a fresh random transform chain every time it is drawn, so the model
sees the whole ~23K-clip corpus under endlessly varying distortion instead of a fixed
cross-product of variants over a small subset of it. Measured cost: ~4 ms per clip, ~13 s
per epoch across 7 workers.

**Leakage.** Speech Commands names files `<speaker>_nohash_<n>.wav`, and a speaker
records many clips. Splitting per file puts the same voice in train and test, and the
resulting test accuracy measures speaker memorization. `split.by: speaker` groups every
clip by its speaker hash and assigns whole speakers to splits, so no voice crosses a
split boundary.

**One front end.** `src/data/features.py` is the only implementation of
"waveform → what the model sees", and the plots call it too. A spectrogram in a report is
always the array the network actually consumed.

## What you get to look at

`main.py train` and `main.py evaluate` both leave `reports/<run>/report.html` — one
self-contained file (images inlined, nothing external to break) with:

- headline tiles: test accuracy, macro F1, parameter count, checkpoint epoch
- train/val loss and accuracy curves, best epoch marked
- row-normalized confusion matrix, with the raw counts as a table
- per-class precision / recall / F1, with support counts
- accuracy vs SNR — the noise-robustness headline metric
- the most confident misclassifications, rendered as features
- the exact config the run used

Every chart is paired with a table of the same numbers, and the individual PNGs are in
`reports/<run>/figures/` if you want them in the thesis directly.

Across runs, `main.py summary` writes `reports/summary.html` — the comparison table, overlaid
validation curves, and overlaid noise-robustness curves, with its figures in
`reports/figures/`.

## Testing

```bash
uv run pytest tests -q
```

37 tests over a synthetic 12-speaker corpus: config merging and override typing, speaker
leakage, feature shape and invariances, every model against every front end, the
prepare-staleness check, the full `main.py run` chain, and the summary page's corpus-mismatch
warning. About a minute, no dataset required.

A wiring check against the real data, when you've changed the training path:

```bash
uv run python main.py train -c configs/experiments/smoke.yaml --set train.overfit_batch=true
# expect: "Overfit-batch reached 100% at step N — wiring OK"
```

If one batch can't be memorized, something upstream is broken and no amount of epochs
will fix it.

## Project layout

```
configs/
  default.yaml          every knob, documented inline
  experiments/          sparse overrides: logmel, crnn, smoke
main.py                 the only entry point: argument parsing and dispatch
src/
  config.py             nested dataclasses, YAML deep-merge, --set overrides
  pipeline.py           the stages: prepare, train, evaluate, report, run, sweep
  evaluation.py         test metrics, SNR sweep, worst-error selection
  labels.py             command <-> index
  data/
    kaggle.py           dataset download
    prepare.py          manifest + splits + waveform packs
    splits.py           speaker-aware, ratio-accurate splitting
    waveforms.py        decode-once packed waveform cache
    augment.py          on-the-fly random transform chains, SNR mixing
    features.py         the single MFCC / log-mel front end
    dataset.py          Dataset + DataLoaders
    manifest.py         manifest.csv read/write
  models/               build_model(config) registry: baseline_cnn, crnn, logreg
  training/             Trainer, metrics, device, seeding
  visualization/
    theme.py            palette + matplotlib style, shared by every figure
    charts.py           curves, confusion, per-class, SNR, run overlays
    audio.py            waveform + feature figures, via the real Featurizer
    report.py           one run's self-contained HTML report
    summary.py          the cross-run comparison page
tests/
```

`data/`, `waveforms/`, `manifest.csv`, `labels.json`, `checkpoints/` and `reports/` are
gitignored — regenerated locally, never checked in.

## Status

Data pipeline, training, evaluation and reporting are done. Not implemented yet: real
recorded background noise (the SNR sweep currently mixes synthetic noise), FAR/FRR against
a negative class, and mobile export (ONNX + quantization). See `TODO.md`.
