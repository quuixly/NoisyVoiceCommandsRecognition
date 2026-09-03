# TODO — Roadmap to on-device command recognition

Goal: a small model that recognizes the 6 commands (`data.commands`) live on a phone,
robustly in noise. The pipeline now runs end to end (`main.py prepare | train | evaluate`);
everything below starts from there.

## Phase 0 — Pipeline (done)

- [x] Config-driven everything: `configs/default.yaml` + `-c experiment.yaml` +
      `--set key.path=value`. Unknown keys raise. Resolved config is saved per run and
      embedded in every checkpoint.
- [x] Single entry point: `main.py` (parsing) + `src/pipeline.py` (stages), replacing
      the four root scripts, then the `nvcr` console script and `scripts/sweep.sh`.
- [x] Speaker-aware splitting — previously the same voice could sit in train and test,
      so test accuracy was partly measuring speaker memorization.
- [x] On-the-fly augmentation over the full corpus, replacing the precomputed
      transform cross-product over a 100-clip subset.
- [x] `gain` disabled: proven no-op after per-row normalization (4x gain moves features
      by 8e-06). Pinned by `tests/test_features.py`.
- [x] Feature front end fixed and made switchable: `n_mels=64, n_mfcc=20` (was 40/40,
      a full-rank DCT), plus a `logmel` option. One implementation shared by the model
      and every plot.
- [x] `WaveformStore` completion markers: an interrupted decode is deleted rather than
      silently reused as zero-filled audio.
- [x] Test suite: 42 tests, ~75s, synthetic corpus, no dataset needed.
- [x] HTML run report + `main.py compare` + `main.py preview`.

## Phase 1 — Baseline model

- [x] `baseline_cnn`: conv stack + global pool, width from config, 186,598 params at the
      default 20-coefficient front end (under the 200K budget).
- [x] Trainer: seeded, AMP opt-in, best/last checkpointing, `--set train.resume=true`,
      early stopping, run-scoped directories.
- [x] Metrics: accuracy, macro F1, confusion matrix, per-class P/R/F1 with support.
- [x] Overfit-a-single-batch wiring check.
- [x] Alternative architectures wired for comparison: `logreg` (the floor any real model
      must clear) and `crnn` (Conv + BiGRU).
- [x] Experiment configs for the comparison: `baseline`, `crnn`, `logreg`, `logmel`,
      `mfcc13`, `no_augment`, `heavy_augment`, `tiny_cnn`, `wide_cnn`.
- [ ] **Run the real comparison**: `uv run python main.py sweep`, then read
      `reports/summary.html`. `baseline_cnn` vs `crnn` vs `logreg`, `mfcc` vs `logmel`,
      and the augmentation gap at low SNR. `main.py compare` overlays the curves.
- [ ] Re-measure the headline accuracy on the speaker-disjoint test set. The old number
      is not comparable — expect it to drop, and that drop is the point.
- [ ] Measure, record time of training, amount of parameters. Include those in reports
- [ ] Basic web interface for manual testing
- [ ] 

## Phase 2 — Noise robustness 

- [x] Eval-time SNR sweep (`eval.snr_db`), plotted as accuracy vs SNR.
- [ ] Replace synthetic gaussian noise in the sweep with **real** recordings: babble,
      traffic, music. `eval.noise` already has the switch; it needs a noise corpus and an
      `add_noise_at_snr` variant that mixes a sampled clip instead of white noise.
      This is the blocker on the thesis headline figure being honest.
- [ ] Train-time background-noise mixing as an augmentation, using the same corpus but a
      disjoint split of it (never evaluate on noise the model trained against).
- [ ] Negative class: silence/unknown-word clips, so FAR (false accept) and FRR can be
      reported separately rather than only accuracy. Needs a 7th label and a confidence
      threshold sweep — `evaluation.predict` already returns max-softmax confidence.
- [ ] Optional: SpecAugment-style time/frequency masking, applied after featurization.

## Phase 3 — Mobile export & optimization & small upgrades

- [ ] `torch.onnx.export` to ONNX as the intermediate format.
- [ ] Quantize: dynamic PTQ first, then static INT8 with a calibration subset of the
      train split; verify accuracy drop < ~1%.
- [ ] Budget: model <= 1 MB on disk, < 10 ms per 1 s window on a mid-range phone CPU.
- [ ] Pick a runtime (TFLite / ExecuTorch / ONNX Runtime Mobile) by deployment target,
      not by benchmark alone.
- [ ] Parity harness: same input tensor through PyTorch and the exported model, assert
      max abs diff < 1e-3.

## Phase 4 — On-device inference loop

- [ ] Streaming capture: 16 kHz mono, 1 s sliding window, hop 250–500 ms.
- [ ] On-device preprocessing must mirror `src/data/features.py` exactly. Export the
      feature config alongside the model — `FeatureConfig.fingerprint()` exists to make
      a mismatch detectable.
- [ ] **Normalization decision.** Training uses per-file z-scoring. A streaming window is
      not a file, so either (a) keep per-window z-scoring and accept the mismatch on
      partial words, or (b) switch training to dataset-level mean/std
      (`features.normalize` already has the hook) and ship those constants. (b) is the
      safer path; it requires a retrain and a re-measure.
- [ ] Decision policy: k consecutive positive windows + confidence threshold +
      refractory period, so one utterance can't double-fire.
- [ ] Ship `labels.json` with the app.

## Phase 5 — Real-world validation

- [ ] Record a held-out test set: our voices, phone mic, real rooms, real noise. Never
      trained on; reported separately from the Kaggle test split.
- [ ] End-to-end latency (audio to decision) and battery cost for continuous listening;
      consider a VAD gate so the model only runs on speech energy.
- [ ] Write up: architecture, robustness curves, on-device numbers.

## Open decisions

- Wake-word-first ("hey ..." then command) vs always-on direct detection — changes the
  FAR requirement and the Phase 4 policy.
- Android only first, or iOS too (CoreML conversion path)?
