# TODO

## Propper mfcc

Analyze and move: 

```python
import numpy as np
import librosa
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
N_MFCC        = 40       # richer representation
MAX_PAD_LEN   = 128      # fixed time-axis length (frames)
PRE_EMPHASIS  = 0.97     # high-freq boost coefficient
TOP_DB        = 30       # silence trimming threshold


def extract_mfcc(
    file: Path,
    n_mfcc: int = N_MFCC,
    max_pad_len: int = MAX_PAD_LEN,
) -> np.ndarray:
    """
    Returns a (3 * n_mfcc, max_pad_len) array:
      - rows 0      ..   n_mfcc-1  → static MFCCs
      - rows n_mfcc .. 2*n_mfcc-1  → delta  (velocity)
      - rows 2*n_mfcc.. 3*n_mfcc-1 → delta2 (acceleration)
    """
    # 1. Load & resample to a fixed rate for consistency
    y, sr = librosa.load(file, sr=16_000, mono=True)

    # 2. Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # 3. Pre-emphasis — boosts high-freq speech components
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])

    # 4. Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=512, hop_length=160,
                                  n_mels=40, fmin=20, fmax=sr // 2)

    # 5. Delta and delta-delta features
    delta  = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # 6. Stack → shape (3*n_mfcc, T)
    combined = np.vstack([mfccs, delta, delta2])

    # 7. Per-feature normalisation (zero mean, unit variance)
    combined = (combined - combined.mean(axis=1, keepdims=True)) / \
               (combined.std(axis=1, keepdims=True) + 1e-8)

    # 8. Pad or truncate to fixed length
    if combined.shape[1] < max_pad_len:
        pad_width = max_pad_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode="constant")
    else:
        combined = combined[:, :max_pad_len]

    return combined  # shape: (3*n_mfcc, max_pad_len) e.g. (120, 128)


def generate_mfcc(start_dir: Path) -> None:
    MFCC_DIR.mkdir(parents=True, exist_ok=True)

    for command in COMMANDS_TO_PROCESS:
        logging.info(f"Processing {start_dir}/{command}")
        command_dir  = start_dir / command
        save_dir     = MFCC_DIR / command
        save_dir.mkdir(parents=True, exist_ok=True)

        for number, file in enumerate(command_dir.iterdir()):
            if not file.is_file():
                continue
            try:
                features = extract_mfcc(file)                    # (120, 128)
                np.save(save_dir / f"{command}_{number}.npy", features)
                logging.info(f"  Saved {command}_{number}.npy  shape={features.shape}")
            except Exception as e:
                logging.warning(f"  Skipped {file.name}: {e}")
```
