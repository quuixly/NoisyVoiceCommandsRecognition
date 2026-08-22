from pathlib import Path
from typing import Iterable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.constants import COMMANDS_TO_PROCESS, MFCC_DIR, TOP_DB


def _load_and_trim(
    file: Path, sr: int = 16_000, top_db: float = TOP_DB
) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(file, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y, sr


def _compute_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y, sr=sr, n_fft=512, hop_length=160, n_mels=40, fmin=20, fmax=sr // 2
    )


def _plot_mfcc_on(ax, mfccs: np.ndarray, title: str):
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficients")
    return img


def generate_mfcc(
    start_dir: Path,
    mfcc_dir: Path = MFCC_DIR,
    commands: Iterable[str] = COMMANDS_TO_PROCESS,
    top_db: float = TOP_DB,
    limit: int = 10,
) -> None:
    mfcc_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        command_dir = start_dir / command
        figures_dir = mfcc_dir / start_dir / command  # TODO : REMOVE START DIR, only for comparison
        figures_dir.mkdir(parents=True, exist_ok=True)
        for number, file in enumerate(sorted(command_dir.iterdir())[:limit]):
            if not file.is_file():
                continue

            y, sr = _load_and_trim(file, top_db=top_db)
            mfccs = _compute_mfcc(y, sr)

            fig, ax = plt.subplots(figsize=(10, 5))
            img = _plot_mfcc_on(ax, mfccs, f"MFCC - {file.stem}")
            fig.colorbar(img, ax=ax)
            fig.savefig(figures_dir / f"{command}_{number}.png")
            plt.close(fig)


def plot_comparison(
    raw_file: Path,
    augmented_file: Path,
    *,
    top_db: float = TOP_DB,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """Show waveform + MFCC for a raw file next to its augmented counterpart."""
    y_raw, sr_raw = _load_and_trim(raw_file, top_db=top_db)
    y_aug, sr_aug = _load_and_trim(augmented_file, top_db=top_db)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    librosa.display.waveshow(y_raw, sr=sr_raw, ax=axes[0, 0])
    axes[0, 0].set_title(f"Waveform (raw) - {raw_file.stem}")
    librosa.display.waveshow(y_aug, sr=sr_aug, ax=axes[0, 1])
    axes[0, 1].set_title(f"Waveform (augmented) - {augmented_file.stem}")

    mfcc_raw = _compute_mfcc(y_raw, sr_raw)
    mfcc_aug = _compute_mfcc(y_aug, sr_aug)
    img_raw = _plot_mfcc_on(axes[1, 0], mfcc_raw, "MFCC (raw)")
    img_aug = _plot_mfcc_on(axes[1, 1], mfcc_aug, "MFCC (augmented)")
    fig.colorbar(img_raw, ax=axes[1, 0])
    fig.colorbar(img_aug, ax=axes[1, 1])
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
