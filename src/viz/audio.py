"""Audio-side figures — all of them drawn through the same `Featurizer` the model
uses, so a spectrogram in a report is literally the array the network consumed."""

from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.data.augment import RandomAugmenter
from src.data.features import Featurizer
from src.viz.charts import _save
from src.viz.theme import SERIES, sequential_cmap, styled


def _show_features(ax, features: np.ndarray, title: str, feature_type: str) -> None:
    """Channel 0 of the extracted tensor: MFCC coefficients, or the log-mel bands."""
    ax.imshow(features[0], origin="lower", aspect="auto", cmap=sequential_cmap(), interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("MFCC coefficient" if feature_type == "mfcc" else "mel band")
    ax.grid(False)


def plot_augmentation_preview(
    config: Config, clips: list[tuple[str, np.ndarray]], out_path: Path, n_variants: int = 2
) -> Path:
    """One row per clip: the clean waveform and its features, then `n_variants`
    augmented versions. This is the check that augmentation is distorting the audio
    without destroying the word — a chain that flattens the features means the
    transform ranges in `augment.transforms` are too aggressive.
    """
    featurizer = Featurizer(config.features)
    augmenter = RandomAugmenter(config.augment, seed=config.seed)
    sample_rate = config.data.sample_rate
    columns = 1 + n_variants

    with styled():
        fig, axes = plt.subplots(
            2 * len(clips), columns, figsize=(4.2 * columns, 4.4 * len(clips)), squeeze=False
        )
        for row, (label, y) in enumerate(clips):
            variants = [("clean", y)] + [
                (f"augmented {i + 1}", augmenter(y, sample_rate)) for i in range(n_variants)
            ]
            for col, (name, wave) in enumerate(variants):
                ax_wave = axes[2 * row][col]
                librosa.display.waveshow(wave, sr=sample_rate, ax=ax_wave, color=SERIES[0 if col == 0 else 1])
                ax_wave.set_title(f"{label} — {name}", fontsize=10)
                ax_wave.set_ylim(-1, 1)
                ax_wave.grid(axis="x", visible=False)
                _show_features(axes[2 * row + 1][col], featurizer(wave, sample_rate), "", config.features.type)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_examples(
    config: Config, examples: list[tuple[str, np.ndarray]], out_path: Path, title: str, columns: int = 4
) -> Path:
    """A grid of feature images with captions — used for the worst misclassifications,
    where seeing the input explains a confusion that a number never will."""
    featurizer = Featurizer(config.features)
    if not examples:
        raise ValueError("plot_examples needs at least one example")
    rows = (len(examples) + columns - 1) // columns

    with styled():
        fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 2.9 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, (caption, wave) in enumerate(examples):
            ax = axes[i // columns][i % columns]
            ax.axis("on")
            _show_features(ax, featurizer(wave, config.data.sample_rate), caption, config.features.type)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="600")
        fig.tight_layout()
        return _save(fig, out_path)
