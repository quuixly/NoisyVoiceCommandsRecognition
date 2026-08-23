"""Every chart the project produces. Each function saves one PNG and returns its path."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.viz.theme import (
    GRID,
    SERIES,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    label_ends,
    sequential_cmap,
    styled,
)


def read_history(path: Path) -> dict[str, list[float]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no epochs yet")
    return {key: [float(r[key]) for r in rows] for key in rows[0]}


def plot_history(history_path: Path, out_path: Path) -> Path:
    """Loss and accuracy against epoch, side by side.

    Two panels rather than two y-axes on one: loss and accuracy have unrelated
    scales, and overlaying them on twin axes is the single most misleading thing a
    training plot can do — the crossover point would be an artifact of the scaling.
    """
    h = read_history(history_path)
    epochs = h["epoch"]

    with styled():
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

        loss_labels = []
        for slot, (key, name) in enumerate([("train_loss", "train"), ("val_loss", "val")]):
            ax_loss.plot(epochs, h[key], color=SERIES[slot], label=name)
            loss_labels.append((epochs[-1], h[key][-1], name, SERIES[slot]))
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("epoch")

        acc_labels = []
        for slot, (key, name) in enumerate([("train_acc", "train"), ("val_acc", "val")]):
            if key not in h:
                continue
            ax_acc.plot(epochs, h[key], color=SERIES[slot], label=name)
            acc_labels.append((epochs[-1], h[key][-1], name, SERIES[slot]))
        best = int(np.argmax(h["val_acc"]))
        ax_acc.scatter([epochs[best]], [h["val_acc"][best]], s=70, color=SERIES[1],
                       edgecolor=SURFACE, linewidth=2, zorder=5)
        ax_acc.annotate(
            f"best {h['val_acc'][best]:.3f} @ epoch {int(epochs[best])}",
            xy=(epochs[best], h["val_acc"][best]), xytext=(-10, -16),
            textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=9,
        )
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylim(0, 1.02)

        for ax in (ax_loss, ax_acc):
            ax.grid(axis="x", visible=False)
            ax.margins(x=0.12)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
        label_ends(ax_loss, loss_labels)
        label_ends(ax_acc, acc_labels)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> Path:
    """Row-normalized confusion heatmap; raw counts printed in each cell.

    Normalized by row because class supports differ — an unnormalized matrix makes
    the largest class look like the best-recognized one. One sequential hue,
    light to dark, since the encoded quantity is magnitude.
    """
    totals = cm.sum(axis=1, keepdims=True)
    rates = np.divide(cm, np.maximum(totals, 1))

    with styled():
        size = max(4.5, 0.85 * len(labels) + 2.0)
        fig, ax = plt.subplots(figsize=(size, size * 0.88))
        im = ax.imshow(rates, cmap=sequential_cmap(), vmin=0, vmax=1)

        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title("Confusion matrix (row-normalized)")
        ax.grid(False)
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells

        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] == 0:
                    continue
                ax.text(
                    j, i, f"{rates[i, j]:.2f}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=8,
                    color=SURFACE if rates[i, j] > 0.55 else TEXT_PRIMARY,
                )
        bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        # ty: ignore[call-non-callable] — matplotlib's stub for Colorbar.outline
        # (_ColorbarSpine) does not expose the Artist methods it inherits at runtime,
        # so every call on it is flagged. Verified working: get_visible() is False after.
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, color=GRID)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_per_class(per_class: dict[str, dict[str, float]], out_path: Path) -> Path:
    """Grouped horizontal bars, one group per command, value printed on every bar."""
    names = list(per_class)
    metrics = ["precision", "recall", "f1"]
    y = np.arange(len(names))
    height = 0.26

    with styled():
        fig, ax = plt.subplots(figsize=(8, 0.72 * len(names) + 1.8))
        for slot, metric in enumerate(metrics):
            values = [per_class[n][metric] for n in names]
            offset = (slot - 1) * (height + 0.02)  # 2px-equivalent gap between bars
            ax.barh(y + offset, values, height=height, color=SERIES[slot], label=metric)
            for yi, v in zip(y + offset, values, strict=True):
                ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=8, color=TEXT_SECONDARY)

        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("score")
        ax.set_title("Per-class precision / recall / F1")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", ncols=3)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_snr_sweep(points: list[tuple[float | None, float]], out_path: Path) -> Path:
    """Accuracy against added-noise SNR — the noise-robustness headline metric.

    `None` on the x side is the clean test set; it is drawn as a reference line
    rather than a point, because "no noise" has no position on a dB axis.
    """
    clean = [acc for snr, acc in points if snr is None]
    noisy = sorted([(float(snr), acc) for snr, acc in points if snr is not None])

    with styled():
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        if clean:
            ax.axhline(clean[0], color=TEXT_MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
            ax.annotate(f"clean {clean[0]:.3f}", xy=(0.005, clean[0]), xycoords=("axes fraction", "data"),
                        xytext=(0, 6), textcoords="offset points", fontsize=9, color=TEXT_SECONDARY)
        if noisy:
            xs = [p[0] for p in noisy]
            ys = [p[1] for p in noisy]
            ax.plot(xs, ys, color=SERIES[0], marker="o", markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            # Tick only where a measurement exists: interpolated tick values on a
            # sweep of 4-5 discrete SNRs invent precision the sweep doesn't have.
            ax.set_xticks(xs, [f"{x:g}" for x in xs])
            ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
            for x, v in zip(xs, ys, strict=True):
                ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel("added-noise SNR (dB)  — lower is noisier")
        ax.set_ylabel("test accuracy")
        ax.set_title("Accuracy vs SNR")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return _save(fig, out_path)


def plot_run_comparison(runs: dict[str, Path], out_path: Path) -> Path:
    """Validation accuracy per epoch for several runs on one axis.

    Colors are assigned by sorted run name, so a run keeps its color whether or not
    the other runs are in the plot.
    """
    with styled():
        fig, ax = plt.subplots(figsize=(9, 4.6))
        labels: list[tuple[float, float, str, str]] = []
        for slot, name in enumerate(sorted(runs)):
            history = read_history(runs[name])
            color = SERIES[slot % len(SERIES)]
            ax.plot(history["epoch"], history["val_acc"], color=color, label=name)
            labels.append((history["epoch"][-1], history["val_acc"][-1], name, color))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.set_title("Run comparison")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.18)
        label_ends(ax, labels)
        fig.tight_layout()
        return _save(fig, out_path)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_snr_comparison(runs: dict[str, list[tuple[float | None, float]]], out_path: Path) -> Path:
    """One accuracy-vs-SNR curve per run, on one axis.

    This is the comparison that actually settles an architecture or front-end choice
    for this thesis: two models can tie on clean accuracy and come apart badly once
    noise is added, and only the slope of these curves shows it. Clean accuracy is
    drawn as a marker off the left edge rather than folded into the line, because
    "no noise" has no position on a dB axis.
    """
    with styled():
        fig, ax = plt.subplots(figsize=(9, 4.8))
        labels: list[tuple[float, float, str, str]] = []
        clean_x = None

        for slot, name in enumerate(sorted(runs)):
            color = SERIES[slot % len(SERIES)]
            noisy = sorted([(float(s), a) for s, a in runs[name] if s is not None])
            clean = [a for s, a in runs[name] if s is None]
            if not noisy:
                continue
            xs = [p[0] for p in noisy]
            ys = [p[1] for p in noisy]
            if clean_x is None:
                clean_x = min(xs) - (max(xs) - min(xs) or 10) * 0.22
            ax.plot(xs, ys, color=color, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            if clean:
                ax.plot([clean_x], clean, color=color, marker="D", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2)
            labels.append((xs[-1], ys[-1], name, color))

        all_snr = sorted({float(s) for points in runs.values() for s, _ in points if s is not None})
        ticks = ([clean_x] if clean_x is not None else []) + all_snr
        tick_labels = (["clean"] if clean_x is not None else []) + [f"{x:g}" for x in all_snr]
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlabel("added-noise SNR (dB)  — lower is noisier")
        ax.set_ylabel("test accuracy")
        ax.set_title("Noise robustness by run")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.16)
        label_ends(ax, labels)
        fig.tight_layout()
        return _save(fig, out_path)
