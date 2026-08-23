"""Shared chart style. Every figure in the project is drawn through this module so
reports look like one system rather than a pile of default-matplotlib plots.

Colors are the validated categorical palette: hues are assigned by slot in a fixed
order and never cycled, so a series keeps its color no matter how many other series
are on the plot. Three slots sit below 3:1 against the surface, so every chart here
also ships direct value labels — and `report.py` renders a table view beside each
figure — which is what makes those slots legal.
"""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e6e5e1"

# Fixed categorical order — index by slot, never by rank and never cycled.
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Single-hue light->dark ramp for magnitude (confusion matrix, heatmaps).
SEQUENTIAL = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
    "#1c5cab", "#184f95", "#104281", "#0d366b",
]

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": TEXT_SECONDARY,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "font.size": 10,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
}


def sequential_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list("nvcr_blue", SEQUENTIAL)


@contextmanager
def styled():
    with plt.rc_context(RC):
        yield


def label_ends(ax, entries: list[tuple[float, float, str, str]], min_gap: float = 0.04) -> None:
    """Direct labels at the ends of lines — identity without hunting the legend, and the
    relief the low-contrast slots require.

    Lines that finish close together would stack their labels on top of each other, so
    entries are nudged apart vertically by at least `min_gap` of the axis height. The
    nudge is display-only: the marks themselves stay where the data puts them.
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    placed: list[float] = []
    for x, y, text, color in sorted(entries, key=lambda e: e[1]):
        target = y
        for other in placed:
            if abs(target - other) < min_gap * span:
                target = other + min_gap * span
        placed.append(target)
        ax.annotate(
            text, xy=(x, y), xytext=(7, (target - y) / span * ax.bbox.height),
            textcoords="offset points", color=color, fontsize=9, fontweight="600",
            va="center", clip_on=False,
        )
