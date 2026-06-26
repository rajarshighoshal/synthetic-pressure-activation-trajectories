"""Shared figure style for the public artifact.

Matches the ICLR paper figures (serif, small fonts, no top/right spines, color-blind
palette, 300-dpi, embedded TrueType). Import `apply()` before plotting and `save()` to
write each figure as `.svg` (for the README) plus `.pdf` (paper-ready).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RCPARAMS = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Color-blind-friendly palette (same hexes as the ICLR paper figures).
CB = {
    "blue": "#2166ac",
    "red": "#d6604d",
    "green": "#4dac26",
    "orange": "#f4a582",
    "purple": "#762a83",
    "gray": "#878787",
}


def apply() -> None:
    matplotlib.rcParams.update(RCPARAMS)


def save(fig, path, *, formats: tuple[str, ...] = ("svg", "png", "pdf")) -> list[Path]:
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return written
