"""Generate the README headline figures from final_summary.json.

Figures use the shared paper style (experiments/paper_style.py): serif, color-blind
palette, no top/right spines, 300-dpi, written as .svg (for the README) + .pdf. Numbers
come only from the committed summary, never hard-coded here (except the dataset-filter
counts, which are the filter design, not a result).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps

JUDGE_LABEL = {"opus": "Opus 4.8", "deepseek": "DeepSeek v4 Pro"}
JUDGE_COLOR = {"opus": ps.CB["blue"], "deepseek": ps.CB["red"]}


def dataset_filter_figure(out: Path) -> None:
    stages = [
        ("Generated\npairs", 1000),
        ("Passed knowledge\ncheck", 114),
        ("Retained\nconversations", 228),
        ("Labeled\nturns", 1824),
    ]
    colors = [ps.CB["gray"], ps.CB["blue"], ps.CB["green"], ps.CB["purple"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    xs = np.arange(len(stages))
    vals = [v for _, v in stages]
    ax.bar(xs, vals, color=colors, width=0.66, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 35, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _ in stages])
    ax.set_ylabel("count")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Dataset filter: keep scenarios Llama first marks false", loc="left")
    ax.grid(axis="x", visible=False)
    ps.save(fig, out)


def per_turn_figure(summary: dict, out: Path) -> None:
    probes = ["mlp", "linear", "pca50", "tangent_subspace", "graph_geodesic",
              "mahalanobis", "class_mahalanobis", "centroid"]
    values = {
        judge: {row["probe"]: row["best_auroc"] for row in block["best_rows"]}
        for judge, block in summary["per_turn"].items()
    }
    probes = [p for p in probes if all(p in values[j] for j in values)]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    xs = np.arange(len(probes))
    width = 0.38
    for k, judge in enumerate(("opus", "deepseek")):
        offs = (k - 0.5) * width
        vals = [values[judge][p] for p in probes]
        ax.bar(xs + offs, vals, width=width, color=JUDGE_COLOR[judge],
               label=JUDGE_LABEL[judge], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("_", " ") for p in probes], rotation=28, ha="right")
    ax.set_ylabel("best AUROC")
    ax.set_ylim(0.5, 0.92)
    ax.set_title("Per-turn acceptance detection (pair-grouped CV)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ps.save(fig, out)


def headline_ci_figure(summary: dict, out: Path) -> None:
    rows = summary["confidence_intervals"]["rows"]
    selected = []
    for row in rows:
        if row["level"] == "per_turn":
            label = f'{JUDGE_LABEL.get(row["judge"], row["judge"])}: per-turn MLP'
        elif row["probe"] == "linear":
            label = f'{JUDGE_LABEL.get(row["judge"], row["judge"])}: traj. {row["feature"]}+linear'
        else:
            label = f'{JUDGE_LABEL.get(row["judge"], row["judge"])}: traj. {row["feature"]}+tangent'
        selected.append((label, row))
    selected.reverse()  # top row at top

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (label, row) in enumerate(selected):
        is_geom = row["probe"] == "tangent_subspace"
        color = ps.CB["purple"] if is_geom else ps.CB["blue"]
        lo, hi, point = row["ci95_low"], row["ci95_high"], row["auroc"]
        ax.plot([lo, hi], [i, i], color=color, lw=3, solid_capstyle="round", zorder=3)
        ax.plot([point], [i], "o", color=color, ms=6, zorder=4)
        ax.text(hi + 0.003, i, f"{point:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([lab for lab, _ in selected])
    ax.set_xlim(0.82, 0.99)
    ax.set_xlabel("AUROC (95% clustered bootstrap)")
    ax.set_title("Headline AUROC: Euclidean vs tangent-subspace", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=ps.CB["blue"], lw=3, label="Euclidean baseline"),
               plt.Line2D([], [], color=ps.CB["purple"], lw=3, label="tangent subspace")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    ps.save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/eval/synthetic_pressure_llama8b/final_summary.json")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    ps.apply()
    summary = json.loads(Path(args.summary).read_text())
    out_dir = Path(args.out_dir)
    dataset_filter_figure(out_dir / "dataset_filter.svg")
    per_turn_figure(summary, out_dir / "per_turn_auroc.svg")
    headline_ci_figure(summary, out_dir / "headline_ci.svg")
    for name in ("dataset_filter", "per_turn_auroc", "headline_ci"):
        print(f"wrote {out_dir / name}.svg (+ .pdf)")


if __name__ == "__main__":
    main()
