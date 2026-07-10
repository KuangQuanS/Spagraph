#!/usr/bin/env python3
"""Recalculate and render Figure 3e with LR pairs as statistical units."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spagraph.analysis.figure3e_statistics import compare_lr_pair_groups


METRICS = {
    "edge_spatial_focality": "Edge spatial focality",
    "celltype_pair_count": "Cell-type-pair count",
}
COLORS = ("#C44E52", "#4C72B0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=330)
    return parser.parse_args()


def p_text(value: float) -> str:
    return f"Holm P = {value:.2e}" if value < 0.001 else f"Holm P = {value:.3f}"


def render_panel(disjoint: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.9))
    summary_by_metric = summary.set_index("metric")
    rng = np.random.default_rng(42)

    for axis, (metric, label) in zip(axes, METRICS.items()):
        attention = disjoint.loc[disjoint["ranking_type"] == "attention", metric].astype(float).to_numpy()
        frequency = disjoint.loc[disjoint["ranking_type"] == "frequency", metric].astype(float).to_numpy()
        values = [attention, frequency]
        box = axis.boxplot(
            values,
            tick_labels=["Attention-only\nLR pairs", "Frequency-only\nLR pairs"],
            patch_artist=True,
            widths=0.52,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            boxprops={"edgecolor": "#555555", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
        for position, (series, color) in enumerate(zip(values, COLORS), start=1):
            jitter = rng.normal(0.0, 0.045, size=len(series))
            axis.scatter(
                np.full(len(series), position) + jitter,
                series,
                s=22,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
        row = summary_by_metric.loc[metric]
        axis.set_title(label, fontweight="bold", pad=8)
        axis.set_ylabel(label)
        axis.text(
            0.5,
            0.97,
            p_text(float(row["holm_p"])),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0"},
        )
        axis.text(
            0.5,
            -0.22,
            f"n={len(attention)}                 n={len(frequency)}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )
        axis.grid(axis="y", color="#EAEAEA", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_dir / "figure3e_lr_pair_statistics.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "figure3e_lr_pair_statistics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_metrics = pd.read_csv(args.pair_metrics)
    summary, disjoint, overlap = compare_lr_pair_groups(pair_metrics, METRICS)
    summary.to_csv(output_dir / "figure3e_statistics.csv", index=False)
    disjoint.to_csv(output_dir / "figure3e_lr_pair_units.csv", index=False)
    pd.DataFrame(
        {"lr_pair": overlap, "exclusion_reason": "selected by both rankings"}
    ).to_csv(output_dir / "figure3e_excluded_overlap.csv", index=False)
    render_panel(disjoint, summary, output_dir, args.dpi)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
