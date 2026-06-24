#!/usr/bin/env python3
"""Plot the 20-seed full-model synthetic LR benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "synthetic_lr_v2_context_20seeds"
OUTPUT = ROOT / "results" / "synthetic_lr_v2_context_20seeds_figure"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#777777"
MAGENTA = "#CC79A7"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "synthetic_v2_multiseed_metrics.csv")
    paired = pd.read_csv(RESULTS / "synthetic_v2_multiseed_paired.csv")
    rankings = pd.read_csv(RESULTS / "synthetic_v2_multiseed_rankings.csv")
    pooled = pd.read_csv(RESULTS / "synthetic_v2_multiseed_pooled.csv").iloc[0]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    ax_a, ax_b, ax_c = axes

    positions = np.arange(len(metrics))
    ax_a.plot(
        positions,
        metrics["candidate_attention_auprc"],
        "o-",
        color=BLUE,
        label="Attention",
        markersize=3,
    )
    ax_a.plot(
        positions,
        metrics["candidate_abundance_auprc"],
        "^:",
        color=GRAY,
        label="Abundance",
        markersize=3,
    )
    ax_a.set_xticks(positions[::3], metrics["seed"].astype(str).iloc[::3])
    ax_a.set_xlabel("Random seed")
    ax_a.set_ylabel("Candidate-set AUPRC")
    ax_a.set_title("Attention outperforms abundance")
    ax_a.legend(frameon=False)
    auprc_difference = (
        metrics["candidate_attention_auprc"]
        - metrics["candidate_abundance_auprc"]
    )
    auprc_p = stats.wilcoxon(
        auprc_difference, alternative="greater"
    ).pvalue
    ax_a.text(
        0.03,
        0.04,
        f"20/20 seeds; P = {auprc_p:.2g}",
        transform=ax_a.transAxes,
        va="bottom",
    )

    differences = paired["local_minus_diffuse"].to_numpy()
    rng = np.random.default_rng(7)
    ax_b.scatter(
        rng.normal(0, 0.055, len(differences)),
        differences,
        s=9,
        alpha=0.45,
        color=ORANGE,
        linewidths=0,
    )
    median = float(np.median(differences))
    ax_b.plot([-0.16, 0.16], [median, median], color="black", linewidth=1.5)
    ax_b.axhline(0, color=GRAY, linestyle="--", linewidth=0.8)
    ax_b.set_xlim(-0.35, 0.35)
    ax_b.set_xticks([0], ["Local - diffuse"])
    ax_b.set_ylabel("Attention-score difference")
    ax_b.set_title("Matched spatial programs")
    ax_b.text(
        0.03,
        0.97,
        (
            f"Wins: {int((differences > 0).sum())}/{len(differences)}\n"
            f"Median: {median:.3f}\n"
            f"P = {pooled.local_vs_diffuse_wilcoxon_p:.3g}"
        ),
        transform=ax_b.transAxes,
        va="top",
    )

    global_rows = rankings.loc[
        rankings["pattern"].eq("global_high_coverage")
    ]
    boxes = ax_c.boxplot(
        [
            global_rows["abundance_rank"],
            global_rows["attention_rank"],
        ],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], [GRAY, MAGENTA]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax_c.set_xticks([1, 2], ["Abundance", "Attention"])
    ax_c.set_ylabel("Rank (1 = highest)")
    ax_c.invert_yaxis()
    ax_c.set_title("Global high-coverage controls")

    for label, ax in zip("abc", axes):
        ax.text(
            -0.16,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "synthetic_lr_v2_20seeds.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT / "synthetic_lr_v2_20seeds.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
