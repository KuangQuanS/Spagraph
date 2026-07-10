#!/usr/bin/env python3
"""Plot the fully synthetic LR benchmark and its LR-identity ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results" / "synthetic_lr_v2_context_multiseed"
NO_ID = ROOT / "results" / "synthetic_lr_v2_context_noid_multiseed"
OUTPUT = ROOT / "results" / "synthetic_lr_v2_summary"

COLORS = {
    "Full model": "#0072B2",
    "No LR identity": "#009E73",
    "Abundance baseline": "#999999",
    "Local positive": "#D55E00",
    "Matched diffuse": "#56B4E9",
    "Global high coverage": "#CC79A7",
    "Random background": "#7F7F7F",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_metrics = pd.read_csv(FULL / "synthetic_v2_multiseed_metrics.csv")
    no_id_metrics = pd.read_csv(NO_ID / "synthetic_v2_multiseed_metrics.csv")
    full_paired = pd.read_csv(FULL / "synthetic_v2_multiseed_paired.csv")
    no_id_paired = pd.read_csv(NO_ID / "synthetic_v2_multiseed_paired.csv")
    rankings = pd.read_csv(FULL / "synthetic_v2_multiseed_rankings.csv")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    seeds = full_metrics["seed"].to_numpy()
    ax_a.plot(
        seeds,
        full_metrics["candidate_attention_auprc"],
        marker="o",
        color=COLORS["Full model"],
        label="Full model",
    )
    ax_a.plot(
        seeds,
        no_id_metrics["candidate_attention_auprc"],
        marker="s",
        linestyle="--",
        color=COLORS["No LR identity"],
        label="No LR identity",
    )
    ax_a.plot(
        seeds,
        full_metrics["candidate_abundance_auprc"],
        marker="^",
        linestyle=":",
        color=COLORS["Abundance baseline"],
        label="Abundance baseline",
    )
    ax_a.set_xlabel("Random seed")
    ax_a.set_ylabel("Candidate-set AUPRC")
    ax_a.set_ylim(0, 0.55)
    ax_a.set_title("Ranking recovery varies across seeds")
    ax_a.legend(frameon=False, fontsize=7)

    rng = np.random.default_rng(7)
    for x, (label, frame, color) in enumerate([
        ("Full model", full_paired, COLORS["Full model"]),
        ("No LR identity", no_id_paired, COLORS["No LR identity"]),
    ]):
        values = frame["local_minus_diffuse"].to_numpy()
        jitter = rng.normal(x, 0.055, len(values))
        ax_b.scatter(jitter, values, s=12, alpha=0.55, color=color, linewidths=0)
        ax_b.plot(
            [x - 0.16, x + 0.16],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=1.5,
        )
    ax_b.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax_b.set_xticks([0, 1], ["Full model", "No LR identity"])
    ax_b.set_ylabel("Local minus diffuse attention")
    ax_b.set_title("Matched spatial comparison (40 families)")
    ax_b.text(
        0.02,
        0.98,
        "Full: 25/40 wins, p=0.071\nNo ID: 25/40 wins, p=0.073",
        transform=ax_b.transAxes,
        va="top",
        fontsize=7,
    )

    pattern_order = [
        "local_positive",
        "matched_diffuse",
        "global_high_coverage",
        "random_background",
    ]
    pattern_labels = [
        "Local\npositive",
        "Matched\ndiffuse",
        "Global high\ncoverage",
        "Random\nbackground",
    ]
    for x, pattern in enumerate(pattern_order):
        values = rankings.loc[
            rankings["pattern"].eq(pattern), "attention_rank"
        ].to_numpy()
        jitter = rng.normal(x, 0.06, len(values))
        color = COLORS[dict(zip(pattern_order, [
            "Local positive",
            "Matched diffuse",
            "Global high coverage",
            "Random background",
        ]))[pattern]]
        ax_c.scatter(jitter, values, s=10, alpha=0.35, color=color, linewidths=0)
        ax_c.boxplot(
            values,
            positions=[x],
            widths=0.42,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": color},
            medianprops={"color": "black"},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
    ax_c.set_xticks(range(len(pattern_labels)), pattern_labels)
    ax_c.set_ylabel("Attention rank (1 = highest)")
    ax_c.invert_yaxis()
    ax_c.set_title("Attention ranks overlap across patterns")

    global_rows = rankings.loc[
        rankings["pattern"].eq("global_high_coverage")
    ]
    local_rows = rankings.loc[rankings["pattern"].eq("local_positive")]
    groups = [
        local_rows["abundance_rank"],
        local_rows["attention_rank"],
        global_rows["abundance_rank"],
        global_rows["attention_rank"],
    ]
    labels = [
        "Local\nabundance",
        "Local\nattention",
        "Global\nabundance",
        "Global\nattention",
    ]
    colors = [
        COLORS["Local positive"],
        COLORS["Local positive"],
        COLORS["Global high coverage"],
        COLORS["Global high coverage"],
    ]
    boxes = ax_d.boxplot(groups, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax_d.set_xticks(range(1, 5), labels)
    ax_d.set_ylabel("Rank (1 = highest)")
    ax_d.invert_yaxis()
    ax_d.set_title("Attention suppresses abundance dominance")

    for label, ax in zip("abcd", axes.ravel()):
        ax.text(
            -0.13,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT / "synthetic_lr_v2_summary.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT / "synthetic_lr_v2_summary.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
