import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_audit(base: Path, output: Path) -> None:
    """Adapt the original two-panel layout to five-run sensitivity analysis."""
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 10})
    loo = pd.read_csv(base / 'leave_one_run_out.csv').sort_values('omitted_seed')
    scores = pd.read_csv(base / 'raw_scores.csv', index_col=0)
    consensus = pd.read_csv(base / 'aggregate.csv').sort_values(['rank', 'lr_pair'])
    selected = consensus.head(12).lr_pair
    values = scores.loc[selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    labels = loo.omitted_seed.astype(str)
    axes[0].plot(labels, loo.spearman, marker='o', linewidth=2, label='Rank correlation')
    axes[0].plot(labels, loo.top10_overlap / loo.top10_union, marker='s', linewidth=2, label='Top 10 Jaccard')
    axes[0].set(ylim=(0, 1.05), xlabel='Omitted training seed', ylabel='Agreement with five-run consensus')
    axes[0].set_title('Leave-one-run-out sensitivity')
    axes[0].grid(axis='y', alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    y = range(len(values))
    axes[1].errorbar(values.mean(axis=1), y, xerr=values.std(axis=1, ddof=1), fmt='o',
                     color='#1f77b4', ecolor='#9ecae1', elinewidth=2, capsize=3)
    axes[1].set_yticks(list(y), values.index.str.replace('_', '-', regex=False))
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean associated-edge attention ± SD')
    axes[1].set_title('Top 12 by consensus rank')
    axes[1].grid(axis='x', alpha=.25)
    for ax, label in zip(axes, ['d', 'e']):
        ax.text(-.12, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight='bold')
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'figs1de.pdf', bbox_inches='tight')
    fig.savefig(output / 'figs1de.svg', bbox_inches='tight')
    fig.savefig(output / 'figs1de.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        required=True,
        help="Directory containing seed_topk_overlap.csv and seed_rank_stability_summary.csv.",
    )
    parser.add_argument('--audit', action='store_true', help='Use verified five-run audit tables.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    base = Path(args.base)
    if args.audit:
        draw_audit(base, args.output_dir or base / 'figures')
        return
    out = base / "scc_seed_stability_summary.png"

    overlap = pd.read_csv(base / "seed_topk_overlap.csv")
    stability = pd.read_csv(base / "seed_rank_stability_summary.csv").head(12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    ax = axes[0]
    for k, frame in overlap.groupby("k"):
        labels = frame["run_a"] + " vs " + frame["run_b"]
        ax.plot(labels, frame["jaccard"], marker="o", linewidth=2, label=f"Top {k}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Top-k LR ranking overlap")
    ax.text(-0.12, 1.06, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    y = range(len(stability))
    ax.errorbar(
        stability["mean_attention"],
        y,
        xerr=stability["attention_sd"],
        fmt="o",
        color="#1f77b4",
        ecolor="#9ecae1",
        elinewidth=2,
        capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(stability["lr_pair"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention score +/- SD")
    ax.set_title("Stable top SCC LR pairs")
    ax.text(-0.28, 1.06, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
