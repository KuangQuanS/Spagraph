#!/usr/bin/env python3
"""Create the compact Stage 3 ablation rank heatmap used in Figure 3d."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "evaluate" / "data" / "GSE144236",
        help="Directory containing the full and ablation run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/ablation_figures).",
    )
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def load_ranked(path: Path, score_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values(score_column, ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "ablation_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_column = "avg_attention_score"
    variants = {
        "Full": load_ranked(
            data_dir / "ccc_analysis" / "_observed_run" / "cellcom" / "lr_pair_statistics.csv",
            score_column,
        ),
        "Edge only": load_ranked(data_dir / "ablation_edge_masking_only" / "lr_pair_statistics.csv", score_column),
        "Node only": load_ranked(data_dir / "ablation_node_masking_only" / "lr_pair_statistics.csv", score_column),
    }

    reference_pairs = variants["Full"].head(args.top_k)["lr_pair"].tolist()
    rank_matrix = pd.DataFrame(index=variants, columns=reference_pairs, dtype=float)
    for name, frame in variants.items():
        lookup = frame.set_index("lr_pair")["rank"]
        for pair in reference_pairs:
            if pair in lookup.index and lookup[pair] <= args.top_k:
                rank_matrix.loc[name, pair] = lookup[pair]

    n_rows, n_cols = rank_matrix.shape
    fig, ax = plt.subplots(figsize=(166.071 / 25.4, 107.258 / 25.4))
    cmap = plt.cm.Blues_r
    norm = mcolors.Normalize(vmin=1, vmax=args.top_k)
    for row_index in range(n_rows):
        for column_index in range(n_cols):
            value = rank_matrix.iloc[row_index, column_index]
            color = "#E0E0E0" if np.isnan(value) else cmap(norm(value))
            y = n_rows - 1 - row_index
            ax.add_patch(
                plt.Rectangle((column_index, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2)
            )
            if not np.isnan(value):
                ax.text(
                    column_index + 0.5,
                    y + 0.5,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="white" if value <= max(1, args.top_k * 0.4) else "black",
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([pair.replace("_", "-") for pair in reference_pairs], fontsize=7, rotation=90)
    for label, pair in zip(ax.get_xticklabels(), reference_pairs):
        if pair in {"TNC_SDC1", "TNXB_SDC1", "THBS1_CD47"}:
            label.set_fontweight("bold")
            label.set_color("#B22222")
    ax.set_yticks(np.arange(n_rows)[::-1] + 0.5)
    ax.set_yticklabels(rank_matrix.index, fontsize=9, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar_axis = fig.add_axes([0.62, 0.91, 0.25, 0.025])
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    colorbar = plt.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Rank", fontsize=8, labelpad=2)
    colorbar.set_ticks([1, 5, 10, args.top_k])
    colorbar.ax.tick_params(labelsize=7)
    plt.subplots_adjust(top=0.85, bottom=0.28, left=0.12, right=0.98)

    for extension, dpi in (("png", 300), ("pdf", None)):
        kwargs = {"dpi": dpi} if dpi else {}
        fig.savefig(output_dir / f"ablation_rank_heatmap.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Saved: {output_dir / 'ablation_rank_heatmap.png'}")


if __name__ == "__main__":
    main()
