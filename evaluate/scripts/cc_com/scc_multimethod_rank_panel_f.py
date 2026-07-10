from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = EVALUATE_DIR / "data" / "GSE144236"
OUTPUT_DIR = DATA_DIR / "cellchat_baseline_figures"

CELLCHAT_SPAGRAPH_PATH = OUTPUT_DIR / "cellchat_spagraph_shared_pair_ranks.csv"
COMMOT_PATH = DATA_DIR / "commot_baseline_perm20_min5" / "commot_pair_summary_cross_group.csv"
GIOTTO_PATH = DATA_DIR / "giotto_baseline_iter20" / "giotto_pair_summary.csv"

PAIRS = [
    ("TNC_SDC1", "anchor"),
    ("TNXB_SDC1", "promoted"),
    ("THBS1_CD47", "promoted"),
    ("CD99_CD99", "deprioritized"),
    ("FN1_CD44", "deprioritized"),
    ("LGALS9_CD44", "deprioritized"),
]

PAIR_GROUPS = [
    ("Shared anchor", ["TNC_SDC1"]),
    ("Spagraph-promoted", ["TNXB_SDC1", "THBS1_CD47"]),
    ("Relative deprioritized", ["CD99_CD99", "FN1_CD44", "LGALS9_CD44"]),
]

ROLE_COLORS = {
    "anchor": "#4C72B0",
    "promoted": "#2E8B57",
    "deprioritized": "#C44E52",
}

METHOD_STYLES = {
    "Spagraph": {"color": "#D7263D", "marker": "D", "size": 74},
    "CellChat": {"color": "#7F8A99", "marker": "o", "size": 58},
    "COMMOT": {"color": "#B7791F", "marker": "s", "size": 58},
    "Giotto": {"color": "#7C3AED", "marker": "^", "size": 62},
}

GROUP_BAND_COLORS = ["#EEF2FF", "#ECFDF5", "#FEF2F2"]


def format_pair(pair: str) -> str:
    return pair.replace("_", "-")


def percentile_from_rank(rank: float, n: int) -> float:
    denom = max(n - 1, 1)
    return 100.0 * (1.0 - (rank - 1.0) / denom)


def load_panel_table() -> pd.DataFrame:
    base = pd.read_csv(CELLCHAT_SPAGRAPH_PATH)
    base = base[base["lr_pair"].isin([p for p, _ in PAIRS])].copy()
    base["pair_role"] = base["lr_pair"].map(dict(PAIRS))

    commot = pd.read_csv(COMMOT_PATH)
    commot = commot[["interaction_name", "commot_rank", "commot_percentile"]].copy()
    commot = commot.rename(columns={"interaction_name": "lr_pair"})

    giotto = pd.read_csv(GIOTTO_PATH)
    giotto = giotto[["interaction_name", "giotto_spatial_rank"]].copy()
    giotto = giotto.rename(columns={"interaction_name": "lr_pair"})
    giotto_n = len(pd.read_csv(GIOTTO_PATH))
    giotto["giotto_percentile"] = giotto["giotto_spatial_rank"].map(lambda x: percentile_from_rank(float(x), giotto_n))

    out = base.merge(commot, on="lr_pair", how="left").merge(giotto, on="lr_pair", how="left")
    out["pair_label"] = out["lr_pair"].map(format_pair)
    out["pair_order"] = out["lr_pair"].map({p: i for i, (p, _) in enumerate(PAIRS)})
    out = out.sort_values("pair_order").reset_index(drop=True)
    out["y"] = np.arange(len(out))[::-1]
    return out


def draw_panel_f(panel_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=160)

    pair_to_y = dict(zip(panel_df["lr_pair"], panel_df["y"]))
    for idx, (group_name, group_pairs) in enumerate(PAIR_GROUPS):
        ys = [pair_to_y[p] for p in group_pairs if p in pair_to_y]
        if not ys:
            continue
        band_lo = min(ys) - 0.48
        band_hi = max(ys) + 0.48
        ax.axhspan(band_lo, band_hi, color=GROUP_BAND_COLORS[idx], zorder=0)
        ax.text(
            94.5,
            band_lo + 0.18,
            group_name,
            ha="right",
            va="bottom",
            fontsize=9.2,
            color="#9CA3AF",
            fontstyle="italic",
        )

    ax.axvline(50, color="#D1D5DB", linewidth=0.9, linestyle=":", zorder=1)

    for row in panel_df.itertuples(index=False):
        values = {
            "CellChat": float(row.cellchat_rank_pct),
            "COMMOT": float(row.commot_percentile),
            "Giotto": float(row.giotto_percentile),
            "Spagraph": float(row.spagraph_rank_pct),
        }
        min_x = min(values.values())
        max_x = max(values.values())
        ax.plot([min_x, max_x], [row.y, row.y], color="#CBD5E1", linewidth=2.2, zorder=1, solid_capstyle="round")

        for method, x in values.items():
            style = METHOD_STYLES[method]
            ax.scatter(
                x,
                row.y,
                s=style["size"],
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.9,
                zorder=3 if method != "Spagraph" else 4,
            )

        sp_x = values["Spagraph"]
        cc_x = values["CellChat"]
        cm_x = values["COMMOT"]
        gi_x = values["Giotto"]

        label_specs = [
            (sp_x, row.y, int(row.spagraph_rank), METHOD_STYLES["Spagraph"]["color"], 9.8),
            (cc_x, row.y, int(row.cellchat_rank), METHOD_STYLES["CellChat"]["color"], 9.8),
            (cm_x, row.y, int(row.commot_rank), METHOD_STYLES["COMMOT"]["color"], 9.8),
            (gi_x, row.y, int(row.giotto_spatial_rank), METHOD_STYLES["Giotto"]["color"], 9.8),
        ]
        for x, y, rank_num, color, fs in label_specs:
            ax.text(
                x,
                y + 0.18,
                f"#{rank_num}",
                fontsize=fs,
                color=color,
                fontweight="bold",
                ha="center",
                va="bottom",
                clip_on=False,
            )

    ax.set_yticks(panel_df["y"])
    ax.set_yticklabels(panel_df["pair_label"], fontsize=11, fontweight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.6, len(panel_df) - 0.02)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Rank Percentile Within Each Method's Evaluable Universe", fontsize=11, fontweight="bold", labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    handles = []
    labels = []
    legend_order = ["Spagraph", "CellChat", "COMMOT", "Giotto"]
    for method in legend_order:
        style = METHOD_STYLES[method]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="none",
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.9,
                markersize=np.sqrt(style["size"]),
            )
        )
        labels.append(method)
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(OUTPUT_DIR / "figureF_multimethod_rank_strip_preview.png", dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_DIR / "figureF_multimethod_rank_strip_preview.pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    panel_df = load_panel_table()
    panel_df.to_csv(OUTPUT_DIR / "figureF_multimethod_rank_strip_preview_values.csv", index=False)
    draw_panel_f(panel_df)


if __name__ == "__main__":
    main()
