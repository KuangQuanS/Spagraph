#!/usr/bin/env python3
"""Create publication-ready CCC figures from analysis CSVs or raw LR communication exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.patheffects as pe
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = EVALUATE_DIR / "data"

DEFAULT_DATASET = "GSE144236"
DATASET_ANALYSIS_DIRS = {
    "CID44971": DATA_ROOT / "CID44971" / "ccc_analysis",
    "GSE144236": DATA_ROOT / "GSE144236" / "ccc_analysis",
    "GSE243275": DATA_ROOT / "GSE243275" / "ccc_analysis",
    "GSE211956_P3": DATA_ROOT / "GSE211956" / "P3" / "ccc_analysis",
}
DATASET_ST_H5AD = {
    "CID44971": EVALUATE_DIR.parent / "spagraph_data" / "database" / "Wu" / "CID44971" / "CID44971_ST.h5ad",
    "GSE144236": EVALUATE_DIR.parent / "spagraph_data" / "database" / "GSE144240" / "GSE144236_P2_ST.h5ad",
    "GSE243275": EVALUATE_DIR.parent / "spagraph_data" / "database" / "GSE243275" / "GSM7782699_ST.h5ad",
    "GSE211956_P3": EVALUATE_DIR.parent / "spagraph_data" / "database" / "GSE211956" / "GSE211956_ST_P3.h5ad",
}

EXPRESSION_STATS_CSV = "expression_attention_pair_stats.csv"
EXPRESSION_SUMMARY_CSV = "expression_attention_summary.csv"
PAIR_METRICS_CSV = "observed_pair_metrics.csv"
EXPECTED_LR_COLUMNS = {
    "src_spot_barcode",
    "dst_spot_barcode",
    "source_cell",
    "target_cell",
    "lr_pair",
    "original_lr_score",
    "attention_score",
}

COLOR_OTHER = "#D9D9D9"
COLOR_ATTENTION = "#C44E52"
COLOR_FREQUENCY = "#4C72B0"
COLOR_BOTH = "#7A5195"
BOX_COLORS = (COLOR_ATTENTION, COLOR_FREQUENCY)

# Figure size is (width, height) in inches.
# To make the main scatter wider/narrower, change SCATTER_FIGSIZE[0].
SCATTER_FIGSIZE = (14, 8)
# To make metric boxplots wider/narrower, change METRIC_BOXPLOT_FIGSIZE[0].
METRIC_BOXPLOT_FIGSIZE = (3.7, 4.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render standalone CCC publication figures from existing CSV outputs."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASET_ANALYSIS_DIRS),
        help="Dataset key when using repository defaults.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory containing observed_pair_metrics.csv and expression_attention_*.csv.",
    )
    parser.add_argument(
        "--lr-csv",
        type=Path,
        default=None,
        help="Optional raw lr_communication.csv path. If provided, metrics are rebuilt from this file.",
    )
    parser.add_argument(
        "--st-h5ad",
        type=Path,
        default=None,
        help="Spatial h5ad used when --lr-csv is given.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for new figures. Defaults to <analysis-dir>/paper_figures.",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=5,
        help="Minimum number of highlighted LR pairs from each ranking to annotate on the scatter plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI used for PNG export.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of LR pairs kept in each ranking group.",
    )
    parser.add_argument(
        "--min-pair-occurrence",
        type=int,
        default=10,
        help="Minimum LR-pair occurrence count retained for analysis.",
    )
    parser.add_argument(
        "--n-spot-neighbors",
        type=int,
        default=8,
        help="KNN graph size for Moran's I.",
    )
    parser.add_argument(
        "--show-titles",
        action="store_true",
        help="Add in-figure titles. Leave off for thesis/paper panel assembly.",
    )
    return parser.parse_args()


def resolve_analysis_dir(args: argparse.Namespace) -> Path:
    if args.analysis_dir is not None:
        return args.analysis_dir.resolve()
    return DATASET_ANALYSIS_DIRS[args.dataset]


def infer_dataset_from_path(path: Path) -> str | None:
    parts = {part for part in path.parts}
    if "GSE144236" in parts:
        return "GSE144236"
    if "CID44971" in parts:
        return "CID44971"
    if "GSE243275" in parts:
        return "GSE243275"
    if "GSE211956" in parts and "P3" in parts:
        return "GSE211956_P3"
    return None


def infer_dataset_label(analysis_dir: Path, fallback: str) -> str:
    if analysis_dir.name == "ccc_analysis":
        parent = analysis_dir.parent.name
        if parent == "P3" and analysis_dir.parent.parent.name == "GSE211956":
            return "GSE211956_P3"
        return parent
    return fallback


def load_required_csvs(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression_stats_path = analysis_dir / EXPRESSION_STATS_CSV
    expression_summary_path = analysis_dir / EXPRESSION_SUMMARY_CSV
    pair_metrics_path = analysis_dir / PAIR_METRICS_CSV

    missing = [str(path) for path in (expression_stats_path, expression_summary_path, pair_metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV files:\n" + "\n".join(missing))

    return (
        pd.read_csv(expression_stats_path),
        pd.read_csv(expression_summary_path),
        pd.read_csv(pair_metrics_path),
    )


def load_lr_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=list(EXPECTED_LR_COLUMNS), low_memory=False)
    df["lr_pair"] = df["lr_pair"].astype(str)
    df["src_spot_barcode"] = df["src_spot_barcode"].astype(str)
    df["dst_spot_barcode"] = df["dst_spot_barcode"].astype(str)
    df["source_cell"] = df["source_cell"].astype(str)
    df["target_cell"] = df["target_cell"].astype(str)
    df["original_lr_score"] = pd.to_numeric(df["original_lr_score"], errors="coerce")
    df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce")
    df = df.dropna(subset=["original_lr_score", "attention_score"]).copy()
    if df.empty:
        raise ValueError(f"No valid LR rows found in {csv_path}")
    return df


def load_spatial_adata(st_h5ad: Path):
    adata = sc.read_h5ad(st_h5ad)
    if "spatial" not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in {st_h5ad}")
    adata.obs_names = adata.obs_names.astype(str)
    return adata


def build_adjacency(coords: np.ndarray, k: int) -> np.ndarray:
    knn = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    adjacency = knn.toarray().astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def filter_known_spots(lr_df: pd.DataFrame, valid_spots: set[str]) -> pd.DataFrame:
    mask = lr_df["src_spot_barcode"].isin(valid_spots) & lr_df["dst_spot_barcode"].isin(valid_spots)
    return lr_df.loc[mask].copy()


def summarize_pairs(lr_df: pd.DataFrame) -> pd.DataFrame:
    return (
        lr_df.groupby("lr_pair", as_index=False)
        .agg(
            occurrence_count=("lr_pair", "size"),
            attention_mean=("attention_score", "mean"),
            attention_sum=("attention_score", "sum"),
            original_lr_sum=("original_lr_score", "sum"),
        )
        .reset_index(drop=True)
    )


def filter_candidate_pairs(pair_stats: pd.DataFrame, min_pair_occurrence: int) -> pd.DataFrame:
    filtered = pair_stats.loc[pair_stats["occurrence_count"] >= min_pair_occurrence].copy()
    if filtered.empty:
        raise ValueError(
            f"No LR pairs satisfy occurrence_count >= {min_pair_occurrence}."
        )
    return filtered.reset_index(drop=True)


def build_pair_spot_map(pair_df: pd.DataFrame, spot_index: dict[str, int]) -> np.ndarray:
    values = np.zeros(len(spot_index), dtype=float)
    src_contrib = pair_df.groupby("src_spot_barcode")["original_lr_score"].sum()
    dst_contrib = pair_df.groupby("dst_spot_barcode")["original_lr_score"].sum()
    for barcode, value in src_contrib.items():
        values[spot_index[barcode]] += float(value)
    for barcode, value in dst_contrib.items():
        values[spot_index[barcode]] += float(value)
    return values


def morans_i(values: np.ndarray, adjacency: np.ndarray) -> float:
    centered = values - values.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return float("nan")
    s0 = float(adjacency.sum())
    if s0 <= 0:
        return float("nan")
    numerator = float(centered.T @ adjacency @ centered)
    return (len(values) / s0) * (numerator / denominator)


def compute_celltype_entropy(pair_df: pd.DataFrame) -> tuple[float, int]:
    counts = pair_df.groupby(["source_cell", "target_cell"]).size()
    n_categories = len(counts)
    if n_categories <= 1:
        return 0.0, n_categories
    probs = counts.values / counts.values.sum()
    raw_entropy = -float(np.sum(probs * np.log2(probs)))
    return raw_entropy / np.log2(n_categories), n_categories


def precompute_candidate_metrics(
    lr_df: pd.DataFrame,
    pair_stats: pd.DataFrame,
    spot_index: dict[str, int],
    adjacency: np.ndarray,
) -> pd.DataFrame:
    pair_groups = lr_df.groupby("lr_pair", sort=False)
    rows = []
    for record in pair_stats.itertuples(index=False):
        pair_df = pair_groups.get_group(record.lr_pair)
        values = build_pair_spot_map(pair_df, spot_index)
        entropy, ct_count = compute_celltype_entropy(pair_df)
        rows.append(
            {
                "lr_pair": record.lr_pair,
                "occurrence_count": record.occurrence_count,
                "attention_mean": record.attention_mean,
                "attention_sum": record.attention_sum,
                "original_lr_sum": record.original_lr_sum,
                "pair_activity_sum": float(values.sum()),
                "moran_i": morans_i(values, adjacency),
                "celltype_entropy": entropy,
                "celltype_pair_count": ct_count,
            }
        )
    return pd.DataFrame(rows)


def select_top_pairs(candidate_metrics: pd.DataFrame, top_k: int, ranking_type: str) -> pd.DataFrame:
    if ranking_type == "attention":
        ranked = candidate_metrics.sort_values(
            ["attention_mean", "occurrence_count", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    elif ranking_type == "frequency":
        ranked = candidate_metrics.sort_values(
            ["occurrence_count", "attention_mean", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    else:
        raise ValueError(f"Unknown ranking type: {ranking_type}")
    result = ranked.head(top_k).copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "ranking_type", ranking_type)
    return result


def build_expression_attention_stats(
    candidate_metrics: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = candidate_metrics.copy()
    stats_df["log10_original_lr_sum_plus1"] = np.log10(stats_df["original_lr_sum"].astype(float) + 1.0)
    stats_df["selection_group"] = "other"
    attention_pairs = top_pairs_df.loc[top_pairs_df["ranking_type"] == "attention", ["lr_pair", "rank"]]
    frequency_pairs = top_pairs_df.loc[top_pairs_df["ranking_type"] == "frequency", ["lr_pair", "rank"]]
    stats_df["attention_rank"] = stats_df["lr_pair"].map(dict(zip(attention_pairs["lr_pair"], attention_pairs["rank"])))
    stats_df["frequency_rank"] = stats_df["lr_pair"].map(dict(zip(frequency_pairs["lr_pair"], frequency_pairs["rank"])))
    is_attention = stats_df["attention_rank"].notna()
    is_frequency = stats_df["frequency_rank"].notna()
    stats_df.loc[is_attention & ~is_frequency, "selection_group"] = "top_attention"
    stats_df.loc[~is_attention & is_frequency, "selection_group"] = "top_frequency"
    stats_df.loc[is_attention & is_frequency, "selection_group"] = "top_both"
    rho, p_value = spearmanr(
        stats_df["original_lr_sum"].astype(float).to_numpy(),
        stats_df["attention_mean"].astype(float).to_numpy(),
    )
    summary_df = pd.DataFrame(
        [
            {
                "spearman_rho": float(rho) if rho == rho else float("nan"),
                "spearman_pvalue": float(p_value) if p_value == p_value else float("nan"),
            }
        ]
    )
    return stats_df, summary_df


def build_metrics_from_lr_csv(
    lr_csv: Path,
    st_h5ad: Path,
    top_k: int,
    min_pair_occurrence: int,
    n_spot_neighbors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr_df = load_lr_table(lr_csv)
    adata = load_spatial_adata(st_h5ad)
    coords = np.asarray(adata.obsm["spatial"])
    spot_names = adata.obs_names.astype(str).tolist()
    lr_df = filter_known_spots(lr_df, set(spot_names))
    adjacency = build_adjacency(coords, n_spot_neighbors)
    spot_index = {barcode: idx for idx, barcode in enumerate(spot_names)}

    pair_stats = filter_candidate_pairs(summarize_pairs(lr_df), min_pair_occurrence)
    candidate_metrics = precompute_candidate_metrics(lr_df, pair_stats, spot_index, adjacency)
    top_attention = select_top_pairs(candidate_metrics, top_k=top_k, ranking_type="attention")
    top_frequency = select_top_pairs(candidate_metrics, top_k=top_k, ranking_type="frequency")
    selected_pairs = pd.concat([top_attention, top_frequency], ignore_index=True)
    expression_pair_stats, expression_summary = build_expression_attention_stats(candidate_metrics, selected_pairs)
    return expression_pair_stats, selected_pairs


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 13,
            "font.weight": "semibold",
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#333333")
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.9)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def format_pvalue(p_value: float) -> str:
    if np.isnan(p_value):
        return "p = NA"
    if p_value < 1e-3:
        return f"p = {p_value:.1e}"
    return f"p = {p_value:.3f}"


def choose_labels(pair_stats_df: pd.DataFrame, label_top_n: int) -> pd.DataFrame:
    # Always label LR pairs that are selected by both rankings,
    # then add the top-N labels from each individual ranking.
    both = pair_stats_df.loc[pair_stats_df["selection_group"] == "top_both"].copy()
    attention = pair_stats_df.loc[pair_stats_df["attention_rank"].fillna(np.inf).le(label_top_n)].copy()
    frequency = pair_stats_df.loc[pair_stats_df["frequency_rank"].fillna(np.inf).le(label_top_n)].copy()
    selected = pd.concat([both, attention, frequency], ignore_index=True).drop_duplicates(subset=["lr_pair"]).copy()
    if selected.empty:
        return selected

    group_priority = {"top_both": 0, "top_attention": 1, "top_frequency": 2}
    selected["group_priority"] = selected["selection_group"].map(group_priority).fillna(3)
    selected["best_rank"] = selected[["attention_rank", "frequency_rank"]].min(axis=1, skipna=True)
    selected = selected.sort_values(
        ["group_priority", "best_rank", "attention_mean", "original_lr_sum"],
        ascending=[True, True, False, False],
    )
    return selected.reset_index(drop=True)


def save_expression_attention_plot(
    pair_stats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    dataset_label: str,
    label_top_n: int,
    show_title: bool,
    dpi: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    style_map = {
        "other": (COLOR_OTHER, 38, 0.72, "Other", 0),
        "top_attention": (COLOR_ATTENTION, 108, 0.96, "Attn-top", 2),
        "top_frequency": (COLOR_FREQUENCY, 108, 0.96, "Freq-top", 2),
        "top_both": (COLOR_BOTH, 128, 0.99, "Both", 3),
    }

    for selection_group, (color, size, alpha, label, zorder) in style_map.items():
        group_df = pair_stats_df.loc[pair_stats_df["selection_group"] == selection_group]
        if group_df.empty:
            continue
        ax.scatter(
            group_df["log10_original_lr_sum_plus1"],
            group_df["attention_mean"],
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0.9 if selection_group != "other" else 0.0,
            edgecolors="white" if selection_group != "other" else "none",
            label=label,
            zorder=zorder,
        )

    label_df = choose_labels(pair_stats_df, label_top_n=label_top_n)
    x_mid = float(pair_stats_df["log10_original_lr_sum_plus1"].median())
    y_mid = float(pair_stats_df["attention_mean"].median())
    ax.set_xlim(
        float(pair_stats_df["log10_original_lr_sum_plus1"].min()) - 0.08,
        float(pair_stats_df["log10_original_lr_sum_plus1"].max()) + 0.38,
    )

    # Automatically spread labels to reduce overlap while keeping guide lines.
    from adjustText import adjust_text
    
    texts = []
    for row in label_df.itertuples(index=False):
        label_text = ax.text(
            row.log10_original_lr_sum_plus1,
            row.attention_mean,
            row.lr_pair,
            fontsize=14.5,
            fontweight="bold",
            color="#222222",
            zorder=4,
        )
        label_text.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.8)])
        texts.append(label_text)

    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=1.0, zorder=3),
            min_arrow_len=15,
            expand=(1.2, 1.2),
            force_text=(0.5, 0.5),
            force_static=(0.5, 0.5)
        )

    rho = float(summary_df.iloc[0]["spearman_rho"])
    p_value = float(summary_df.iloc[0]["spearman_pvalue"])
    annotation = f"rho = {rho:.3f}\n{format_pvalue(p_value)}"
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#C7C7C7", "alpha": 0.96},
    )

    if show_title:
        ax.set_title(f"{dataset_label}: LR abundance vs attention", pad=10)
    ax.set_xlabel("log10(LR abundance + 1)")
    ax.set_ylabel("Mean attention")
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.grid(color="#EAEAEA", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        markerscale=1.3,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontsize(14.5)
    style_axis(ax)
    ax.tick_params(labelsize=14.5)
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    save_kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() == ".png" and dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def build_boxplot_panel(
    ax: plt.Axes,
    metrics_df: pd.DataFrame,
    metric: str,
    label: str,
    show_title: bool,
) -> None:
    attention = metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna().to_numpy()
    frequency = metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna().to_numpy()
    values = [attention, frequency]

    box = ax.boxplot(
        values,
        tick_labels=[f"Attn-top\n(n={len(attention)})", f"Freq-top\n(n={len(frequency)})"],
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
        boxprops={"edgecolor": "#666666", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    rng = np.random.default_rng(42)
    for pos, (series, color) in enumerate(zip(values, BOX_COLORS), start=1):
        if len(series) == 0:
            continue
        jitter = rng.normal(0.0, 0.045, size=len(series))
        ax.scatter(
            np.full(len(series), pos) + jitter,
            series,
            s=20,
            color=color,
            alpha=0.82,
            linewidths=0.3,
            edgecolors="white",
            zorder=3,
        )

    style_axis(ax)
    if show_title:
        ax.set_title(label, pad=8)
    else:
        ax.set_title(label, pad=8, fontsize=11)
    ax.set_ylabel(label)

    p_value = float("nan")
    if len(attention) and len(frequency):
        p_value = float(mannwhitneyu(attention, frequency, alternative="two-sided").pvalue)
    ax.text(
        0.5,
        0.98,
        format_pvalue(p_value),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.95},
    )


def save_metric_boxplot(
    metrics_df: pd.DataFrame,
    output_path: Path,
    dataset_label: str,
    show_title: bool,
    metric: str,
    label: str,
    dpi: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=METRIC_BOXPLOT_FIGSIZE)
    build_boxplot_panel(ax=ax, metrics_df=metrics_df, metric=metric, label=label, show_title=show_title)
    if show_title:
        fig.suptitle(f"{dataset_label}: {label}", y=1.02, fontsize=12)
    fig.tight_layout()
    save_kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() == ".png" and dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def export_figures(
    analysis_dir: Path,
    output_dir: Path,
    dataset_label: str,
    expression_stats_df: pd.DataFrame,
    expression_summary_df: pd.DataFrame,
    pair_metrics_df: pd.DataFrame,
    label_top_n: int,
    dpi: int,
    show_titles: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    expression_pdf = output_dir / "expression_vs_attention_publication.pdf"
    expression_png = output_dir / "expression_vs_attention_publication.png"
    metric_specs = [
        ("moran_i", "Moran's I", "moran_i_boxplot_publication"),
        ("celltype_pair_count", "Cell-type Pair Count", "celltype_pair_count_boxplot_publication"),
    ]

    save_expression_attention_plot(
        pair_stats_df=expression_stats_df,
        summary_df=expression_summary_df,
        output_path=expression_pdf,
        dataset_label=dataset_label,
        label_top_n=label_top_n,
        show_title=show_titles,
        dpi=dpi,
    )
    save_expression_attention_plot(
        pair_stats_df=expression_stats_df,
        summary_df=expression_summary_df,
        output_path=expression_png,
        dataset_label=dataset_label,
        label_top_n=label_top_n,
        show_title=show_titles,
        dpi=dpi,
    )
    for metric, label, stem in metric_specs:
        save_metric_boxplot(
            metrics_df=pair_metrics_df,
            output_path=output_dir / f"{stem}.pdf",
            dataset_label=dataset_label,
            show_title=show_titles,
            metric=metric,
            label=label,
            dpi=dpi,
        )
        save_metric_boxplot(
            metrics_df=pair_metrics_df,
            output_path=output_dir / f"{stem}.png",
            dataset_label=dataset_label,
            show_title=show_titles,
            metric=metric,
            label=label,
            dpi=dpi,
        )

    summary_txt = output_dir / "paper_figure_inputs.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"dataset={dataset_label}",
                f"analysis_dir={analysis_dir}",
                f"expression_stats={analysis_dir / EXPRESSION_STATS_CSV}",
                f"expression_summary={analysis_dir / EXPRESSION_SUMMARY_CSV}",
                f"pair_metrics={analysis_dir / PAIR_METRICS_CSV}",
                f"label_top_n={label_top_n}",
                f"show_titles={show_titles}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_st_h5ad(args: argparse.Namespace, dataset_key: str | None) -> Path:
    if args.st_h5ad is not None:
        return args.st_h5ad.resolve()
    if dataset_key is not None and dataset_key in DATASET_ST_H5AD:
        return DATASET_ST_H5AD[dataset_key]
    raise ValueError("Could not infer st_h5ad from --lr-csv. Please pass --st-h5ad explicitly.")


def main() -> None:
    args = parse_args()
    apply_publication_style()

    if args.lr_csv is not None:
        lr_csv = args.lr_csv.resolve()
        dataset_key = infer_dataset_from_path(lr_csv)
        st_h5ad = resolve_st_h5ad(args, dataset_key)
        dataset_label = ""
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else lr_csv.parent / "paper_figures_from_lr"
        )
        expression_stats_df, pair_metrics_df = build_metrics_from_lr_csv(
            lr_csv=lr_csv,
            st_h5ad=st_h5ad,
            top_k=args.top_k,
            min_pair_occurrence=args.min_pair_occurrence,
            n_spot_neighbors=args.n_spot_neighbors,
        )
        _, expression_summary_df = build_expression_attention_stats(
            expression_stats_df.drop(
                columns=["log10_original_lr_sum_plus1", "selection_group", "attention_rank", "frequency_rank"],
                errors="ignore",
            ),
            pair_metrics_df,
        )
        analysis_dir = lr_csv.parent
    else:
        analysis_dir = resolve_analysis_dir(args)
        dataset_label = infer_dataset_label(analysis_dir, fallback=args.dataset)
        output_dir = args.output_dir.resolve() if args.output_dir is not None else (analysis_dir / "paper_figures")
        expression_stats_df, expression_summary_df, pair_metrics_df = load_required_csvs(analysis_dir)

    export_figures(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        dataset_label=dataset_label,
        expression_stats_df=expression_stats_df,
        expression_summary_df=expression_summary_df,
        pair_metrics_df=pair_metrics_df,
        label_top_n=args.label_top_n,
        dpi=args.dpi,
        show_titles=args.show_titles,
    )
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
