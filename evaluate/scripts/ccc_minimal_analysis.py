#!/usr/bin/env python3
"""Observed-only CCC analysis: Moran's I, cell-type specificity, and abundance-attention decoupling."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import kneighbors_graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent
DATA_ROOT = EVALUATE_DIR / "data"
REPO_ROOT = EVALUATE_DIR.parent
DATABASE_ROOT = REPO_ROOT / "spagraph_data" / "database"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spagraph


EXPECTED_LR_COLUMNS = {
    "src_spot_barcode",
    "dst_spot_barcode",
    "source_cell",
    "target_cell",
    "lr_pair",
    "original_lr_score",
    "attention_score",
}

ANALYSIS_MODE = "observed_moran_specificity_expression_decoupling"
PAIR_SIGNAL = "original_lr_score_sum"
MIN_PAIR_OCCURRENCE = 10
OBSERVED_RUN_DIRNAME = "_observed_run"
OBSERVED_CELLCOM_DIRNAME = "cellcom"
CELLCOM_UNIFIED_CSV = "lr_communication.csv"
LEGACY_OUTPUT_FILES = (
    "boundary_spots.csv",
    "matched_random_summary.csv",
    "matched_random_group_summary.csv",
    "permutation_summary.csv",
    "permutation_top_pairs.csv",
)
LEGACY_FIGURE_FILES = (
    "matched_random_null_distribution.pdf",
    "moran_attention_vs_frequency.pdf",
    "permutation_null_distribution.pdf",
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset_dir: Path
    composition_csv: Path
    spot_cell_expr_csv: Path | None
    st_h5ad: Path
    output_dir_override: Path | None = None
    n_spot_neighbors: int = 8
    ligand_expr_threshold: float = 3.0
    receptor_expr_threshold: float = 3.0
    allow_same_celltype_comm: bool = True
    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 0
    save_lr_scores_csv: bool = False
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        if self.output_dir_override is not None:
            return self.output_dir_override
        return self.dataset_dir / "ccc_analysis"


DATASETS = {
    "CID44971": DatasetConfig(
        key="CID44971",
        dataset_dir=DATA_ROOT / "CID44971",
        composition_csv=DATA_ROOT / "CID44971" / "CID44971_ST_composition.csv",
        spot_cell_expr_csv=DATA_ROOT / "CID44971" / "CID44971_ST_spot_cell_expr.csv",
        st_h5ad=DATABASE_ROOT / "Wu" / "CID44971" / "CID44971_ST.h5ad",
    ),
    "GSE243275": DatasetConfig(
        key="GSE243275",
        dataset_dir=DATA_ROOT / "GSE243275",
        composition_csv=DATA_ROOT / "GSE243275" / "GSM7782699_ST_composition.csv",
        spot_cell_expr_csv=DATA_ROOT / "GSE243275" / "GSM7782699_ST_spot_cell_expr.csv",
        st_h5ad=DATABASE_ROOT / "GSE243275" / "GSM7782699_ST.h5ad",
    ),
    "GSE144236": DatasetConfig(
        key="GSE144236",
        dataset_dir=DATA_ROOT / "GSE144236",
        composition_csv=DATA_ROOT / "GSE144236" / "Spatial_composition.csv",
        spot_cell_expr_csv=DATA_ROOT / "GSE144236" / "Spatial_spot_cell_expr.csv",
        st_h5ad=DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad",
    ),
    "GSE211956_P3": DatasetConfig(
        key="GSE211956_P3",
        dataset_dir=DATA_ROOT / "GSE211956" / "P3",
        composition_csv=DATA_ROOT / "GSE211956" / "P3" / "GSE211956_ST_P3_cell_composition.csv",
        spot_cell_expr_csv=None,
        st_h5ad=DATABASE_ROOT / "GSE211956" / "GSE211956_ST_P3.h5ad",
    ),
}


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observed-only CCC analysis: Moran's I, cell-type specificity, and expression-attention decoupling."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", *sorted(DATASETS)],
        help="Datasets to run. Use 'all' to run every configured dataset.",
    )
    parser.add_argument("--run-name", default=None, help="Run label for explicit input mode.")
    parser.add_argument("--composition-csv", type=Path, default=None)
    parser.add_argument("--spot-cell-expr-csv", type=Path, default=None)
    parser.add_argument("--st-h5ad", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-spot-neighbors", type=int, default=None)
    parser.add_argument("--ligand-expr-threshold", type=float, default=None)
    parser.add_argument("--receptor-expr-threshold", type=float, default=None)
    parser.add_argument("--allow-same-celltype-comm", type=parse_bool, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-lr-scores-csv", type=parse_bool, default=None)
    parser.add_argument("--cellcom-seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-pair-occurrence", type=int, default=MIN_PAIR_OCCURRENCE)
    parser.add_argument("--seed", type=int, default=42, help="Analysis seed for deterministic output metadata.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining datasets if one dataset fails.",
    )
    return parser.parse_args()


def resolve_dataset_keys(selected: list[str]) -> list[str]:
    if "all" in selected:
        return list(DATASETS)
    return selected


def uses_explicit_config(args: argparse.Namespace) -> bool:
    explicit_fields = [
        "run_name",
        "composition_csv",
        "spot_cell_expr_csv",
        "st_h5ad",
        "output_dir",
    ]
    return any(getattr(args, field) is not None for field in explicit_fields)


def apply_runtime_overrides(config: DatasetConfig, args: argparse.Namespace) -> DatasetConfig:
    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir_override"] = args.output_dir
    if args.n_spot_neighbors is not None:
        overrides["n_spot_neighbors"] = args.n_spot_neighbors
    if args.ligand_expr_threshold is not None:
        overrides["ligand_expr_threshold"] = args.ligand_expr_threshold
    if args.receptor_expr_threshold is not None:
        overrides["receptor_expr_threshold"] = args.receptor_expr_threshold
    if args.allow_same_celltype_comm is not None:
        overrides["allow_same_celltype_comm"] = args.allow_same_celltype_comm
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.save_lr_scores_csv is not None:
        overrides["save_lr_scores_csv"] = args.save_lr_scores_csv
    if args.cellcom_seed is not None:
        overrides["seed"] = args.cellcom_seed
    if not overrides:
        return config
    return replace(config, **overrides)


def build_explicit_config(args: argparse.Namespace) -> DatasetConfig:
    if args.output_root is not None:
        raise ValueError("Explicit mode uses --output-dir and does not support --output-root.")

    required = {
        "run_name": args.run_name,
        "composition_csv": args.composition_csv,
        "spot_cell_expr_csv": args.spot_cell_expr_csv,
        "st_h5ad": args.st_h5ad,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Explicit mode is missing required input arguments: " + ", ".join(missing))

    return DatasetConfig(
        key=args.run_name,
        dataset_dir=args.composition_csv.parent,
        composition_csv=args.composition_csv,
        spot_cell_expr_csv=args.spot_cell_expr_csv,
        st_h5ad=args.st_h5ad,
        output_dir_override=args.output_dir,
        n_spot_neighbors=args.n_spot_neighbors if args.n_spot_neighbors is not None else 8,
        ligand_expr_threshold=args.ligand_expr_threshold if args.ligand_expr_threshold is not None else 3.0,
        receptor_expr_threshold=args.receptor_expr_threshold if args.receptor_expr_threshold is not None else 3.0,
        allow_same_celltype_comm=(
            args.allow_same_celltype_comm if args.allow_same_celltype_comm is not None else True
        ),
        epochs=args.epochs if args.epochs is not None else 200,
        batch_size=args.batch_size if args.batch_size is not None else 64,
        num_workers=args.num_workers if args.num_workers is not None else 0,
        save_lr_scores_csv=args.save_lr_scores_csv if args.save_lr_scores_csv is not None else False,
        seed=args.cellcom_seed if args.cellcom_seed is not None else args.seed,
    )


def resolve_output_dir(config: DatasetConfig, output_root: Path | None) -> Path:
    if output_root is None:
        return config.output_dir
    return output_root / config.key


def require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")


def resolve_spot_cell_expr_csv(config: DatasetConfig, required: bool) -> Path | None:
    if config.spot_cell_expr_csv is not None and config.spot_cell_expr_csv.exists():
        return config.spot_cell_expr_csv

    matches = sorted(config.dataset_dir.glob("*_spot_cell_expr.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{config.key} has multiple '*_spot_cell_expr.csv' files under {config.dataset_dir}: "
            + ", ".join(path.name for path in matches)
        )
    if required:
        if config.spot_cell_expr_csv is not None:
            raise FileNotFoundError(
                f"{config.key} is missing its configured Stage 2 spot-cell expression file: {config.spot_cell_expr_csv}"
            )
        raise FileNotFoundError(
            f"{config.key} is missing a Stage 2 '*_spot_cell_expr.csv' file under {config.dataset_dir}. "
            "Rerun deconvolution with save_reconstructed_genes=True before Stage 3 reruns."
        )
    return None


def read_lr_table(csv_path: Path) -> pd.DataFrame:
    require_existing(csv_path)
    try:
        return pd.read_csv(csv_path, usecols=list(EXPECTED_LR_COLUMNS))
    except ValueError as exc:
        raise ValueError(
            f"{csv_path} is missing required columns from {sorted(EXPECTED_LR_COLUMNS)}"
        ) from exc


def load_spatial_adata(st_h5ad: Path) -> sc.AnnData:
    require_existing(st_h5ad)
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
    grouped = (
        lr_df.groupby("lr_pair", as_index=False)
        .agg(
            occurrence_count=("lr_pair", "size"),
            attention_mean=("attention_score", "mean"),
            attention_sum=("attention_score", "sum"),
            original_lr_sum=("original_lr_score", "sum"),
        )
    )
    return grouped.reset_index(drop=True)


def filter_candidate_pairs(pair_stats: pd.DataFrame, min_pair_occurrence: int) -> pd.DataFrame:
    filtered = pair_stats.loc[pair_stats["occurrence_count"] >= min_pair_occurrence].copy()
    if filtered.empty:
        raise ValueError(
            f"No LR pairs satisfy occurrence_count >= {min_pair_occurrence}. "
            "Lower the threshold or inspect the Stage 3 communication export."
        )
    return filtered.reset_index(drop=True)


def build_pair_spot_map(pair_df: pd.DataFrame, spot_index: dict[str, int]) -> np.ndarray:
    values = np.zeros(len(spot_index), dtype=float)
    if pair_df.empty:
        return values

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
    if pair_df.empty:
        return float("nan"), 0
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


def compare_groups(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_to_compare = ["moran_i", "celltype_entropy", "celltype_pair_count"]
    rows = []
    for metric in metrics_to_compare:
        attention = metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna()
        frequency = metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna()
        p_value = float("nan")
        if len(attention) and len(frequency):
            p_value = float(mannwhitneyu(attention, frequency, alternative="two-sided").pvalue)
        rows.append(
            {
                "metric": metric,
                "attention_mean": attention.mean() if len(attention) else float("nan"),
                "frequency_mean": frequency.mean() if len(frequency) else float("nan"),
                "attention_median": attention.median() if len(attention) else float("nan"),
                "frequency_median": frequency.median() if len(frequency) else float("nan"),
                "attention_n": len(attention),
                "frequency_n": len(frequency),
                "mannwhitney_pvalue": p_value,
            }
        )
    return pd.DataFrame(rows)


def build_expression_attention_stats(
    candidate_metrics: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    analysis_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = candidate_metrics.copy()
    stats_df["log10_original_lr_sum_plus1"] = np.log10(stats_df["original_lr_sum"].astype(float) + 1.0)
    stats_df["log10_occurrence_count_plus1"] = np.log10(stats_df["occurrence_count"].astype(float) + 1.0)
    stats_df["selection_group"] = "other"
    stats_df["attention_rank"] = pd.NA
    stats_df["frequency_rank"] = pd.NA

    attention_pairs = top_pairs_df.loc[top_pairs_df["ranking_type"] == "attention", ["lr_pair", "rank"]]
    frequency_pairs = top_pairs_df.loc[top_pairs_df["ranking_type"] == "frequency", ["lr_pair", "rank"]]
    attention_rank_map = dict(zip(attention_pairs["lr_pair"], attention_pairs["rank"]))
    frequency_rank_map = dict(zip(frequency_pairs["lr_pair"], frequency_pairs["rank"]))

    stats_df["attention_rank"] = stats_df["lr_pair"].map(attention_rank_map)
    stats_df["frequency_rank"] = stats_df["lr_pair"].map(frequency_rank_map)

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
                "x_metric": "log10(original_lr_abundance + 1)",
                "y_metric": "attention_mean",
                "n_pairs": int(len(stats_df)),
                "analysis_seed": analysis_seed,
                "spearman_rho": float(rho) if rho == rho else float("nan"),
                "spearman_pvalue": float(p_value) if p_value == p_value else float("nan"),
                "top_attention_count": int(is_attention.sum()),
                "top_frequency_count": int(is_frequency.sum()),
            }
        ]
    )
    return stats_df, summary_df


def save_metric_boxplot(metrics_df: pd.DataFrame, output_path: Path, dataset_key: str) -> None:
    panel_specs = [
        ("moran_i", "Moran's I"),
        ("celltype_entropy", "Cell-type Entropy"),
        ("celltype_pair_count", "Cell-type Pair Count"),
    ]
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(5 * len(panel_specs), 5))
    if len(panel_specs) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, panel_specs):
        values = [
            metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna().to_numpy(),
            metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna().to_numpy(),
        ]
        ax.boxplot(values, labels=["attention", "frequency"], patch_artist=True)
        ax.set_title(f"{dataset_key}: {label}")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        if len(values[0]) and len(values[1]):
            p_value = float(mannwhitneyu(values[0], values[1], alternative="two-sided").pvalue)
            ax.text(
                0.5,
                0.98,
                f"p = {p_value:.2e}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_expression_attention_plot(
    pair_stats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    dataset_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    style_map = {
        "other": ("#bdbdbd", 28, 0.55, "other"),
        "top_attention": ("#d73027", 48, 0.9, "top attention"),
        "top_frequency": ("#4575b4", 48, 0.9, "top frequency"),
        "top_both": ("#7b3294", 56, 0.95, "top both"),
    }

    for selection_group, (color, size, alpha, label) in style_map.items():
        group_df = pair_stats_df.loc[pair_stats_df["selection_group"] == selection_group]
        if group_df.empty:
            continue
        ax.scatter(
            group_df["log10_original_lr_sum_plus1"],
            group_df["attention_mean"],
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0,
            label=label,
        )

    attention_labels = (
        pair_stats_df.loc[pair_stats_df["attention_rank"].notna()]
        .sort_values("attention_rank")
        .head(10)
    )
    for row in attention_labels.itertuples(index=False):
        ax.annotate(
            row.lr_pair,
            (row.log10_original_lr_sum_plus1, row.attention_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#7f0000",
        )

    rho = summary_df.iloc[0]["spearman_rho"]
    p_value = summary_df.iloc[0]["spearman_pvalue"]
    ax.set_title(f"{dataset_key}: LR abundance vs attention")
    ax.set_xlabel("log10(original LR abundance + 1)")
    ax.set_ylabel("Mean attention score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"Spearman rho = {rho:.3f}\np = {p_value:.3e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_lr_table(
    lr_df: pd.DataFrame,
    spot_names: list[str],
    coords: np.ndarray,
    config: DatasetConfig,
    top_k: int,
    min_pair_occurrence: int,
    analysis_seed: int,
) -> dict[str, pd.DataFrame]:
    lr_df = filter_known_spots(lr_df, set(spot_names))
    adjacency = build_adjacency(coords, config.n_spot_neighbors)
    spot_index = {barcode: idx for idx, barcode in enumerate(spot_names)}

    pair_stats = filter_candidate_pairs(summarize_pairs(lr_df), min_pair_occurrence)
    candidate_metrics = precompute_candidate_metrics(lr_df, pair_stats, spot_index, adjacency)

    top_attention = select_top_pairs(candidate_metrics, top_k=top_k, ranking_type="attention")
    top_frequency = select_top_pairs(candidate_metrics, top_k=top_k, ranking_type="frequency")
    selected_pairs = pd.concat([top_attention, top_frequency], ignore_index=True)
    group_summary = compare_groups(selected_pairs)
    expression_pair_stats, expression_summary = build_expression_attention_stats(
        candidate_metrics,
        selected_pairs,
        analysis_seed,
    )

    return {
        "top_pairs": selected_pairs,
        "pair_metrics": selected_pairs.copy(),
        "group_summary": group_summary,
        "expression_pair_stats": expression_pair_stats,
        "expression_summary": expression_summary,
    }


def run_cellcom_rerun(
    config: DatasetConfig,
    spot_cell_expr_csv: Path,
    st_h5ad: Path,
    cellcom_output_dir: Path,
    device: str,
) -> Path:
    spagraph.cellcom(
        deconv_dir=str(config.dataset_dir),
        st_h5ad=str(st_h5ad),
        output_dir=str(cellcom_output_dir),
        composition_csv=str(config.composition_csv),
        spot_cell_expr_csv=str(spot_cell_expr_csv),
        n_spot_neighbors=config.n_spot_neighbors,
        ligand_expr_threshold=config.ligand_expr_threshold,
        receptor_expr_threshold=config.receptor_expr_threshold,
        allow_same_celltype_comm=config.allow_same_celltype_comm,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        save_lr_scores_csv=config.save_lr_scores_csv,
        export_unified_csv=True,
        export_filtered_csv=False,
        seed=config.seed,
        device=device,
    )
    unified_csv = cellcom_output_dir / CELLCOM_UNIFIED_CSV
    require_existing(unified_csv)
    return unified_csv


def run_observed_cellcom(
    config: DatasetConfig,
    output_dir: Path,
    spot_cell_expr_csv: Path,
    device: str,
) -> Path:
    observed_output_dir = output_dir / OBSERVED_RUN_DIRNAME / OBSERVED_CELLCOM_DIRNAME
    return run_cellcom_rerun(
        config=config,
        spot_cell_expr_csv=spot_cell_expr_csv,
        st_h5ad=config.st_h5ad,
        cellcom_output_dir=observed_output_dir,
        device=device,
    )


def remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def cleanup_outputs(output_dir: Path, figures_dir: Path) -> None:
    for filename in LEGACY_OUTPUT_FILES:
        remove_file_if_exists(output_dir / filename)
    for filename in LEGACY_FIGURE_FILES:
        remove_file_if_exists(figures_dir / filename)
    shutil.rmtree(output_dir / "_cellcom_inputs", ignore_errors=True)


def save_run_info(
    config: DatasetConfig,
    output_dir: Path,
    top_k: int,
    min_pair_occurrence: int,
    analysis_seed: int,
) -> None:
    try:
        resolved_spot_cell_expr = resolve_spot_cell_expr_csv(config, required=False)
    except FileNotFoundError as exc:
        resolved_spot_cell_expr = f"unresolved ({exc})"

    info_lines = [
        f"analysis_mode={ANALYSIS_MODE}",
        f"dataset={config.key}",
        f"dataset_dir={config.dataset_dir}",
        f"composition_csv={config.composition_csv}",
        f"spot_cell_expr_csv={resolved_spot_cell_expr if resolved_spot_cell_expr is not None else 'missing'}",
        f"st_h5ad={config.st_h5ad}",
        f"output_dir={output_dir}",
        f"observed_source={OBSERVED_RUN_DIRNAME}/{OBSERVED_CELLCOM_DIRNAME}/{CELLCOM_UNIFIED_CSV}",
        f"pair_signal={PAIR_SIGNAL}",
        f"top_k={top_k}",
        f"min_pair_occurrence={min_pair_occurrence}",
        f"analysis_seed={analysis_seed}",
        f"cellcom_seed={config.seed}",
        f"n_spot_neighbors={config.n_spot_neighbors}",
        f"ligand_expr_threshold={config.ligand_expr_threshold}",
        f"receptor_expr_threshold={config.receptor_expr_threshold}",
        f"allow_same_celltype_comm={config.allow_same_celltype_comm}",
        f"epochs={config.epochs}",
        f"batch_size={config.batch_size}",
        f"num_workers={config.num_workers}",
        f"save_lr_scores_csv={config.save_lr_scores_csv}",
    ]
    (output_dir / "analysis_run_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")


def save_outputs(
    config: DatasetConfig,
    output_dir: Path,
    observed_results: dict[str, pd.DataFrame],
    top_k: int,
    min_pair_occurrence: int,
    analysis_seed: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cleanup_outputs(output_dir=output_dir, figures_dir=figures_dir)

    observed_results["top_pairs"].to_csv(output_dir / "observed_top_pairs.csv", index=False)
    observed_results["pair_metrics"].to_csv(output_dir / "observed_pair_metrics.csv", index=False)
    observed_results["group_summary"].to_csv(output_dir / "observed_group_summary.csv", index=False)
    observed_results["expression_pair_stats"].to_csv(
        output_dir / "expression_attention_pair_stats.csv",
        index=False,
    )
    observed_results["expression_summary"].to_csv(
        output_dir / "expression_attention_summary.csv",
        index=False,
    )

    save_run_info(
        config=config,
        output_dir=output_dir,
        top_k=top_k,
        min_pair_occurrence=min_pair_occurrence,
        analysis_seed=analysis_seed,
    )

    save_metric_boxplot(
        observed_results["pair_metrics"],
        figures_dir / "metrics_attention_vs_frequency.pdf",
        config.key,
    )
    save_expression_attention_plot(
        observed_results["expression_pair_stats"],
        observed_results["expression_summary"],
        figures_dir / "expression_vs_attention.pdf",
        config.key,
    )


def save_combined_outputs(
    combined_dir: Path,
    dataset_summaries: list[pd.DataFrame],
    dataset_pair_metrics: list[pd.DataFrame],
    dataset_top_pairs: list[pd.DataFrame],
    dataset_expression_pair_stats: list[pd.DataFrame],
    dataset_expression_summaries: list[pd.DataFrame],
    run_status_rows: list[dict[str, object]],
) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    if dataset_summaries:
        pd.concat(dataset_summaries, ignore_index=True).to_csv(
            combined_dir / "combined_observed_group_summary.csv",
            index=False,
        )
    if dataset_pair_metrics:
        pd.concat(dataset_pair_metrics, ignore_index=True).to_csv(
            combined_dir / "combined_observed_pair_metrics.csv",
            index=False,
        )
    if dataset_top_pairs:
        pd.concat(dataset_top_pairs, ignore_index=True).to_csv(
            combined_dir / "combined_observed_top_pairs.csv",
            index=False,
        )
    if dataset_expression_pair_stats:
        pd.concat(dataset_expression_pair_stats, ignore_index=True).to_csv(
            combined_dir / "combined_expression_attention_pair_stats.csv",
            index=False,
        )
    if dataset_expression_summaries:
        pd.concat(dataset_expression_summaries, ignore_index=True).to_csv(
            combined_dir / "combined_expression_attention_summary.csv",
            index=False,
        )
    pd.DataFrame(run_status_rows).to_csv(combined_dir / "dataset_run_status.csv", index=False)


def run_dataset(config: DatasetConfig, args: argparse.Namespace) -> dict[str, object]:
    output_dir = resolve_output_dir(config, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_spatial_adata(config.st_h5ad)
    coords = np.asarray(adata.obsm["spatial"])
    spot_names = adata.obs_names.astype(str).tolist()
    require_existing(config.composition_csv)
    spot_cell_expr_csv = resolve_spot_cell_expr_csv(config, required=True)

    print(f"[{config.key}] Rerunning observed Stage 3...")
    observed_lr_csv = run_observed_cellcom(
        config=config,
        output_dir=output_dir,
        spot_cell_expr_csv=spot_cell_expr_csv,
        device=args.device,
    )
    observed_lr_df = read_lr_table(observed_lr_csv)

    print(f"[{config.key}] Running observed analysis...")
    observed_results = analyze_lr_table(
        lr_df=observed_lr_df,
        spot_names=spot_names,
        coords=coords,
        config=config,
        top_k=args.top_k,
        min_pair_occurrence=args.min_pair_occurrence,
        analysis_seed=args.seed,
    )

    save_outputs(
        config=config,
        output_dir=output_dir,
        observed_results=observed_results,
        top_k=args.top_k,
        min_pair_occurrence=args.min_pair_occurrence,
        analysis_seed=args.seed,
    )

    observed_summary = observed_results["group_summary"].copy()
    observed_summary.insert(0, "dataset", config.key)
    observed_pair_metrics = observed_results["pair_metrics"].copy()
    observed_pair_metrics.insert(0, "dataset", config.key)
    observed_top_pairs = observed_results["top_pairs"].copy()
    observed_top_pairs.insert(0, "dataset", config.key)
    expression_pair_stats = observed_results["expression_pair_stats"].copy()
    expression_pair_stats.insert(0, "dataset", config.key)
    expression_summary = observed_results["expression_summary"].copy()
    expression_summary.insert(0, "dataset", config.key)

    note = "Observed-only workflow with Moran, cell-type specificity, and expression-attention decoupling."
    return {
        "dataset": config.key,
        "output_dir": output_dir,
        "observed_summary": observed_summary,
        "observed_pair_metrics": observed_pair_metrics,
        "observed_top_pairs": observed_top_pairs,
        "expression_pair_stats": expression_pair_stats,
        "expression_summary": expression_summary,
        "status_row": {
            "dataset": config.key,
            "output_dir": str(output_dir),
            "status": "ok",
            "note": note,
        },
    }


def main() -> None:
    args = parse_args()

    if uses_explicit_config(args):
        config = build_explicit_config(args)
        result = run_dataset(config, args)
        print(f"[{config.key}] Outputs saved to: {result['output_dir']}")
        return

    dataset_keys = resolve_dataset_keys(args.dataset)
    configs = [apply_runtime_overrides(DATASETS[dataset_key], args) for dataset_key in dataset_keys]

    dataset_summaries: list[pd.DataFrame] = []
    dataset_pair_metrics: list[pd.DataFrame] = []
    dataset_top_pairs: list[pd.DataFrame] = []
    dataset_expression_pair_stats: list[pd.DataFrame] = []
    dataset_expression_summaries: list[pd.DataFrame] = []
    run_status_rows: list[dict[str, object]] = []

    for config in configs:
        try:
            result = run_dataset(config, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[{config.key}] FAILED: {exc}")
            traceback.print_exc()
            run_status_rows.append(
                {
                    "dataset": config.key,
                    "output_dir": str(resolve_output_dir(config, args.output_root)),
                    "status": "failed",
                    "note": str(exc),
                }
            )
            continue

        dataset_summaries.append(result["observed_summary"])
        dataset_pair_metrics.append(result["observed_pair_metrics"])
        dataset_top_pairs.append(result["observed_top_pairs"])
        dataset_expression_pair_stats.append(result["expression_pair_stats"])
        dataset_expression_summaries.append(result["expression_summary"])
        run_status_rows.append(result["status_row"])
        print(f"[{config.key}] Outputs saved to: {result['output_dir']}")

    combined_dir = (
        args.output_root / "_combined"
        if args.output_root is not None
        else DATA_ROOT / "ccc_analysis_summary"
    )
    save_combined_outputs(
        combined_dir=combined_dir,
        dataset_summaries=dataset_summaries,
        dataset_pair_metrics=dataset_pair_metrics,
        dataset_top_pairs=dataset_top_pairs,
        dataset_expression_pair_stats=dataset_expression_pair_stats,
        dataset_expression_summaries=dataset_expression_summaries,
        run_status_rows=run_status_rows,
    )
    print(f"Combined summaries saved to: {combined_dir}")


if __name__ == "__main__":
    main()
