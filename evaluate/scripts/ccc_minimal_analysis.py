#!/usr/bin/env python3
"""Minimal CCC analysis pipeline for GSE243275.

This script implements three analysis pieces on top of existing Stage 3 CCC
outputs:
1. coordinate permutation negative control,
2. Moran's I comparison for top attention vs top frequency LR pairs,
3. boundary enrichment analysis.

The first working version is intentionally scoped to GSE243275 because this is
the only dataset in the repository that currently exposes the full set of
inputs needed for a clean rerun workflow.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.neighbors import kneighbors_graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
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


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    lr_communication: Path
    composition_csv: Path
    spot_cell_expr_csv: Path
    st_h5ad: Path
    output_dir: Path
    dcis_columns: tuple[str, ...]
    myo_columns: tuple[str, ...]
    n_spot_neighbors: int
    ligand_expr_threshold: float
    receptor_expr_threshold: float
    allow_same_celltype_comm: bool
    epochs: int
    batch_size: int
    seed: int
    boundary_threshold: float = 0.10


DATASETS = {
    "GSE243275": DatasetConfig(
        dataset="GSE243275",
        lr_communication=REPO_ROOT / "spagraph_data" / "evaluate" / "GSE243275" / "lr_communication.csv",
        composition_csv=REPO_ROOT / "spagraph_data" / "evaluate" / "GSE243275" / "GSM7782699_ST_composition.csv",
        spot_cell_expr_csv=REPO_ROOT / "spagraph_data" / "evaluate" / "GSE243275" / "GSM7782699_ST_spot_cell_expr.csv",
        st_h5ad=REPO_ROOT / "spagraph_data" / "database" / "GSE243275" / "GSM7782699_ST.h5ad",
        output_dir=REPO_ROOT / "spagraph_data" / "evaluate" / "GSE243275" / "ccc_analysis",
        dcis_columns=("DCIS 1", "DCIS 2"),
        myo_columns=("Myoepi ACTA2+", "Myoepi KRT15+"),
        n_spot_neighbors=8,
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=3.0,
        allow_same_celltype_comm=True,
        epochs=200,
        batch_size=64,
        seed=42,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CCC analysis pipeline.")
    parser.add_argument("--dataset", default="GSE243275", choices=sorted(DATASETS))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-perm-runs", action="store_true")
    return parser.parse_args()


def require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")


def load_inputs(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    require_existing(config.lr_communication)
    require_existing(config.composition_csv)
    require_existing(config.spot_cell_expr_csv)
    require_existing(config.st_h5ad)

    lr_df = pd.read_csv(config.lr_communication)
    missing_columns = EXPECTED_LR_COLUMNS - set(lr_df.columns)
    if missing_columns:
        raise ValueError(
            f"lr_communication.csv is missing required columns: {sorted(missing_columns)}"
        )

    composition = pd.read_csv(config.composition_csv, index_col=0)
    adata = sc.read_h5ad(config.st_h5ad)
    if "spatial" not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in {config.st_h5ad}")

    return lr_df, composition, adata


def build_adjacency(coords: np.ndarray, k: int) -> np.ndarray:
    knn = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    adjacency = knn.toarray().astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def filter_known_spots(
    lr_df: pd.DataFrame,
    valid_spots: set[str],
) -> pd.DataFrame:
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
    return grouped.sort_values(
        ["attention_mean", "occurrence_count", "attention_sum", "lr_pair"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def top_pairs(pair_stats: pd.DataFrame, top_k: int, ranking_type: str) -> pd.DataFrame:
    if ranking_type == "attention":
        ranked = pair_stats.sort_values(
            ["attention_mean", "occurrence_count", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    elif ranking_type == "frequency":
        ranked = pair_stats.sort_values(
            ["occurrence_count", "attention_mean", "attention_sum", "lr_pair"],
            ascending=[False, False, False, True],
        )
    else:
        raise ValueError(f"Unknown ranking type: {ranking_type}")

    result = ranked.head(top_k).copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "ranking_type", ranking_type)
    return result


def build_pair_spot_map(
    lr_df: pd.DataFrame,
    lr_pair: str,
    spot_index: dict[str, int],
    mode: str,
) -> np.ndarray:
    values = np.zeros(len(spot_index), dtype=float)
    pair_df = lr_df.loc[lr_df["lr_pair"] == lr_pair, ["src_spot_barcode", "dst_spot_barcode", "attention_score"]]
    if pair_df.empty:
        return values

    if mode == "attention":
        src_contrib = pair_df.groupby("src_spot_barcode")["attention_score"].sum()
        dst_contrib = pair_df.groupby("dst_spot_barcode")["attention_score"].sum()
    elif mode == "frequency":
        src_contrib = pair_df.groupby("src_spot_barcode").size()
        dst_contrib = pair_df.groupby("dst_spot_barcode").size()
    else:
        raise ValueError(f"Unknown map mode: {mode}")

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


def build_boundary_mask(
    composition: pd.DataFrame,
    adjacency: np.ndarray,
    config: DatasetConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    for column in (*config.dcis_columns, *config.myo_columns):
        if column not in composition.columns:
            raise ValueError(f"Required composition column is missing: {column}")

    dcis_score = composition.loc[:, config.dcis_columns].sum(axis=1)
    myo_score = composition.loc[:, config.myo_columns].sum(axis=1)

    dcis_mask = (dcis_score >= config.boundary_threshold) & (dcis_score >= myo_score)
    myo_mask = (myo_score >= config.boundary_threshold) & (myo_score > dcis_score)

    has_myo_neighbor = adjacency @ myo_mask.to_numpy(dtype=float) > 0
    has_dcis_neighbor = adjacency @ dcis_mask.to_numpy(dtype=float) > 0

    boundary_mask = (dcis_mask.to_numpy() & has_myo_neighbor) | (myo_mask.to_numpy() & has_dcis_neighbor)
    if not boundary_mask.any():
        raise ValueError(
            "Boundary mask is empty. Check the composition columns or the boundary threshold."
        )

    boundary_df = pd.DataFrame(
        {
            "spot_barcode": composition.index.astype(str),
            "dcis_score": dcis_score.to_numpy(),
            "myo_score": myo_score.to_numpy(),
            "is_dcis_boundary_candidate": dcis_mask.to_numpy(),
            "is_myo_boundary_candidate": myo_mask.to_numpy(),
            "is_boundary": boundary_mask,
        }
    )
    return boundary_mask, boundary_df


def boundary_enrichment(values: np.ndarray, boundary_mask: np.ndarray, eps: float = 1e-9) -> float:
    inside = values[boundary_mask]
    outside = values[~boundary_mask]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    return float(np.log2((inside.mean() + eps) / (outside.mean() + eps)))


def evaluate_top_pairs(
    lr_df: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    spot_index: dict[str, int],
    adjacency: np.ndarray,
    boundary_mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for record in top_pairs_df.itertuples(index=False):
        values = build_pair_spot_map(
            lr_df=lr_df,
            lr_pair=record.lr_pair,
            spot_index=spot_index,
            mode=record.ranking_type,
        )
        rows.append(
            {
                "ranking_type": record.ranking_type,
                "rank": record.rank,
                "lr_pair": record.lr_pair,
                "pair_activity_sum": float(values.sum()),
                "moran_i": morans_i(values, adjacency),
                "boundary_enrichment": boundary_enrichment(values, boundary_mask),
            }
        )
    return pd.DataFrame(rows)


def compare_groups(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for metric in ("moran_i", "boundary_enrichment"):
        attention = metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna()
        frequency = metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna()
        p_value = float("nan")
        if len(attention) > 0 and len(frequency) > 0:
            p_value = float(mannwhitneyu(attention, frequency, alternative="two-sided").pvalue)

        summary_rows.append(
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
    return pd.DataFrame(summary_rows)


def save_boxplot(
    metrics_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    values = [
        metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna().to_numpy(),
        metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna().to_numpy(),
    ]
    ax.boxplot(values, labels=["attention", "frequency"], patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_permutation_figure(
    perm_summary: pd.DataFrame,
    observed_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_specs = [
        ("mean_moran_attention", "mean_moran_frequency", "moran_i", "Mean Moran's I"),
        (
            "mean_boundary_attention",
            "mean_boundary_frequency",
            "boundary_enrichment",
            "Mean log2 boundary enrichment",
        ),
    ]

    observed_map = observed_summary.set_index("metric")
    colors = {"attention": "#c0392b", "frequency": "#2980b9"}

    for ax, (att_col, freq_col, metric_key, title) in zip(axes, metric_specs):
        if perm_summary.empty:
            ax.text(0.5, 0.5, "No permutations requested", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.hist(
            perm_summary[att_col].dropna(),
            bins=20,
            alpha=0.55,
            color=colors["attention"],
            label="perm attention",
        )
        ax.hist(
            perm_summary[freq_col].dropna(),
            bins=20,
            alpha=0.55,
            color=colors["frequency"],
            label="perm frequency",
        )
        if metric_key in observed_map.index:
            ax.axvline(
                observed_map.loc[metric_key, "attention_mean"],
                color=colors["attention"],
                linestyle="--",
                linewidth=2,
                label="obs attention",
            )
            ax.axvline(
                observed_map.loc[metric_key, "frequency_mean"],
                color=colors["frequency"],
                linestyle="--",
                linewidth=2,
                label="obs frequency",
            )
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Permutation count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_lr_table(
    lr_df: pd.DataFrame,
    composition: pd.DataFrame,
    spot_names: list[str],
    coords: np.ndarray,
    config: DatasetConfig,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    valid_spots = set(spot_names)
    lr_df = filter_known_spots(lr_df, valid_spots)
    composition = composition.reindex(spot_names).fillna(0.0)
    adjacency = build_adjacency(coords, config.n_spot_neighbors)
    boundary_mask, boundary_df = build_boundary_mask(composition, adjacency, config)
    spot_index = {barcode: idx for idx, barcode in enumerate(spot_names)}

    pair_stats = summarize_pairs(lr_df)
    top_attention = top_pairs(pair_stats, top_k=top_k, ranking_type="attention")
    top_frequency = top_pairs(pair_stats, top_k=top_k, ranking_type="frequency")
    observed_top_pairs = pd.concat([top_attention, top_frequency], ignore_index=True)
    metrics_df = evaluate_top_pairs(lr_df, observed_top_pairs, spot_index, adjacency, boundary_mask)
    summary_df = compare_groups(metrics_df)

    return {
        "pair_stats": pair_stats,
        "top_pairs": observed_top_pairs,
        "pair_metrics": metrics_df,
        "group_summary": summary_df,
        "boundary_spots": boundary_df,
    }


def write_permuted_h5ad(
    source_h5ad: Path,
    output_h5ad: Path,
    permutation_seed: int,
) -> None:
    adata = sc.read_h5ad(source_h5ad)
    coords = np.asarray(adata.obsm["spatial"]).copy()
    rng = np.random.default_rng(permutation_seed)
    permuted_indices = rng.permutation(coords.shape[0])
    adata.obsm["spatial"] = coords[permuted_indices]
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_h5ad)


def run_single_permutation(
    config: DatasetConfig,
    permutation_index: int,
    base_seed: int,
    top_k: int,
    keep_perm_runs: bool,
    spot_names: list[str],
    composition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm_root = config.output_dir / "_perm_runs" / f"perm_{permutation_index:03d}"
    perm_h5ad = perm_root / "permuted_spatial.h5ad"
    perm_output = perm_root / "cellcom"
    perm_lr_csv = perm_output / "lr_communication.csv"

    if not perm_lr_csv.exists():
        write_permuted_h5ad(
            source_h5ad=config.st_h5ad,
            output_h5ad=perm_h5ad,
            permutation_seed=base_seed + permutation_index,
        )
        spagraph.cellcom(
            deconv_dir=str(config.composition_csv.parent),
            st_h5ad=str(perm_h5ad),
            output_dir=str(perm_output),
            n_spot_neighbors=config.n_spot_neighbors,
            ligand_expr_threshold=config.ligand_expr_threshold,
            receptor_expr_threshold=config.receptor_expr_threshold,
            allow_same_celltype_comm=config.allow_same_celltype_comm,
            epochs=config.epochs,
            batch_size=config.batch_size,
            seed=config.seed,
            device="cuda",
        )

    perm_lr_df = pd.read_csv(perm_lr_csv)
    perm_adata = sc.read_h5ad(perm_h5ad)
    perm_results = analyze_lr_table(
        lr_df=perm_lr_df,
        composition=composition,
        spot_names=spot_names,
        coords=np.asarray(perm_adata.obsm["spatial"]),
        config=config,
        top_k=top_k,
    )

    summary_row = {
        "permutation_index": permutation_index,
        "mean_moran_attention": perm_results["group_summary"].loc[
            perm_results["group_summary"]["metric"] == "moran_i", "attention_mean"
        ].iloc[0],
        "mean_moran_frequency": perm_results["group_summary"].loc[
            perm_results["group_summary"]["metric"] == "moran_i", "frequency_mean"
        ].iloc[0],
        "mean_boundary_attention": perm_results["group_summary"].loc[
            perm_results["group_summary"]["metric"] == "boundary_enrichment", "attention_mean"
        ].iloc[0],
        "mean_boundary_frequency": perm_results["group_summary"].loc[
            perm_results["group_summary"]["metric"] == "boundary_enrichment", "frequency_mean"
        ].iloc[0],
    }
    summary_df = pd.DataFrame([summary_row])

    perm_top_pairs = perm_results["top_pairs"].copy()
    perm_top_pairs.insert(0, "permutation_index", permutation_index)

    if not keep_perm_runs:
        shutil.rmtree(perm_root, ignore_errors=True)
        gc.collect()

    return summary_df, perm_top_pairs


def save_outputs(
    config: DatasetConfig,
    observed_results: dict[str, pd.DataFrame],
    permutation_summary: pd.DataFrame,
    permutation_top_pairs: pd.DataFrame,
) -> None:
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    observed_results["top_pairs"].to_csv(config.output_dir / "observed_top_pairs.csv", index=False)
    observed_results["pair_metrics"].to_csv(config.output_dir / "observed_pair_metrics.csv", index=False)
    observed_results["group_summary"].to_csv(config.output_dir / "observed_group_summary.csv", index=False)
    observed_results["boundary_spots"].to_csv(config.output_dir / "boundary_spots.csv", index=False)
    permutation_summary.to_csv(config.output_dir / "permutation_summary.csv", index=False)
    permutation_top_pairs.to_csv(config.output_dir / "permutation_top_pairs.csv", index=False)

    save_boxplot(
        metrics_df=observed_results["pair_metrics"],
        metric="moran_i",
        title=f"{config.dataset}: Moran's I for top LR pairs",
        ylabel="Moran's I",
        output_path=figures_dir / "moran_attention_vs_frequency.pdf",
    )
    save_boxplot(
        metrics_df=observed_results["pair_metrics"],
        metric="boundary_enrichment",
        title=f"{config.dataset}: boundary enrichment for top LR pairs",
        ylabel="log2 enrichment",
        output_path=figures_dir / "boundary_enrichment_attention_vs_frequency.pdf",
    )
    save_permutation_figure(
        perm_summary=permutation_summary,
        observed_summary=observed_results["group_summary"],
        output_path=figures_dir / "permutation_null_distribution.pdf",
    )


def main() -> None:
    args = parse_args()
    config = DATASETS[args.dataset]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lr_df, composition, adata = load_inputs(config)
    coords = np.asarray(adata.obsm["spatial"])
    spot_names = adata.obs_names.astype(str).tolist()
    observed_results = analyze_lr_table(
        lr_df=lr_df,
        composition=composition,
        spot_names=spot_names,
        coords=coords,
        config=config,
        top_k=args.top_k,
    )

    permutation_summaries: list[pd.DataFrame] = []
    permutation_top_pair_frames: list[pd.DataFrame] = []
    for permutation_index in range(1, args.n_permutations + 1):
        summary_df, top_pairs_df = run_single_permutation(
            config=config,
            permutation_index=permutation_index,
            base_seed=args.seed,
            top_k=args.top_k,
            keep_perm_runs=args.keep_perm_runs,
            spot_names=spot_names,
            composition=composition,
        )
        permutation_summaries.append(summary_df)
        permutation_top_pair_frames.append(top_pairs_df)

    permutation_summary = (
        pd.concat(permutation_summaries, ignore_index=True)
        if permutation_summaries
        else pd.DataFrame(
            columns=[
                "permutation_index",
                "mean_moran_attention",
                "mean_moran_frequency",
                "mean_boundary_attention",
                "mean_boundary_frequency",
            ]
        )
    )
    permutation_top_pairs = (
        pd.concat(permutation_top_pair_frames, ignore_index=True)
        if permutation_top_pair_frames
        else pd.DataFrame(
            columns=[
                "permutation_index",
                "ranking_type",
                "rank",
                "lr_pair",
                "occurrence_count",
                "attention_mean",
                "attention_sum",
                "original_lr_sum",
            ]
        )
    )

    save_outputs(
        config=config,
        observed_results=observed_results,
        permutation_summary=permutation_summary,
        permutation_top_pairs=permutation_top_pairs,
    )

    print(f"Observed outputs saved to: {config.output_dir}")
    if args.n_permutations:
        print(f"Permutation summaries saved to: {config.output_dir / 'permutation_summary.csv'}")


if __name__ == "__main__":
    main()
