#!/usr/bin/env python3
"""CCC analysis workflow for the manuscript-priority datasets."""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
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

ANALYSIS_MODE = "moran_only"
PAIR_SIGNAL = "original_lr_score_sum"
MIN_PAIR_OCCURRENCE = 30
OBSERVED_RUN_DIRNAME = "_observed_run"
OBSERVED_CELLCOM_DIRNAME = "cellcom"
CELLCOM_UNIFIED_CSV = "lr_communication.csv"


@dataclass(frozen=True)
class BoundaryConfig:
    region_name: str
    group_a_name: str
    group_a_columns: tuple[str, ...]
    group_b_name: str
    group_b_columns: tuple[str, ...]
    threshold: float = 0.10


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset_dir: Path
    lr_communication: Path | None
    composition_csv: Path
    spot_cell_expr_csv: Path | None
    st_h5ad: Path
    boundary: BoundaryConfig
    output_dir_override: Path | None = None
    n_spot_neighbors: int = 8
    ligand_expr_threshold: float = 3.0
    receptor_expr_threshold: float = 3.0
    allow_same_celltype_comm: bool = True
    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 0
    save_lr_scores_csv: bool = False
    export_unified_csv: bool = False
    export_filtered_csv: bool = True
    seed: int = 42

    @property
    def output_dir(self) -> Path:
        if self.output_dir_override is not None:
            return self.output_dir_override
        return self.dataset_dir / "ccc_analysis"


DATASETS = {
    "GSE243275": DatasetConfig(
        key="GSE243275",
        dataset_dir=DATA_ROOT / "GSE243275",
        lr_communication=DATA_ROOT / "GSE243275" / "lr_communication.csv",
        composition_csv=DATA_ROOT / "GSE243275" / "GSM7782699_ST_composition.csv",
        spot_cell_expr_csv=DATA_ROOT / "GSE243275" / "GSM7782699_ST_spot_cell_expr.csv",
        st_h5ad=DATABASE_ROOT / "GSE243275" / "GSM7782699_ST.h5ad",
        boundary=BoundaryConfig(
            region_name="myoepithelial_boundary",
            group_a_name="dcis",
            group_a_columns=("DCIS 1", "DCIS 2"),
            group_b_name="myoepi",
            group_b_columns=("Myoepi ACTA2+", "Myoepi KRT15+"),
        ),
    ),
    "GSE144236": DatasetConfig(
        key="GSE144236",
        dataset_dir=DATA_ROOT / "GSE144236",
        lr_communication=DATA_ROOT / "GSE144236" / "lr_communication.csv",
        composition_csv=DATA_ROOT / "GSE144236" / "Spatial_composition.csv",
        spot_cell_expr_csv=DATA_ROOT / "GSE144236" / "Spatial_spot_cell_expr.csv",
        st_h5ad=DATABASE_ROOT / "GSE144240" / "GSE144236_P2_ST.h5ad",
        boundary=BoundaryConfig(
            region_name="tumor_stroma_interface",
            group_a_name="tumor",
            group_a_columns=("Epithelial",),
            group_b_name="stroma",
            group_b_columns=("Fibroblast",),
        ),
    ),
    "GSE211956_P3": DatasetConfig(
        key="GSE211956_P3",
        dataset_dir=DATA_ROOT / "GSE211956" / "P3",
        lr_communication=DATA_ROOT / "GSE211956" / "P3" / "lr_communication.csv",
        composition_csv=DATA_ROOT / "GSE211956" / "P3" / "GSE211956_ST_P3_cell_composition.csv",
        spot_cell_expr_csv=None,
        st_h5ad=DATABASE_ROOT / "GSE211956" / "GSE211956_ST_P3.h5ad",
        boundary=BoundaryConfig(
            region_name="tumor_stroma_interface",
            group_a_name="tumor",
            group_a_columns=("Tumour cells",),
            group_b_name="stroma",
            group_b_columns=(
                "Fibro1 (EIF4A3, STAR)",
                "Fibro2 (RBP1, DCN)",
                "Fibro3 (RAMP1, CFD)",
                "Fibro5 (FN1, COL3A1)",
                "Myofibroblasts",
            ),
        ),
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
    parser = argparse.ArgumentParser(description="CCC permutation / Moran / niche workflow.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", *sorted(DATASETS)],
        help="Datasets to run. Use 'all' to run every configured dataset.",
    )
    parser.add_argument("--run-name", default=None, help="Explicit run label for custom input mode.")
    parser.add_argument("--lr-communication-csv", type=Path, default=None)
    parser.add_argument("--composition-csv", type=Path, default=None)
    parser.add_argument("--spot-cell-expr-csv", type=Path, default=None)
    parser.add_argument("--st-h5ad", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--region-name", default=None)
    parser.add_argument("--group-a-name", default=None)
    parser.add_argument("--group-a-columns", nargs="+", default=None)
    parser.add_argument("--group-b-name", default=None)
    parser.add_argument("--group-b-columns", nargs="+", default=None)
    parser.add_argument("--boundary-threshold", type=float, default=0.10)
    parser.add_argument("--n-spot-neighbors", type=int, default=None)
    parser.add_argument("--ligand-expr-threshold", type=float, default=None)
    parser.add_argument("--receptor-expr-threshold", type=float, default=None)
    parser.add_argument("--allow-same-celltype-comm", type=parse_bool, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-lr-scores-csv", type=parse_bool, default=None)
    parser.add_argument("--export-unified-csv", type=parse_bool, default=None)
    parser.add_argument("--export-filtered-csv", type=parse_bool, default=None)
    parser.add_argument("--cellcom-seed", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-perm-runs", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining datasets even if one dataset fails.",
    )
    return parser.parse_args()


def resolve_dataset_keys(selected: list[str]) -> list[str]:
    if "all" in selected:
        return list(DATASETS)
    return selected


def uses_explicit_config(args: argparse.Namespace) -> bool:
    explicit_fields = [
        "run_name",
        "lr_communication_csv",
        "composition_csv",
        "spot_cell_expr_csv",
        "st_h5ad",
        "output_dir",
        "region_name",
        "group_a_name",
        "group_a_columns",
        "group_b_name",
        "group_b_columns",
    ]
    return any(getattr(args, field) is not None for field in explicit_fields)


def build_boundary_from_args(args: argparse.Namespace) -> BoundaryConfig:
    return BoundaryConfig(
        region_name=args.region_name or "ignored",
        group_a_name=args.group_a_name or "ignored",
        group_a_columns=tuple(args.group_a_columns or ()),
        group_b_name=args.group_b_name or "ignored",
        group_b_columns=tuple(args.group_b_columns or ()),
        threshold=args.boundary_threshold,
    )


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
    if args.export_unified_csv is not None:
        overrides["export_unified_csv"] = args.export_unified_csv
    if args.export_filtered_csv is not None:
        overrides["export_filtered_csv"] = args.export_filtered_csv
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
        raise ValueError(
            "Explicit mode is missing required input arguments: " + ", ".join(missing)
        )

    boundary = build_boundary_from_args(args)
    return DatasetConfig(
        key=args.run_name,
        dataset_dir=args.composition_csv.parent,
        lr_communication=args.lr_communication_csv,
        composition_csv=args.composition_csv,
        spot_cell_expr_csv=args.spot_cell_expr_csv,
        st_h5ad=args.st_h5ad,
        boundary=boundary,
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
        save_lr_scores_csv=(
            args.save_lr_scores_csv if args.save_lr_scores_csv is not None else False
        ),
        export_unified_csv=(
            args.export_unified_csv if args.export_unified_csv is not None else False
        ),
        export_filtered_csv=(
            args.export_filtered_csv if args.export_filtered_csv is not None else True
        ),
        seed=args.cellcom_seed if args.cellcom_seed is not None else args.seed,
    )


def resolve_output_dir(config: DatasetConfig, output_root: Path | None) -> Path:
    if output_root is None:
        return config.output_dir
    return output_root / config.key


def slugify(label: str) -> str:
    return re.sub(r"\W+", "_", label.strip().lower()).strip("_")


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
            + ", ".join(str(path.name) for path in matches)
        )
    if required:
        if config.spot_cell_expr_csv is not None:
            raise FileNotFoundError(
                f"{config.key} is missing its configured Stage 2 spot-cell expression file: "
                f"{config.spot_cell_expr_csv}"
            )
        raise FileNotFoundError(
            f"{config.key} is missing a Stage 2 '*_spot_cell_expr.csv' file under {config.dataset_dir}. "
            "Rerun deconvolution with save_reconstructed_genes=True before permutation reruns."
        )
    return None


def ensure_permutation_inputs(config: DatasetConfig) -> None:
    resolve_spot_cell_expr_csv(config, required=True)


def read_lr_table(csv_path: Path) -> pd.DataFrame:
    require_existing(csv_path)
    try:
        return pd.read_csv(csv_path, usecols=list(EXPECTED_LR_COLUMNS))
    except ValueError as exc:
        raise ValueError(f"{csv_path} is missing required columns from {sorted(EXPECTED_LR_COLUMNS)}") from exc


def resolve_cellcom_lr_csv(cellcom_output_dir: Path) -> Path | None:
    unified_csv = cellcom_output_dir / CELLCOM_UNIFIED_CSV
    if unified_csv.exists():
        return unified_csv
    return None


def load_inputs(config: DatasetConfig) -> sc.AnnData:
    require_existing(config.st_h5ad)

    adata = sc.read_h5ad(config.st_h5ad)
    if "spatial" not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in {config.st_h5ad}")
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
    return grouped.sort_values(
        ["attention_mean", "occurrence_count", "attention_sum", "lr_pair"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def filter_candidate_pairs(pair_stats: pd.DataFrame, min_pair_occurrence: int) -> pd.DataFrame:
    filtered = pair_stats.loc[pair_stats["occurrence_count"] >= min_pair_occurrence].copy()
    if filtered.empty:
        raise ValueError(
            f"No LR pairs satisfy occurrence_count >= {min_pair_occurrence}. "
            "Lower the threshold or inspect the Stage 3 communication export."
        )
    return filtered.reset_index(drop=True)


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
) -> np.ndarray:
    values = np.zeros(len(spot_index), dtype=float)
    pair_df = lr_df.loc[
        lr_df["lr_pair"] == lr_pair,
        ["src_spot_barcode", "dst_spot_barcode", "original_lr_score"],
    ]
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


def build_boundary_mask(
    composition: pd.DataFrame,
    adjacency: np.ndarray,
    boundary: BoundaryConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    required_columns = (*boundary.group_a_columns, *boundary.group_b_columns)
    for column in required_columns:
        if column not in composition.columns:
            raise ValueError(f"Required composition column is missing: {column}")

    group_a_score = composition.loc[:, boundary.group_a_columns].sum(axis=1)
    group_b_score = composition.loc[:, boundary.group_b_columns].sum(axis=1)

    group_a_mask = (group_a_score >= boundary.threshold) & (group_a_score >= group_b_score)
    group_b_mask = (group_b_score >= boundary.threshold) & (group_b_score > group_a_score)

    has_group_b_neighbor = adjacency @ group_b_mask.to_numpy(dtype=float) > 0
    has_group_a_neighbor = adjacency @ group_a_mask.to_numpy(dtype=float) > 0
    boundary_mask = (group_a_mask.to_numpy() & has_group_b_neighbor) | (group_b_mask.to_numpy() & has_group_a_neighbor)

    if not boundary_mask.any():
        raise ValueError(
            f"{boundary.region_name} boundary mask is empty. "
            f"Check the configured columns or threshold ({boundary.threshold})."
        )

    group_a_slug = slugify(boundary.group_a_name)
    group_b_slug = slugify(boundary.group_b_name)
    boundary_df = pd.DataFrame(
        {
            "spot_barcode": composition.index.astype(str),
            f"{group_a_slug}_score": group_a_score.to_numpy(),
            f"{group_b_slug}_score": group_b_score.to_numpy(),
            f"is_{group_a_slug}_dominant": group_a_mask.to_numpy(),
            f"is_{group_b_slug}_dominant": group_b_mask.to_numpy(),
            "is_boundary": boundary_mask,
            "region_name": boundary.region_name,
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
) -> pd.DataFrame:
    rows = []
    for record in top_pairs_df.itertuples(index=False):
        values = build_pair_spot_map(
            lr_df=lr_df,
            lr_pair=record.lr_pair,
            spot_index=spot_index,
        )
        rows.append(
            {
                "ranking_type": record.ranking_type,
                "rank": record.rank,
                "lr_pair": record.lr_pair,
                "occurrence_count": record.occurrence_count,
                "attention_mean": record.attention_mean,
                "pair_activity_sum": float(values.sum()),
                "moran_i": morans_i(values, adjacency),
            }
        )
    return pd.DataFrame(rows)


def compare_groups(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric = "moran_i"
    attention = metrics_df.loc[metrics_df["ranking_type"] == "attention", metric].dropna()
    frequency = metrics_df.loc[metrics_df["ranking_type"] == "frequency", metric].dropna()
    p_value = float("nan")
    if len(attention) > 0 and len(frequency) > 0:
        p_value = float(mannwhitneyu(attention, frequency, alternative="two-sided").pvalue)

    return pd.DataFrame(
        [
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
        ]
    )


def save_boxplot(metrics_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Path) -> None:
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


def save_permutation_figure(perm_summary: pd.DataFrame, observed_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    observed_map = observed_summary.set_index("metric")
    colors = {"attention": "#c0392b", "frequency": "#2980b9"}
    if perm_summary.empty:
        ax.text(0.5, 0.5, "No permutations requested", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(
            perm_summary["mean_moran_attention"].dropna(),
            bins=20,
            alpha=0.55,
            color=colors["attention"],
            label="perm attention",
        )
        ax.hist(
            perm_summary["mean_moran_frequency"].dropna(),
            bins=20,
            alpha=0.55,
            color=colors["frequency"],
            label="perm frequency",
        )
        if "moran_i" in observed_map.index:
            ax.axvline(
                observed_map.loc["moran_i", "attention_mean"],
                color=colors["attention"],
                linestyle="--",
                linewidth=2,
                label="obs attention",
            )
            ax.axvline(
                observed_map.loc["moran_i", "frequency_mean"],
                color=colors["frequency"],
                linestyle="--",
                linewidth=2,
                label="obs frequency",
            )
        ax.set_title("Mean Moran's I")
        ax.set_xlabel("Value")
        ax.set_ylabel("Permutation count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_lr_table(
    lr_df: pd.DataFrame,
    spot_names: list[str],
    coords: np.ndarray,
    config: DatasetConfig,
    top_k: int,
    min_pair_occurrence: int = MIN_PAIR_OCCURRENCE,
) -> dict[str, pd.DataFrame]:
    valid_spots = set(spot_names)
    lr_df = filter_known_spots(lr_df, valid_spots)
    adjacency = build_adjacency(coords, config.n_spot_neighbors)
    spot_index = {barcode: idx for idx, barcode in enumerate(spot_names)}

    pair_stats = filter_candidate_pairs(summarize_pairs(lr_df), min_pair_occurrence=min_pair_occurrence)
    top_attention = top_pairs(pair_stats, top_k=top_k, ranking_type="attention")
    top_frequency = top_pairs(pair_stats, top_k=top_k, ranking_type="frequency")
    observed_top_pairs = pd.concat([top_attention, top_frequency], ignore_index=True)
    metrics_df = evaluate_top_pairs(lr_df, observed_top_pairs, spot_index, adjacency)
    summary_df = compare_groups(metrics_df)

    return {
        "top_pairs": observed_top_pairs,
        "pair_metrics": metrics_df,
        "group_summary": summary_df,
    }


def write_permuted_h5ad(source_h5ad: Path, output_h5ad: Path, permutation_seed: int) -> None:
    adata = sc.read_h5ad(source_h5ad)
    coords = np.asarray(adata.obsm["spatial"]).copy()
    rng = np.random.default_rng(permutation_seed)
    adata.obsm["spatial"] = coords[rng.permutation(coords.shape[0])]
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_h5ad)


def link_or_copy_file(source: Path, destination: Path) -> Path:
    require_existing(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def stage_cellcom_inputs(config: DatasetConfig, output_dir: Path) -> tuple[Path, Path]:
    staged_dir = output_dir / "_cellcom_inputs"
    spot_cell_expr_csv = resolve_spot_cell_expr_csv(config, required=True)
    link_or_copy_file(config.composition_csv, staged_dir / config.composition_csv.name)
    staged_spot_expr = link_or_copy_file(spot_cell_expr_csv, staged_dir / spot_cell_expr_csv.name)
    return staged_dir, staged_spot_expr


def run_cellcom_rerun(
    config: DatasetConfig,
    staged_deconv_dir: Path,
    spot_cell_expr_csv: Path,
    st_h5ad: Path,
    cellcom_output_dir: Path,
    device: str,
) -> Path:
    spagraph.cellcom(
        deconv_dir=str(staged_deconv_dir),
        st_h5ad=str(st_h5ad),
        output_dir=str(cellcom_output_dir),
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
    staged_deconv_dir: Path,
    spot_cell_expr_csv: Path,
    device: str,
) -> Path:
    observed_output_dir = output_dir / OBSERVED_RUN_DIRNAME / OBSERVED_CELLCOM_DIRNAME
    return run_cellcom_rerun(
        config=config,
        staged_deconv_dir=staged_deconv_dir,
        spot_cell_expr_csv=spot_cell_expr_csv,
        st_h5ad=config.st_h5ad,
        cellcom_output_dir=observed_output_dir,
        device=device,
    )


def run_single_permutation(
    config: DatasetConfig,
    output_dir: Path,
    permutation_index: int,
    base_seed: int,
    top_k: int,
    keep_perm_runs: bool,
    spot_names: list[str],
    staged_deconv_dir: Path,
    spot_cell_expr_csv: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm_root = output_dir / "_perm_runs" / f"perm_{permutation_index:03d}"
    perm_h5ad = perm_root / "permuted_spatial.h5ad"
    perm_output = perm_root / "cellcom"
    write_permuted_h5ad(config.st_h5ad, perm_h5ad, base_seed + permutation_index)
    perm_lr_csv = run_cellcom_rerun(
        config=config,
        staged_deconv_dir=staged_deconv_dir,
        spot_cell_expr_csv=spot_cell_expr_csv,
        st_h5ad=perm_h5ad,
        cellcom_output_dir=perm_output,
        device=device,
    )

    perm_lr_df = read_lr_table(perm_lr_csv)
    perm_adata = sc.read_h5ad(perm_h5ad)
    perm_adata.obs_names = perm_adata.obs_names.astype(str)
    perm_results = analyze_lr_table(
        lr_df=perm_lr_df,
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
    }
    summary_df = pd.DataFrame([summary_row])
    perm_top_pairs = perm_results["top_pairs"].copy()
    perm_top_pairs.insert(0, "permutation_index", permutation_index)

    if not keep_perm_runs:
        shutil.rmtree(perm_root, ignore_errors=True)
        gc.collect()

    return summary_df, perm_top_pairs


def empty_permutation_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "permutation_index",
            "mean_moran_attention",
            "mean_moran_frequency",
        ]
    )


def empty_permutation_top_pairs() -> pd.DataFrame:
    return pd.DataFrame(
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


def save_run_info(
    config: DatasetConfig,
    output_dir: Path,
    top_k: int,
    requested_n_permutations: int,
    executed_n_permutations: int,
    note: str,
) -> None:
    try:
        resolved_spot_cell_expr = resolve_spot_cell_expr_csv(config, required=False)
    except FileNotFoundError as exc:
        resolved_spot_cell_expr = f"unresolved ({exc})"
    info_lines = [
        f"analysis_mode={ANALYSIS_MODE}",
        f"dataset={config.key}",
        f"dataset_dir={config.dataset_dir}",
        f"legacy_lr_communication_input={config.lr_communication if config.lr_communication is not None else 'ignored'}",
        f"composition_csv={config.composition_csv}",
        f"spot_cell_expr_csv={resolved_spot_cell_expr if resolved_spot_cell_expr is not None else 'missing'}",
        f"st_h5ad={config.st_h5ad}",
        f"output_dir={output_dir}",
        f"observed_source={OBSERVED_RUN_DIRNAME}/{OBSERVED_CELLCOM_DIRNAME}/{CELLCOM_UNIFIED_CSV}",
        f"pair_signal={PAIR_SIGNAL}",
        f"min_pair_occurrence={MIN_PAIR_OCCURRENCE}",
        "boundary_args_ignored=true",
        f"region_name={config.boundary.region_name}",
        f"group_a={config.boundary.group_a_name}:{','.join(config.boundary.group_a_columns)}",
        f"group_b={config.boundary.group_b_name}:{','.join(config.boundary.group_b_columns)}",
        f"boundary_threshold={config.boundary.threshold}",
        f"top_k={top_k}",
        f"requested_n_permutations={requested_n_permutations}",
        f"executed_n_permutations={executed_n_permutations}",
        f"n_spot_neighbors={config.n_spot_neighbors}",
    ]
    if note:
        info_lines.append(f"note={note}")
    (output_dir / "analysis_run_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")


def save_outputs(
    config: DatasetConfig,
    output_dir: Path,
    observed_results: dict[str, pd.DataFrame],
    permutation_summary: pd.DataFrame,
    permutation_top_pairs: pd.DataFrame,
    top_k: int,
    requested_n_permutations: int,
    executed_n_permutations: int,
    note: str,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    observed_results["top_pairs"].to_csv(output_dir / "observed_top_pairs.csv", index=False)
    observed_results["pair_metrics"].to_csv(output_dir / "observed_pair_metrics.csv", index=False)
    observed_results["group_summary"].to_csv(output_dir / "observed_group_summary.csv", index=False)
    permutation_summary.to_csv(output_dir / "permutation_summary.csv", index=False)
    permutation_top_pairs.to_csv(output_dir / "permutation_top_pairs.csv", index=False)
    save_run_info(config, output_dir, top_k, requested_n_permutations, executed_n_permutations, note)

    stale_paths = [
        output_dir / "boundary_spots.csv",
        figures_dir / "boundary_enrichment_attention_vs_frequency.pdf",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()

    save_boxplot(
        observed_results["pair_metrics"],
        "moran_i",
        f"{config.key}: Moran's I for top LR pairs",
        "Moran's I",
        figures_dir / "moran_attention_vs_frequency.pdf",
    )

    permutation_figure_path = figures_dir / "permutation_null_distribution.pdf"
    if permutation_summary.empty:
        if permutation_figure_path.exists():
            permutation_figure_path.unlink()
    else:
        save_permutation_figure(
            permutation_summary,
            observed_results["group_summary"],
            permutation_figure_path,
        )


def save_combined_outputs(
    combined_dir: Path,
    dataset_summaries: list[pd.DataFrame],
    dataset_pair_metrics: list[pd.DataFrame],
    dataset_top_pairs: list[pd.DataFrame],
    run_status_rows: list[dict[str, object]],
) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    if dataset_summaries:
        pd.concat(dataset_summaries, ignore_index=True).to_csv(combined_dir / "combined_observed_group_summary.csv", index=False)
    if dataset_pair_metrics:
        pd.concat(dataset_pair_metrics, ignore_index=True).to_csv(combined_dir / "combined_observed_pair_metrics.csv", index=False)
    if dataset_top_pairs:
        pd.concat(dataset_top_pairs, ignore_index=True).to_csv(combined_dir / "combined_observed_top_pairs.csv", index=False)
    pd.DataFrame(run_status_rows).to_csv(combined_dir / "dataset_run_status.csv", index=False)


def run_dataset(config: DatasetConfig, args: argparse.Namespace) -> dict[str, object]:
    output_dir = resolve_output_dir(config, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    executed_n_permutations = args.n_permutations
    note = (
        "Moran-only workflow: observed is rerun under _observed_run/cellcom, "
        "boundary arguments are ignored, and Moran maps use summed original_lr_score."
    )
    adata = load_inputs(config)
    coords = np.asarray(adata.obsm["spatial"])
    spot_names = adata.obs_names.astype(str).tolist()
    staged_deconv_dir, spot_cell_expr_csv = stage_cellcom_inputs(config, output_dir)

    print(f"[{config.key}] Rerunning observed Stage 3...")
    observed_lr_csv = run_observed_cellcom(
        config=config,
        output_dir=output_dir,
        staged_deconv_dir=staged_deconv_dir,
        spot_cell_expr_csv=spot_cell_expr_csv,
        device=args.device,
    )
    observed_lr_df = read_lr_table(observed_lr_csv)

    print(f"[{config.key}] Running observed Moran analysis...")
    observed_results = analyze_lr_table(
        lr_df=observed_lr_df,
        spot_names=spot_names,
        coords=coords,
        config=config,
        top_k=args.top_k,
    )

    permutation_summaries: list[pd.DataFrame] = []
    permutation_top_pair_frames: list[pd.DataFrame] = []
    for permutation_index in range(1, executed_n_permutations + 1):
        print(f"[{config.key}] Permutation {permutation_index}/{executed_n_permutations}")
        summary_df, top_pairs_df = run_single_permutation(
            config=config,
            output_dir=output_dir,
            permutation_index=permutation_index,
            base_seed=args.seed,
            top_k=args.top_k,
            keep_perm_runs=args.keep_perm_runs,
            spot_names=spot_names,
            staged_deconv_dir=staged_deconv_dir,
            spot_cell_expr_csv=spot_cell_expr_csv,
            device=args.device,
        )
        permutation_summaries.append(summary_df)
        permutation_top_pair_frames.append(top_pairs_df)

    permutation_summary = (
        pd.concat(permutation_summaries, ignore_index=True)
        if permutation_summaries
        else empty_permutation_summary()
    )
    permutation_top_pairs = (
        pd.concat(permutation_top_pair_frames, ignore_index=True)
        if permutation_top_pair_frames
        else empty_permutation_top_pairs()
    )

    save_outputs(
        config=config,
        output_dir=output_dir,
        observed_results=observed_results,
        permutation_summary=permutation_summary,
        permutation_top_pairs=permutation_top_pairs,
        top_k=args.top_k,
        requested_n_permutations=args.n_permutations,
        executed_n_permutations=executed_n_permutations,
        note=note,
    )

    observed_summary = observed_results["group_summary"].copy()
    observed_summary.insert(0, "dataset", config.key)
    observed_pair_metrics = observed_results["pair_metrics"].copy()
    observed_pair_metrics.insert(0, "dataset", config.key)
    observed_top_pairs = observed_results["top_pairs"].copy()
    observed_top_pairs.insert(0, "dataset", config.key)

    return {
        "dataset": config.key,
        "output_dir": output_dir,
        "observed_summary": observed_summary,
        "observed_pair_metrics": observed_pair_metrics,
        "observed_top_pairs": observed_top_pairs,
        "status_row": {
            "dataset": config.key,
            "output_dir": str(output_dir),
            "requested_n_permutations": args.n_permutations,
            "executed_n_permutations": executed_n_permutations,
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
                    "requested_n_permutations": args.n_permutations,
                    "executed_n_permutations": 0,
                    "status": "failed",
                    "note": str(exc),
                }
            )
            continue

        dataset_summaries.append(result["observed_summary"])
        dataset_pair_metrics.append(result["observed_pair_metrics"])
        dataset_top_pairs.append(result["observed_top_pairs"])
        run_status_rows.append(result["status_row"])
        print(f"[{config.key}] Outputs saved to: {result['output_dir']}")

    combined_dir = args.output_root / "_combined" if args.output_root is not None else DATA_ROOT / "ccc_analysis_summary"
    save_combined_outputs(
        combined_dir=combined_dir,
        dataset_summaries=dataset_summaries,
        dataset_pair_metrics=dataset_pair_metrics,
        dataset_top_pairs=dataset_top_pairs,
        run_status_rows=run_status_rows,
    )
    print(f"Combined summaries saved to: {combined_dir}")


if __name__ == "__main__":
    main()
