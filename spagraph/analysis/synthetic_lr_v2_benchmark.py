"""Fully synthetic LR re-ranking benchmark with matched spatial controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph


CELL_TYPES = ["Tumor", "CAF", "Tcell", "DC", "Endothelial", "Pericyte"]
PAIRINGS = [
    ("CAF", "Tumor"),
    ("DC", "Tcell"),
    ("Endothelial", "Pericyte"),
    ("Tumor", "CAF"),
]


@dataclass
class SyntheticV2Config:
    output: str = "results/synthetic_lr_v2"
    grid_side: int = 22
    n_families: int = 8
    n_global_decoys: int = 6
    n_random_decoys: int = 10
    epochs: int = 30
    seed: int = 42
    device: str = "cuda"
    run_model: bool = True
    ablation_no_lr_identity: bool = False


def _grid(grid_side: int, rng: np.random.Generator) -> np.ndarray:
    axis = np.linspace(0, 1, grid_side, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords + rng.normal(0, 0.003, size=coords.shape)


def _composition(coords: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    x, y = coords[:, 0], coords[:, 1]
    centers = {
        "Tumor": (0.30, 0.48),
        "CAF": (0.68, 0.48),
        "Tcell": (0.28, 0.78),
        "DC": (0.20, 0.72),
        "Endothelial": (0.76, 0.76),
        "Pericyte": (0.82, 0.68),
    }
    weights = []
    for cell_type in CELL_TYPES:
        cx, cy = centers[cell_type]
        distance2 = (x - cx) ** 2 + (y - cy) ** 2
        weights.append(0.025 + np.exp(-distance2 / 0.055))
    values = np.column_stack(weights)
    values *= rng.lognormal(0, 0.08, size=values.shape)
    values /= values.sum(axis=1, keepdims=True)
    names = [f"spot{i:04d}" for i in range(len(coords))]
    return pd.DataFrame(values, index=names, columns=CELL_TYPES)


def _compact_indices(
    coords: np.ndarray,
    center: np.ndarray,
    n: int,
) -> np.ndarray:
    distance = np.sqrt(((coords - center) ** 2).sum(axis=1))
    return np.argsort(distance)[:n]


def _spread_indices(
    coords: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = rng.permutation(len(coords))
    selected = [int(candidates[0])]
    min_distance = np.full(len(coords), np.inf)
    for _ in range(1, n):
        last = coords[selected[-1]]
        distance = np.sqrt(((coords - last) ** 2).sum(axis=1))
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1
        selected.append(int(np.argmax(min_distance)))
    return np.asarray(selected, dtype=int)


def _row_indices(
    spots: Iterable[int],
    cell_type: str,
    n_spots: int,
) -> np.ndarray:
    cell_idx = CELL_TYPES.index(cell_type)
    return np.asarray(list(spots), dtype=int) * len(CELL_TYPES) + cell_idx


def _spatial_dispersion(coords: np.ndarray, spots: np.ndarray) -> float:
    selected = coords[np.unique(spots)]
    if len(selected) < 2:
        return 0.0
    center = selected.mean(axis=0)
    return float(np.sqrt(((selected - center) ** 2).sum(axis=1)).mean())


def _neighbor_overlap(
    source_spots: np.ndarray,
    target_spots: np.ndarray,
    knn: np.ndarray,
) -> int:
    return int(knn[np.ix_(np.unique(source_spots), np.unique(target_spots))].sum())


def _assign(
    artificial: np.ndarray,
    gene_to_idx: Dict[str, int],
    gene: str,
    rows: np.ndarray,
    values: np.ndarray,
) -> None:
    artificial[rows, gene_to_idx[gene]] = values


def _make_truth_and_artificial_expression(
    coords: np.ndarray,
    config: SyntheticV2Config,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[str], np.ndarray, pd.DataFrame]:
    n_spots = len(coords)
    n_rows = n_spots * len(CELL_TYPES)
    truth_rows = []
    assignments: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit_rows = []
    knn = kneighbors_graph(
        coords, n_neighbors=6, mode="connectivity", include_self=False
    ).toarray()

    centers = np.asarray([
        [0.50, 0.48],
        [0.24, 0.75],
        [0.78, 0.72],
        [0.50, 0.30],
        [0.40, 0.65],
        [0.64, 0.62],
        [0.34, 0.34],
        [0.70, 0.28],
    ])
    genes = []

    for family in range(config.n_families):
        source_type, target_type = PAIRINGS[family % len(PAIRINGS)]
        active_n = 24 if family == config.n_families - 1 else 64 + 4 * (family % 3)
        ligand_values = rng.lognormal(np.log(24 + 2 * family), 0.22, active_n)
        receptor_values = rng.lognormal(np.log(18 + 2 * family), 0.22, active_n)

        local_spots = _compact_indices(coords, centers[family], active_n)
        diffuse_source = _spread_indices(coords, active_n, rng)
        diffuse_target = _spread_indices(coords, active_n, rng)
        separated_source = _compact_indices(coords, np.asarray([0.14, 0.18]), active_n)
        separated_target = _compact_indices(coords, np.asarray([0.86, 0.82]), active_n)

        variants = {
            "local_positive": (local_spots, local_spots, 1),
            "matched_diffuse": (diffuse_source, diffuse_target, 0),
            "matched_separated": (separated_source, separated_target, 0),
        }
        for variant, (source_spots, target_spots, is_positive) in variants.items():
            prefix = {
                "local_positive": "P",
                "matched_diffuse": "D",
                "matched_separated": "S",
            }[variant]
            ligand = f"L{prefix}{family + 1:02d}"
            receptor = f"R{prefix}{family + 1:02d}"
            genes.extend([ligand, receptor])
            source_rows = _row_indices(source_spots, source_type, n_spots)
            target_rows = _row_indices(target_spots, target_type, n_spots)
            assignments[ligand] = (
                source_rows,
                rng.permutation(ligand_values),
            )
            assignments[receptor] = (
                target_rows,
                rng.permutation(receptor_values),
            )
            truth_rows.append({
                "ligand": ligand,
                "receptor": receptor,
                "lr_pair": f"{ligand}_{receptor}",
                "is_positive": is_positive,
                "pattern": variant,
                "family": family + 1,
                "source_celltype": source_type,
                "target_celltype": target_type,
            })
            audit_rows.append({
                "lr_pair": f"{ligand}_{receptor}",
                "pattern": variant,
                "family": family + 1,
                "ligand_active_n": len(source_rows),
                "receptor_active_n": len(target_rows),
                "ligand_mean_cp10k": float(ligand_values.mean()),
                "receptor_mean_cp10k": float(receptor_values.mean()),
                "source_dispersion": _spatial_dispersion(coords, source_spots),
                "target_dispersion": _spatial_dispersion(coords, target_spots),
                "neighbor_overlap": _neighbor_overlap(
                    source_spots, target_spots, knn
                ),
            })

    all_spots = np.arange(n_spots)
    global_n = round(0.85 * n_spots)
    for index in range(config.n_global_decoys):
        source_type, target_type = PAIRINGS[index % len(PAIRINGS)]
        ligand = f"LG{index + 1:02d}"
        receptor = f"RG{index + 1:02d}"
        genes.extend([ligand, receptor])
        source_spots = rng.choice(all_spots, global_n, replace=False)
        target_spots = rng.choice(all_spots, global_n, replace=False)
        ligand_values = rng.lognormal(np.log(32), 0.25, global_n)
        receptor_values = rng.lognormal(np.log(25), 0.25, global_n)
        assignments[ligand] = (
            _row_indices(source_spots, source_type, n_spots),
            ligand_values,
        )
        assignments[receptor] = (
            _row_indices(target_spots, target_type, n_spots),
            receptor_values,
        )
        truth_rows.append({
            "ligand": ligand,
            "receptor": receptor,
            "lr_pair": f"{ligand}_{receptor}",
            "is_positive": 0,
            "pattern": "global_high_coverage",
            "family": np.nan,
            "source_celltype": source_type,
            "target_celltype": target_type,
        })
        audit_rows.append({
            "lr_pair": f"{ligand}_{receptor}",
            "pattern": "global_high_coverage",
            "family": np.nan,
            "ligand_active_n": global_n,
            "receptor_active_n": global_n,
            "ligand_mean_cp10k": float(ligand_values.mean()),
            "receptor_mean_cp10k": float(receptor_values.mean()),
            "source_dispersion": _spatial_dispersion(coords, source_spots),
            "target_dispersion": _spatial_dispersion(coords, target_spots),
            "neighbor_overlap": _neighbor_overlap(
                source_spots, target_spots, knn
            ),
        })

    random_n = 64
    for index in range(config.n_random_decoys):
        source_type, target_type = PAIRINGS[index % len(PAIRINGS)]
        ligand = f"LX{index + 1:02d}"
        receptor = f"RX{index + 1:02d}"
        genes.extend([ligand, receptor])
        source_spots = rng.choice(all_spots, random_n, replace=False)
        target_spots = rng.choice(all_spots, random_n, replace=False)
        ligand_values = rng.lognormal(np.log(28), 0.28, random_n)
        receptor_values = rng.lognormal(np.log(21), 0.28, random_n)
        assignments[ligand] = (
            _row_indices(source_spots, source_type, n_spots),
            ligand_values,
        )
        assignments[receptor] = (
            _row_indices(target_spots, target_type, n_spots),
            receptor_values,
        )
        truth_rows.append({
            "ligand": ligand,
            "receptor": receptor,
            "lr_pair": f"{ligand}_{receptor}",
            "is_positive": 0,
            "pattern": "random_background",
            "family": np.nan,
            "source_celltype": source_type,
            "target_celltype": target_type,
        })
        audit_rows.append({
            "lr_pair": f"{ligand}_{receptor}",
            "pattern": "random_background",
            "family": np.nan,
            "ligand_active_n": random_n,
            "receptor_active_n": random_n,
            "ligand_mean_cp10k": float(ligand_values.mean()),
            "receptor_mean_cp10k": float(receptor_values.mean()),
            "source_dispersion": _spatial_dispersion(coords, source_spots),
            "target_dispersion": _spatial_dispersion(coords, target_spots),
            "neighbor_overlap": _neighbor_overlap(
                source_spots, target_spots, knn
            ),
        })

    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    artificial = np.zeros((n_rows, len(genes)), dtype=np.float32)
    for gene, (rows, values) in assignments.items():
        _assign(artificial, gene_to_idx, gene, rows, values)

    truth = pd.DataFrame(truth_rows)
    audit = pd.DataFrame(audit_rows)
    return truth, genes, artificial, audit


def _expression(
    composition: pd.DataFrame,
    coords: np.ndarray,
    lr_genes: list[str],
    artificial: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    background_genes = [f"G{i:03d}" for i in range(120)]
    background = np.empty((len(artificial), len(background_genes)), dtype=np.float32)
    remaining = 1e4 - artificial.sum(axis=1)
    if np.any(remaining <= 0):
        raise ValueError("Artificial LR programs exceed the CP10k row budget")
    region_centers = np.asarray([
        [0.50, 0.48],
        [0.24, 0.75],
        [0.78, 0.72],
        [0.50, 0.30],
        [0.40, 0.65],
        [0.64, 0.62],
        [0.34, 0.34],
        [0.70, 0.28],
    ])
    for row in range(len(background)):
        spot_idx = row // len(CELL_TYPES)
        cell_idx = row % len(CELL_TYPES)
        weights = rng.gamma(0.7, 1.0, len(background_genes))
        marker_start = cell_idx * 10
        weights[marker_start:marker_start + 10] *= 8.0
        for region_idx, center in enumerate(region_centers):
            distance2 = float(((coords[spot_idx] - center) ** 2).sum())
            spatial_factor = 1.0 + 7.0 * np.exp(-distance2 / 0.025)
            region_start = 60 + region_idx * 5
            weights[region_start:region_start + 5] *= spatial_factor
        dropout = rng.random(len(background_genes)) < 0.18
        weights[dropout] = 0
        if weights.sum() == 0:
            weights[0] = 1
        background[row] = weights / weights.sum() * remaining[row]
    values = np.column_stack([artificial, background])
    index = [
        f"{spot}_{cell_type}"
        for spot in composition.index
        for cell_type in CELL_TYPES
    ]
    frame = pd.DataFrame(values, index=index, columns=lr_genes + background_genes)
    frame.index.name = "spot_cell"
    return frame

def _validate_design(
    truth: pd.DataFrame,
    audit: pd.DataFrame,
    expression: pd.DataFrame,
) -> None:
    row_totals = expression.sum(axis=1).to_numpy()
    if not np.allclose(row_totals, 1e4, rtol=1e-5, atol=1e-2):
        raise AssertionError("Expression rows are not exactly CP10k normalized")
    for family, frame in audit.dropna(subset=["family"]).groupby("family"):
        frame = frame.set_index("pattern")
        for column in [
            "ligand_active_n",
            "receptor_active_n",
            "ligand_mean_cp10k",
            "receptor_mean_cp10k",
        ]:
            values = frame.loc[
                ["local_positive", "matched_diffuse", "matched_separated"], column
            ].to_numpy(dtype=float)
            if not np.allclose(values, values[0]):
                raise AssertionError(f"Family {family} is not marginally matched: {column}")
        local_dispersion = frame.loc[
            "local_positive", ["source_dispersion", "target_dispersion"]
        ].mean()
        diffuse_dispersion = frame.loc[
            "matched_diffuse", ["source_dispersion", "target_dispersion"]
        ].mean()
        if not local_dispersion < diffuse_dispersion:
            raise AssertionError(f"Family {family} local program is not more compact")
    if truth["is_positive"].sum() != truth["family"].nunique():
        raise AssertionError("Each matched family must contain exactly one positive")


def generate_synthetic_v2_inputs(config: SyntheticV2Config) -> Dict[str, Path]:
    output = Path(config.output)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    coords = _grid(config.grid_side, rng)
    composition = _composition(coords, rng)
    truth, lr_genes, artificial, audit = _make_truth_and_artificial_expression(
        coords, config, rng
    )
    expression = _expression(composition, coords, lr_genes, artificial, rng)
    _validate_design(truth, audit, expression)

    weighted = np.zeros((len(composition), expression.shape[1]), dtype=np.float32)
    for cell_idx, cell_type in enumerate(CELL_TYPES):
        rows = np.arange(len(composition)) * len(CELL_TYPES) + cell_idx
        weighted += (
            expression.iloc[rows].to_numpy()
            * composition[cell_type].to_numpy()[:, None]
        )
    adata = ad.AnnData(
        X=sparse.csr_matrix(weighted),
        obs=pd.DataFrame(index=composition.index),
        var=pd.DataFrame(index=expression.columns),
    )
    adata.obsm["spatial"] = coords * 400

    paths = {
        "h5ad": data_dir / "synthetic_v2_st.h5ad",
        "composition": data_dir / "synthetic_v2_composition.csv",
        "expression": data_dir / "synthetic_v2_spot_cell_expr.csv",
        "lr_database": data_dir / "synthetic_v2_lr_database.csv",
        "truth": data_dir / "synthetic_v2_truth.csv",
        "audit": data_dir / "synthetic_v2_design_audit.csv",
    }
    adata.write_h5ad(paths["h5ad"])
    composition.to_csv(paths["composition"])
    expression.to_csv(paths["expression"])
    truth[["ligand", "receptor"]].to_csv(paths["lr_database"], index=False)
    truth.to_csv(paths["truth"], index=False)
    audit.to_csv(paths["audit"], index=False)
    return paths


def _rank_metric(
    merged: pd.DataFrame,
    score_column: str,
    prefix: str,
) -> Dict[str, float]:
    ranked = merged.sort_values(score_column, ascending=False).reset_index(drop=True)
    y_true = ranked["is_positive"].to_numpy(dtype=int)
    scores = ranked[score_column].to_numpy(dtype=float)
    metrics = {
        f"{prefix}_auprc": float(average_precision_score(y_true, scores)),
        f"{prefix}_auroc": float(roc_auc_score(y_true, scores)),
    }
    for requested_k in [8, 10, 20]:
        k = min(requested_k, len(ranked))
        metrics[f"{prefix}_precision_at_{requested_k}"] = float(y_true[:k].mean())
        metrics[f"{prefix}_recall_at_{requested_k}"] = float(
            y_true[:k].sum() / y_true.sum()
        )
    return metrics


def evaluate_synthetic_v2(output: Path) -> pd.DataFrame:
    truth = pd.read_csv(output / "data" / "synthetic_v2_truth.csv")
    audit = pd.read_csv(output / "data" / "synthetic_v2_design_audit.csv")
    model_stats = pd.read_csv(output / "cellcom" / "lr_pair_statistics.csv")
    merged = truth.merge(audit, on=["lr_pair", "pattern", "family"], how="left")
    merged = merged.merge(model_stats, on="lr_pair", how="left")
    for column in ["avg_attention_score", "occurrence_count"]:
        floor = merged[column].min() - 1 if merged[column].notna().any() else 0
        merged[column] = merged[column].fillna(floor)
    merged["attention_rank"] = merged["avg_attention_score"].rank(
        ascending=False, method="min"
    )
    merged["abundance_rank"] = merged["occurrence_count"].rank(
        ascending=False, method="min"
    )

    metrics = {}
    metrics.update(_rank_metric(merged, "avg_attention_score", "all_attention"))
    metrics.update(_rank_metric(merged, "occurrence_count", "all_abundance"))
    candidate_pairs = merged.loc[
        ~merged["pattern"].eq("matched_separated")
    ].copy()
    metrics.update(_rank_metric(
        candidate_pairs, "avg_attention_score", "candidate_attention"
    ))
    metrics.update(_rank_metric(
        candidate_pairs, "occurrence_count", "candidate_abundance"
    ))
    paired_rows = []
    for family, frame in merged.dropna(subset=["family"]).groupby("family"):
        indexed = frame.set_index("pattern")
        local = indexed.loc["local_positive"]
        diffuse = indexed.loc["matched_diffuse"]
        separated = indexed.loc["matched_separated"]
        paired_rows.append({
            "family": int(family),
            "local_attention": local["avg_attention_score"],
            "diffuse_attention": diffuse["avg_attention_score"],
            "separated_attention": separated["avg_attention_score"],
            "local_minus_diffuse": (
                local["avg_attention_score"] - diffuse["avg_attention_score"]
            ),
            "local_minus_separated": (
                local["avg_attention_score"] - separated["avg_attention_score"]
            ),
            "local_rank": local["attention_rank"],
            "diffuse_rank": diffuse["attention_rank"],
            "separated_rank": separated["attention_rank"],
        })
    paired = pd.DataFrame(paired_rows)
    metrics["paired_local_beats_diffuse_rate"] = float(
        (paired["local_minus_diffuse"] > 0).mean()
    )
    metrics["paired_local_beats_separated_rate"] = float(
        (paired["local_minus_separated"] > 0).mean()
    )
    metrics["paired_local_vs_diffuse_wilcoxon_p"] = float(
        stats.wilcoxon(
            paired["local_minus_diffuse"], alternative="greater"
        ).pvalue
    )
    metrics["paired_local_vs_separated_wilcoxon_p"] = float(
        stats.wilcoxon(
            paired["local_minus_separated"], alternative="greater"
        ).pvalue
    )
    metrics["global_decoy_median_attention_rank"] = float(
        merged.loc[
            merged["pattern"].eq("global_high_coverage"), "attention_rank"
        ].median()
    )
    metrics["global_decoy_median_abundance_rank"] = float(
        merged.loc[
            merged["pattern"].eq("global_high_coverage"), "abundance_rank"
        ].median()
    )
    merged.sort_values("attention_rank").to_csv(
        output / "synthetic_v2_ranking.csv", index=False
    )
    paired.to_csv(output / "synthetic_v2_paired_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "synthetic_v2_metrics.csv", index=False)
    with open(output / "synthetic_v2_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return merged


def run_synthetic_v2(config: SyntheticV2Config) -> pd.DataFrame:
    output = Path(config.output)
    paths = generate_synthetic_v2_inputs(config)
    with open(output / "benchmark_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    if not config.run_model:
        return pd.read_csv(paths["truth"])

    from spagraph.training.cellcom import run_cellcom

    run_cellcom(
        deconv_dir=str(paths["composition"].parent),
        st_h5ad=str(paths["h5ad"]),
        output_dir=str(output / "cellcom"),
        composition_csv=str(paths["composition"]),
        spot_cell_expr_csv=str(paths["expression"]),
        lr_database_csv=str(paths["lr_database"]),
        n_spot_neighbors=6,
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=1.0,
        lr_score_threshold=0.5,
        min_comm_edges=1,
        use_hvg_for_communication=False,
        allow_same_celltype_comm=False,
        gat_hidden_dims="64,32",
        gat_heads=4,
        gat_dropout=0.2,
        output_dim=32,
        mlp_latent_dim=32,
        mlp_hidden_dims="64,32",
        batch_size=8,
        epochs=config.epochs,
        early_stop_patience=max(5, config.epochs // 3),
        early_stop_min_delta=0.001,
        attention_threshold=0.0,
        export_unified_csv=False,
        export_filtered_csv=False,
        save_lr_scores_csv=True,
        ablation_no_lr_identity=config.ablation_no_lr_identity,
        device=config.device,
        seed=config.seed,
    )
    return evaluate_synthetic_v2(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/synthetic_lr_v2")
    parser.add_argument("--grid-side", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--ablation-no-lr-identity", action="store_true")
    args = parser.parse_args()
    config = SyntheticV2Config(
        output=args.output,
        grid_side=args.grid_side,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        run_model=not args.generate_only,
        ablation_no_lr_identity=args.ablation_no_lr_identity,
    )
    result = run_synthetic_v2(config)
    columns = [
        column
        for column in [
            "lr_pair",
            "pattern",
            "is_positive",
            "avg_attention_score",
            "attention_rank",
            "occurrence_count",
            "abundance_rank",
        ]
        if column in result.columns
    ]
    print(result[columns].sort_values(columns[-1]).to_string(index=False))


if __name__ == "__main__":
    main()
