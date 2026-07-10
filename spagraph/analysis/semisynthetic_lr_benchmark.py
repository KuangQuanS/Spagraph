"""Semi-synthetic LR re-ranking benchmark on a real SCC tissue backbone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph


SCENARIOS = {
    "interface": {
        "score_columns": (["Fibroblast"], ["Epithelial"]),
        "sources": [["Fibroblast"]] * 3,
        "targets": [["Epithelial"]] * 3,
        "n_spots": 90,
        "prefix": "INT",
    },
    "immune": {
        "score_columns": (["CD1C", "Mac", "LC"], ["Tcell"]),
        "sources": [["CD1C"], ["Mac"], ["LC"]],
        "targets": [["Tcell"]] * 3,
        "n_spots": 70,
        "prefix": "IMM",
    },
    "perivascular": {
        "score_columns": (["Endothelial Cell"], ["Fibroblast"]),
        "sources": [["Endothelial Cell"]] * 3,
        "targets": [["Fibroblast"]] * 3,
        "n_spots": 70,
        "prefix": "PER",
    },
    "rare": {
        "score_columns": (["PDC", "CLEC9A", "ASDC"], ["Tcell", "Mac"]),
        "sources": [["PDC"], ["CLEC9A"], ["ASDC"]],
        "targets": [["Tcell"], ["Mac"], ["Tcell", "Mac"]],
        "n_spots": 35,
        "prefix": "RAR",
    },
}

CID44971_SCENARIOS = {
    "interface": {
        "score_columns": (["CAFs"], ["Cancer Epithelial"]),
        "sources": [["CAFs"]] * 3,
        "targets": [["Cancer Epithelial"]] * 3,
        "n_spots": 140,
        "prefix": "INT",
    },
    "immune": {
        "score_columns": (["CAFs", "Myeloid"], ["T-cells", "B-cells"]),
        "sources": [["CAFs"], ["Myeloid"], ["Myeloid"]],
        "targets": [["T-cells"], ["B-cells"], ["T-cells", "B-cells"]],
        "n_spots": 110,
        "prefix": "IMM",
    },
    "perivascular": {
        "score_columns": (["Endothelial"], ["PVL", "CAFs"]),
        "sources": [["Endothelial"]] * 3,
        "targets": [["PVL"], ["CAFs"], ["PVL", "CAFs"]],
        "n_spots": 100,
        "prefix": "PER",
    },
    "rare": {
        "score_columns": (["Plasmablasts", "B-cells"], ["T-cells", "Myeloid"]),
        "sources": [["Plasmablasts"], ["B-cells"], ["Plasmablasts", "B-cells"]],
        "targets": [["T-cells"], ["Myeloid"], ["T-cells", "Myeloid"]],
        "n_spots": 50,
        "prefix": "RAR",
    },
}


@dataclass
class SemisyntheticBenchmarkConfig:
    output: str = "results/semisynthetic_lr_benchmark"
    composition_csv: str = "evaluate/data/GSE144236/Spatial_composition.csv"
    spot_cell_expr_csv: str = "evaluate/data/GSE144236/Spatial_spot_cell_expr.csv"
    st_h5ad: str = "spagraph_data/database/GSE144240/GSE144236_P2_ST.h5ad"
    epochs: int = 30
    seed: int = 42
    device: str = "cuda"
    run_model: bool = True
    templates_per_scenario: int = 3
    include_separated_controls: bool = True


def _sum_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(0.0, index=frame.index)
    return frame[available].sum(axis=1)


def _localized_mask(
    score: pd.Series,
    coords: np.ndarray,
    n_spots: int,
) -> np.ndarray:
    values = score.to_numpy(dtype=float)
    positive = np.flatnonzero(values > 0)
    mask = np.zeros(len(values), dtype=bool)
    if positive.size == 0:
        return mask
    seed_idx = positive[np.argmax(values[positive])]
    distances = np.sqrt(((coords - coords[seed_idx]) ** 2).sum(axis=1))
    positive_distances = distances[positive]
    scale = max(float(np.quantile(positive_distances, 0.35)), 1.0)
    localized_score = values * np.exp(-((distances / scale) ** 2))
    selected = np.argsort(localized_score)[::-1][: min(n_spots, positive.size)]
    selected = selected[localized_score[selected] > 0]
    mask[selected] = True
    return mask


def _farthest_mask(mask: np.ndarray, coords: np.ndarray, n_spots: int) -> np.ndarray:
    shifted = np.zeros(len(mask), dtype=bool)
    source_indices = np.flatnonzero(mask)
    if source_indices.size == 0:
        return shifted
    center = coords[source_indices].mean(axis=0)
    distances = np.sqrt(((coords - center) ** 2).sum(axis=1))
    selected = np.argsort(distances)[::-1][:n_spots]
    shifted[selected] = True
    return shifted


def _eligible_rows(
    spot_by_row: np.ndarray,
    celltype_by_row: np.ndarray,
    spot_mask: np.ndarray,
    cell_types: Iterable[str],
) -> np.ndarray:
    return np.flatnonzero(
        np.isin(celltype_by_row, list(cell_types)) & spot_mask[spot_by_row]
    )


def _spread_rows(
    eligible: np.ndarray,
    spot_by_row: np.ndarray,
    coords: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    if len(eligible) <= n_rows:
        return eligible
    candidate_coords = coords[spot_by_row[eligible]]
    selected = [int(np.argmax(candidate_coords[:, 0] + candidate_coords[:, 1]))]
    min_distance = np.full(len(eligible), np.inf)
    for _ in range(1, n_rows):
        last = candidate_coords[selected[-1]]
        distance = np.sqrt(((candidate_coords - last) ** 2).sum(axis=1))
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -1
        selected.append(int(np.argmax(min_distance)))
    return eligible[np.asarray(selected)]


def _candidate_edge_count(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    spot_by_row: np.ndarray,
    knn_mask: np.ndarray,
) -> int:
    if not len(source_rows) or not len(target_rows):
        return 0
    source_spots, source_counts = np.unique(
        spot_by_row[source_rows], return_counts=True
    )
    target_spots, target_counts = np.unique(
        spot_by_row[target_rows], return_counts=True
    )
    adjacency = knn_mask[np.ix_(source_spots, target_spots)]
    return int((adjacency * source_counts[:, None] * target_counts[None, :]).sum())


def _spatial_dispersion(
    selected_rows: np.ndarray,
    spot_by_row: np.ndarray,
    coords: np.ndarray,
) -> float:
    spots = np.unique(spot_by_row[selected_rows])
    if len(spots) < 2:
        return 0.0
    selected_coords = coords[spots]
    center = selected_coords.mean(axis=0)
    return float(np.sqrt(((selected_coords - center) ** 2).sum(axis=1)).mean())


def _edge_count_matched_diffuse_rows(
    source_pool: np.ndarray,
    target_pool: np.ndarray,
    source_count: int,
    target_count: int,
    target_edge_count: int,
    spot_by_row: np.ndarray,
    coords: np.ndarray,
    knn_mask: np.ndarray,
    rng: np.random.Generator,
    n_trials: int = 1200,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    best = None
    source_count = min(source_count, len(source_pool))
    target_count = min(target_count, len(target_pool))
    source_coords = coords[spot_by_row[source_pool]]
    target_coords = coords[spot_by_row[target_pool]]
    for _ in range(n_trials):
        n_clusters = int(rng.integers(2, 8))
        center_candidates = rng.choice(
            source_pool,
            size=min(len(source_pool), max(40, n_clusters * 8)),
            replace=False,
        )
        center_rows = _spread_rows(
            center_candidates, spot_by_row, coords, n_clusters
        )
        centers = coords[spot_by_row[center_rows]]
        source_distance = np.min(
            np.sqrt(((source_coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        target_distance = np.min(
            np.sqrt(((target_coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        pool_multiplier = float(rng.uniform(1.05, 2.25))
        source_candidate_count = min(
            len(source_pool),
            max(source_count, int(np.ceil(source_count * pool_multiplier))),
        )
        target_candidate_count = min(
            len(target_pool),
            max(target_count, int(np.ceil(target_count * pool_multiplier))),
        )
        source_candidates = source_pool[np.argpartition(
            source_distance, source_candidate_count - 1
        )[:source_candidate_count]]
        target_candidates = target_pool[np.argpartition(
            target_distance, target_candidate_count - 1
        )[:target_candidate_count]]
        source = rng.choice(source_candidates, size=source_count, replace=False)
        target = rng.choice(target_candidates, size=target_count, replace=False)
        edge_count = _candidate_edge_count(source, target, spot_by_row, knn_mask)
        relative_error = abs(edge_count - target_edge_count) / max(1, target_edge_count)
        dispersion = (
            _spatial_dispersion(source, spot_by_row, coords)
            + _spatial_dispersion(target, spot_by_row, coords)
        )
        objective = (relative_error, -dispersion)
        if best is None or objective < best[0]:
            best = (objective, source, target, edge_count, dispersion)
    assert best is not None
    return best[1], best[2], int(best[3]), float(best[4])


def _assign_gene_values(
    expression: pd.DataFrame,
    row_sums: pd.Series,
    gene: str,
    selected_rows: np.ndarray,
    cp10k_values: np.ndarray,
) -> None:
    expression[gene] = 0.0
    if len(selected_rows) == 0:
        return
    values = np.resize(np.asarray(cp10k_values, dtype=float), len(selected_rows))
    expression.iloc[selected_rows, expression.columns.get_loc(gene)] = (
        row_sums.iloc[selected_rows].to_numpy() * values / 1e4
    )


def _build_truth_and_expression(
    composition: pd.DataFrame,
    expression: pd.DataFrame,
    coords: np.ndarray,
    seed: int,
    templates_per_scenario: int = 3,
    include_separated_controls: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    spot_to_idx = {spot: idx for idx, spot in enumerate(composition.index)}
    split_index = expression.index.to_series().str.rsplit("_", n=1, expand=True)
    spot_by_row = split_index[0].map(spot_to_idx).to_numpy()
    celltype_by_row = split_index[1].to_numpy()
    valid_rows = ~pd.isna(spot_by_row)
    if not valid_rows.all():
        expression = expression.loc[valid_rows].copy()
        spot_by_row = spot_by_row[valid_rows].astype(int)
        celltype_by_row = celltype_by_row[valid_rows]
    else:
        spot_by_row = spot_by_row.astype(int)
    row_sums = expression.sum(axis=1).clip(lower=1e-6)
    knn_mask = kneighbors_graph(
        coords, n_neighbors=10, mode="connectivity", include_self=False
    ).toarray()

    if "Fibroblast" in composition.columns:
        scenarios = SCENARIOS
        template_source = "Fibroblast"
        template_target = "Epithelial"
        template_gene_specs = [
            ("tnxb_sdc1_sparse", "TNXB", "SDC1"),
            ("tnc_sdc1", "TNC", "SDC1"),
            ("tnc_sdc1_strong", "TNC", "SDC1"),
        ]
    elif "CAFs" in composition.columns:
        scenarios = CID44971_SCENARIOS
        template_source = "CAFs"
        template_target = "Cancer Epithelial"
        template_gene_specs = [
            ("tnxb_sdc1_sparse", "TNXB", "SDC1"),
            ("thbs2_cd44_medium", "THBS2", "CD44"),
            ("fn1_itgb1_high", "FN1", "ITGB1"),
        ]
    else:
        raise ValueError(f"Unsupported composition columns: {composition.columns.tolist()}")

    empirical_cp10k = expression.div(row_sums, axis=0) * 1e4
    empirical_meta = pd.DataFrame(
        {"celltype": celltype_by_row},
        index=expression.index,
    )
    empirical_templates = []
    for template_name, ligand_gene, receptor_gene in template_gene_specs:
        ligand_pool = empirical_cp10k.loc[
            empirical_meta["celltype"].eq(template_source)
            & empirical_cp10k[ligand_gene].gt(3),
            ligand_gene,
        ].to_numpy()
        receptor_pool = empirical_cp10k.loc[
            empirical_meta["celltype"].eq(template_target)
            & empirical_cp10k[receptor_gene].gt(1),
            receptor_gene,
        ].to_numpy()
        if not len(ligand_pool) or not len(receptor_pool):
            raise ValueError(
                f"Empirical template unavailable: {template_name} "
                f"({ligand_gene}, {receptor_gene})"
            )
        empirical_templates.append((template_name, ligand_pool, receptor_pool))

    region_masks: Dict[str, np.ndarray] = {}
    truth_rows: List[dict] = []
    positive_rows: List[dict] = []
    global_mask = np.ones(len(composition), dtype=bool)
    coverages = [0.35, 0.65, 0.85]
    template_specs = [
        (name, ligand_pool, receptor_pool, coverage)
        for (name, ligand_pool, receptor_pool), coverage in zip(
            empirical_templates, coverages
        )
    ]
    if templates_per_scenario <= 1:
        selected_template_specs = [template_specs[1]]
    elif templates_per_scenario == 2:
        selected_template_specs = template_specs[:2]
    else:
        selected_template_specs = template_specs

    for scenario, spec in scenarios.items():
        source_score = _sum_columns(composition, spec["score_columns"][0])
        target_score = _sum_columns(composition, spec["score_columns"][1])
        region_score = np.sqrt(source_score * target_score)
        region_mask = _localized_mask(region_score, coords, spec["n_spots"])
        region_masks[scenario] = region_mask
        prefix = spec["prefix"]

        for i, (template, ligand_pool, receptor_pool, coverage) in enumerate(
            selected_template_specs, start=1
        ):
            source_types = spec["sources"][i - 1]
            target_types = spec["targets"][i - 1]
            source_region_rows = _eligible_rows(
                spot_by_row, celltype_by_row, region_mask, source_types
            )
            target_region_rows = _eligible_rows(
                spot_by_row, celltype_by_row, region_mask, target_types
            )
            source_count = max(1, round(len(source_region_rows) * coverage))
            target_count = max(1, round(len(target_region_rows) * coverage))
            source_selected = rng.choice(
                source_region_rows,
                size=min(source_count, len(source_region_rows)),
                replace=False,
            )
            target_selected = rng.choice(
                target_region_rows,
                size=min(target_count, len(target_region_rows)),
                replace=False,
            )
            ligand_values = rng.choice(ligand_pool, size=len(source_selected), replace=True)
            receptor_values = rng.choice(
                receptor_pool, size=len(target_selected), replace=True
            )
            positive_edge_count = _candidate_edge_count(
                source_selected, target_selected, spot_by_row, knn_mask
            )

            ligand = f"L{prefix}{i}"
            receptor = f"R{prefix}{i}"
            _assign_gene_values(
                expression, row_sums, ligand, source_selected, ligand_values
            )
            _assign_gene_values(
                expression, row_sums, receptor, target_selected, receptor_values
            )
            positive = {
                "ligand": ligand,
                "receptor": receptor,
                "is_positive": 1,
                "pattern": scenario,
                "template": template,
                "region": scenario,
                "source_rows": len(source_selected),
                "target_rows": len(target_selected),
                "matched_group": f"{scenario}:{template}:{i}",
                "designed_edge_count": positive_edge_count,
                "edge_count_relative_error": 0.0,
            }
            truth_rows.append(positive)
            positive_rows.append(positive)

            diffuse_source_pool = _eligible_rows(
                spot_by_row, celltype_by_row, global_mask, source_types
            )
            diffuse_target_pool = _eligible_rows(
                spot_by_row, celltype_by_row, global_mask, target_types
            )
            diffuse_source, diffuse_target, diffuse_edge_count, diffuse_dispersion = (
                _edge_count_matched_diffuse_rows(
                    source_pool=diffuse_source_pool,
                    target_pool=diffuse_target_pool,
                    source_count=len(source_selected),
                    target_count=len(target_selected),
                    target_edge_count=positive_edge_count,
                    spot_by_row=spot_by_row,
                    coords=coords,
                    knn_mask=knn_mask,
                    rng=rng,
                )
            )
            diffuse_ligand = f"LD{prefix}{i}"
            diffuse_receptor = f"RD{prefix}{i}"
            _assign_gene_values(
                expression,
                row_sums,
                diffuse_ligand,
                diffuse_source,
                rng.permutation(ligand_values),
            )
            _assign_gene_values(
                expression,
                row_sums,
                diffuse_receptor,
                diffuse_target,
                rng.permutation(receptor_values),
            )
            truth_rows.append({
                "ligand": diffuse_ligand,
                "receptor": diffuse_receptor,
                "is_positive": 0,
                "pattern": "diffuse_edge_count_matched",
                "template": template,
                "region": scenario,
                "source_rows": len(diffuse_source),
                "target_rows": len(diffuse_target),
                "matched_group": positive["matched_group"],
                "designed_edge_count": diffuse_edge_count,
                "edge_count_relative_error": (
                    abs(diffuse_edge_count - positive_edge_count)
                    / max(1, positive_edge_count)
                ),
                "spatial_dispersion": diffuse_dispersion,
            })

            if include_separated_controls:
                far_mask = _farthest_mask(region_mask, coords, int(region_mask.sum()))
                far_target_pool = _eligible_rows(
                    spot_by_row, celltype_by_row, far_mask, target_types
                )
                far_target = rng.choice(
                    far_target_pool,
                    size=min(len(target_selected), len(far_target_pool)),
                    replace=False,
                )
                separated_ligand = f"LS{prefix}{i}"
                separated_receptor = f"RS{prefix}{i}"
                _assign_gene_values(
                    expression,
                    row_sums,
                    separated_ligand,
                    source_selected,
                    rng.permutation(ligand_values),
                )
                _assign_gene_values(
                    expression,
                    row_sums,
                    separated_receptor,
                    far_target,
                    rng.choice(receptor_pool, size=len(far_target), replace=True),
                )
                truth_rows.append({
                    "ligand": separated_ligand,
                    "receptor": separated_receptor,
                    "is_positive": 0,
                    "pattern": "spatially_separated_matched",
                    "template": template,
                    "region": scenario,
                    "source_rows": len(source_selected),
                    "target_rows": len(far_target),
                    "matched_group": positive["matched_group"],
                    "designed_edge_count": _candidate_edge_count(
                        source_selected, far_target, spot_by_row, knn_mask
                    ),
                    "edge_count_relative_error": 1.0,
                })

    n_templates = len(selected_template_specs)
    for i, row in enumerate(positive_rows):
        other = positive_rows[(i + n_templates) % len(positive_rows)]
        truth_rows.append({
            "ligand": row["ligand"],
            "receptor": other["receptor"],
            "is_positive": 0,
            "pattern": "partner_swap_cross_region",
            "template": row["template"],
            "region": row["region"],
            "source_rows": row["source_rows"],
            "target_rows": other["target_rows"],
            "matched_group": row["matched_group"],
            "designed_edge_count": np.nan,
            "edge_count_relative_error": np.nan,
        })

    all_cell_types = composition.columns.tolist()
    for i, scale in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], start=1):
        source_pool = _eligible_rows(
            spot_by_row, celltype_by_row, global_mask, all_cell_types
        )
        selected = _spread_rows(
            source_pool, spot_by_row, coords, min(646, len(source_pool))
        )
        ligand = f"LHI{i}"
        receptor = f"RHI{i}"
        _assign_gene_values(
            expression,
            row_sums,
            ligand,
            selected,
            rng.choice(
                empirical_templates[1][1] * scale,
                size=len(selected),
                replace=True,
            ),
        )
        _assign_gene_values(
            expression,
            row_sums,
            receptor,
            selected,
            rng.choice(
                empirical_templates[1][2] * scale,
                size=len(selected),
                replace=True,
            ),
        )
        truth_rows.append({
            "ligand": ligand,
            "receptor": receptor,
            "is_positive": 0,
            "pattern": "diffuse_abundance_challenge",
            "template": f"{empirical_templates[1][0]}_x{scale:g}",
            "region": "",
            "source_rows": len(selected),
            "target_rows": len(selected),
            "matched_group": f"abundance_challenge:{i}",
            "designed_edge_count": _candidate_edge_count(
                selected, selected, spot_by_row, knn_mask
            ),
            "edge_count_relative_error": np.nan,
        })

    truth = pd.DataFrame(truth_rows)
    regions = pd.DataFrame(region_masks, index=composition.index)
    truth["lr_pair"] = truth["ligand"] + "_" + truth["receptor"]
    return expression, truth, regions


def generate_semisynthetic_inputs(config: SemisyntheticBenchmarkConfig) -> Dict[str, Path]:
    output = Path(config.output)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    composition = pd.read_csv(config.composition_csv, index_col=0)
    expression = pd.read_csv(config.spot_cell_expr_csv, index_col=0)
    expression = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    adata = ad.read_h5ad(config.st_h5ad)
    composition = composition.reindex(adata.obs_names).fillna(0.0)
    expression, truth, regions = _build_truth_and_expression(
        composition=composition,
        expression=expression,
        coords=np.asarray(adata.obsm["spatial"]),
        seed=config.seed,
        templates_per_scenario=config.templates_per_scenario,
        include_separated_controls=config.include_separated_controls,
    )

    composition_path = data_dir / "semisynthetic_composition.csv"
    expression_path = data_dir / "semisynthetic_spot_cell_expr.csv"
    lr_path = data_dir / "semisynthetic_lr_database.csv"
    truth_path = data_dir / "semisynthetic_lr_truth.csv"
    regions_path = data_dir / "semisynthetic_regions.csv"
    composition.to_csv(composition_path)
    expression.to_csv(expression_path)
    truth[["ligand", "receptor"]].to_csv(lr_path, index=False)
    truth.to_csv(truth_path, index=False)
    regions.to_csv(regions_path)
    return {
        "composition": composition_path,
        "expression": expression_path,
        "lr_database": lr_path,
        "truth": truth_path,
        "regions": regions_path,
    }


def _ranking_metrics(ranking: pd.DataFrame, score_column: str) -> dict:
    ordered = ranking.sort_values(score_column, ascending=False).reset_index(drop=True)
    y_true = ordered["is_positive"].to_numpy()
    y_score = ordered[score_column].to_numpy()
    metrics = {
        f"{score_column}_auprc": float(average_precision_score(y_true, y_score)),
        f"{score_column}_auroc": float(roc_auc_score(y_true, y_score)),
    }
    positive_count = max(1, int(y_true.sum()))
    for requested_k in (12, 20, 40):
        k = min(requested_k, len(ordered))
        metrics[f"{score_column}_precision_at_{requested_k}"] = float(y_true[:k].mean())
        metrics[f"{score_column}_recall_at_{requested_k}"] = float(
            y_true[:k].sum() / positive_count
        )
    return metrics


def evaluate_semisynthetic_benchmark(output: Path) -> pd.DataFrame:
    truth = pd.read_csv(output / "data" / "semisynthetic_lr_truth.csv")
    regions = pd.read_csv(
        output / "data" / "semisynthetic_regions.csv", index_col=0
    ).astype(bool)
    stats = pd.read_csv(output / "cellcom" / "lr_pair_statistics.csv")
    ranking = truth.merge(stats, on="lr_pair", how="left")
    communication_path = output / "cellcom" / "lr_communication.csv"
    if communication_path.exists():
        communication = pd.read_csv(communication_path)
        edge_stats = communication.groupby("lr_pair").agg(
            edge_count=("attention_score", "size"),
            mean_edge_attention=("attention_score", "mean"),
            q90_edge_attention=("attention_score", lambda x: x.quantile(0.9)),
            mean_original_lr_score=("original_lr_score", "mean"),
        )
        ranking = ranking.merge(edge_stats, left_on="lr_pair", right_index=True, how="left")
        localization_rows = []
        for row in truth.itertuples(index=False):
            pair_edges = communication.loc[communication["lr_pair"].eq(row.lr_pair)]
            if pair_edges.empty or not row.region or row.region not in regions.columns:
                localization_rows.append({
                    "lr_pair": row.lr_pair,
                    "region_edge_fraction": 0.0,
                    "region_attention_mass_fraction": 0.0,
                    "region_mean_attention": 0.0,
                    "outside_mean_attention": 0.0,
                    "localization_contrast": 0.0,
                })
                continue
            src_inside = pair_edges["src_spot_barcode"].map(
                regions[row.region]
            ).fillna(False).to_numpy()
            dst_inside = pair_edges["dst_spot_barcode"].map(
                regions[row.region]
            ).fillna(False).to_numpy()
            inside = src_inside & dst_inside
            attention = pair_edges["attention_score"].to_numpy(dtype=float)
            inside_attention = attention[inside]
            outside_attention = attention[~inside]
            attention_sum = float(attention.sum())
            localization_rows.append({
                "lr_pair": row.lr_pair,
                "region_edge_fraction": float(inside.mean()),
                "region_attention_mass_fraction": (
                    float(inside_attention.sum() / attention_sum)
                    if attention_sum > 0 else 0.0
                ),
                "region_mean_attention": (
                    float(inside_attention.mean()) if len(inside_attention) else 0.0
                ),
                "outside_mean_attention": (
                    float(outside_attention.mean()) if len(outside_attention) else 0.0
                ),
                "localization_contrast": (
                    float(inside_attention.mean() - outside_attention.mean())
                    if len(inside_attention) and len(outside_attention)
                    else 0.0
                ),
            })
        ranking = ranking.merge(pd.DataFrame(localization_rows), on="lr_pair", how="left")

    score_columns = [
        column
        for column in [
            "avg_attention_score",
            "q90_edge_attention",
            "mean_edge_attention",
            "region_edge_fraction",
            "region_attention_mass_fraction",
        ]
        if column in ranking.columns
    ]
    for column in score_columns:
        fallback = ranking[column].min() - 1 if ranking[column].notna().any() else 0.0
        ranking[column] = ranking[column].fillna(fallback)
        ranking[f"{column}_rank"] = ranking[column].rank(method="min", ascending=False)

    metrics = {
        "n_candidates": int(len(ranking)),
        "n_positives": int(ranking["is_positive"].sum()),
    }
    for column in score_columns:
        metrics.update(_ranking_metrics(ranking, column))
    scenario_ranks = (
        ranking.loc[ranking["is_positive"] == 1]
        .groupby("pattern")["avg_attention_score_rank"]
        .agg(["min", "median", "max"])
        .to_dict(orient="index")
    )
    metrics["positive_rank_by_scenario"] = scenario_ranks

    ranking.sort_values("avg_attention_score", ascending=False).to_csv(
        output / "semisynthetic_lr_ranking.csv", index=False
    )
    with open(output / "semisynthetic_lr_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return ranking


def run_semisynthetic_benchmark(config: SemisyntheticBenchmarkConfig) -> pd.DataFrame:
    output = Path(config.output)
    paths = generate_semisynthetic_inputs(config)
    with open(output / "benchmark_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    if not config.run_model:
        return pd.read_csv(paths["truth"])

    from spagraph.training.cellcom import run_cellcom

    run_cellcom(
        deconv_dir=str(paths["composition"].parent),
        st_h5ad=config.st_h5ad,
        output_dir=str(output / "cellcom"),
        composition_csv=str(paths["composition"]),
        spot_cell_expr_csv=str(paths["expression"]),
        lr_database_csv=str(paths["lr_database"]),
        n_spot_neighbors=10,
        ligand_expr_threshold=3.0,
        receptor_expr_threshold=1.0,
        lr_score_threshold=1.0,
        min_comm_edges=1,
        use_hvg_for_communication=False,
        allow_same_celltype_comm=False,
        gat_hidden_dims="128,64",
        gat_heads=4,
        gat_dropout=0.3,
        output_dim=64,
        mlp_latent_dim=64,
        mlp_hidden_dims="128,64",
        batch_size=4,
        epochs=config.epochs,
        early_stop_patience=max(8, config.epochs // 3),
        early_stop_min_delta=0.001,
        attention_threshold=0.0,
        export_unified_csv=True,
        export_filtered_csv=False,
        save_lr_scores_csv=True,
        device=config.device,
        seed=config.seed,
    )
    return evaluate_semisynthetic_benchmark(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/semisynthetic_lr_benchmark")
    parser.add_argument("--composition-csv", default=SemisyntheticBenchmarkConfig.composition_csv)
    parser.add_argument("--spot-cell-expr-csv", default=SemisyntheticBenchmarkConfig.spot_cell_expr_csv)
    parser.add_argument("--st-h5ad", default=SemisyntheticBenchmarkConfig.st_h5ad)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--templates-per-scenario", type=int, default=3)
    parser.add_argument("--exclude-separated-controls", action="store_true")
    args = parser.parse_args()
    config = SemisyntheticBenchmarkConfig(
        output=args.output,
        composition_csv=args.composition_csv,
        spot_cell_expr_csv=args.spot_cell_expr_csv,
        st_h5ad=args.st_h5ad,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        run_model=not args.generate_only,
        templates_per_scenario=args.templates_per_scenario,
        include_separated_controls=not args.exclude_separated_controls,
    )
    ranking = run_semisynthetic_benchmark(config)
    columns = [
        "lr_pair",
        "is_positive",
        "pattern",
        "template",
        "source_rows",
        "target_rows",
    ]
    columns += [column for column in ["avg_attention_score", "avg_attention_score_rank"] if column in ranking]
    print(ranking[columns].sort_values(columns[-1], ascending=True).to_string(index=False))


if __name__ == "__main__":
    main()
