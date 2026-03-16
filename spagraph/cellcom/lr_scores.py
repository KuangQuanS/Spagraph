#!/usr/bin/env python3
"""LR communication score calculation utilities."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm


def _build_filtered_gene_indices(gene_names: List[str], gene_name_to_idx: Dict[str, int]) -> set[int]:
    filtered_indices: set[int] = set()
    for gene_name in gene_names:
        gene_upper = gene_name.upper()
        if gene_upper.startswith("MT-"):
            filtered_indices.add(gene_name_to_idx[gene_upper])
        elif gene_upper.startswith("HB"):
            filtered_indices.add(gene_name_to_idx[gene_upper])
        elif "PSEUDO" in gene_upper or gene_upper.endswith("-AS1") or gene_upper.startswith("LOC"):
            filtered_indices.add(gene_name_to_idx[gene_upper])
    return filtered_indices


def _build_active_gene_sets(
    spot_cell_names: List[str],
    spot_cell_name_to_idx: Dict[str, int],
    spot_cell_expr_array: np.ndarray,
    filtered_gene_indices: set[int],
    allowed_gene_indices: Optional[set[int]],
    ligand_expr_threshold: float,
    receptor_expr_threshold: float,
) -> Tuple[Dict[str, set[int]], Dict[str, set[int]], Dict[str, float]]:
    ligand_sets: Dict[str, set[int]] = {}
    receptor_sets: Dict[str, set[int]] = {}
    total_counts: Dict[str, float] = {}

    for spot_cell_name in spot_cell_names:
        idx = spot_cell_name_to_idx[spot_cell_name]
        cell_expr = spot_cell_expr_array[idx, :]
        total_count = float(cell_expr.sum())
        total_counts[spot_cell_name] = total_count

        if total_count > 0:
            cell_expr_normalized = cell_expr / total_count * 1e4
        else:
            cell_expr_normalized = cell_expr

        ligand_active_indices = set(np.where(cell_expr_normalized > ligand_expr_threshold)[0]) - filtered_gene_indices
        receptor_active_indices = set(np.where(cell_expr_normalized > receptor_expr_threshold)[0]) - filtered_gene_indices

        if allowed_gene_indices is not None:
            ligand_active_indices &= allowed_gene_indices
            receptor_active_indices &= allowed_gene_indices

        ligand_sets[spot_cell_name] = ligand_active_indices
        receptor_sets[spot_cell_name] = receptor_active_indices

    return ligand_sets, receptor_sets, total_counts


def _build_valid_lr_pairs(
    lr_pairs: List[Tuple[str, str]],
    gene_name_to_idx: Dict[str, int],
) -> Tuple[List[Tuple[int, Tuple[int, ...], str, str]], Dict[int, List[Tuple[int, Tuple[int, ...], str, str]]]]:
    valid_lr_pairs: List[Tuple[int, Tuple[int, ...], str, str]] = []
    valid_lr_pairs_by_ligand: Dict[int, List[Tuple[int, Tuple[int, ...], str, str]]] = {}

    for ligand, receptor in lr_pairs:
        lig_idx = gene_name_to_idx.get(ligand.upper())
        if lig_idx is None:
            continue

        rec_indices: List[int] = []
        found_all = True
        for receptor_gene in receptor.split("_"):
            rec_idx = gene_name_to_idx.get(receptor_gene.strip().upper())
            if rec_idx is None:
                found_all = False
                break
            rec_indices.append(rec_idx)

        if not found_all:
            continue

        record = (lig_idx, tuple(rec_indices), ligand, receptor)
        valid_lr_pairs.append(record)
        valid_lr_pairs_by_ligand.setdefault(lig_idx, []).append(record)

    return valid_lr_pairs, valid_lr_pairs_by_ligand


def calculate_lr_scores(
    spot_coords: np.ndarray,
    composition: Optional[pd.DataFrame],
    args: Any,
    adata: Any,
    cell_full_expr: pd.DataFrame,
    lr_pairs: List[Tuple[str, str]],
    output_dir: str,
    n_neighbors: int = 20,
    hvg_genes: Optional[List[str]] = None,
    ligand_expr_threshold: float = 3.0,
    receptor_expr_threshold: float = 1.0,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Calculate KNN neighborhoods and LR communication scores."""
    if composition is None:
        raise ValueError("composition is required for LR score calculation")

    n_spots = spot_coords.shape[0]
    knn = kneighbors_graph(spot_coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    knn_mask = knn.toarray()

    lr_comm_distance_threshold = 500.0
    ligand_expr_threshold = getattr(args, "ligand_expr_threshold", ligand_expr_threshold)
    receptor_expr_threshold = getattr(args, "receptor_expr_threshold", receptor_expr_threshold)
    lr_score_threshold = getattr(args, "lr_score_threshold", 3.0)

    print(
        "LR scores:          "
        f"spots={n_spots}, knn={n_neighbors}, dist_thr={lr_comm_distance_threshold}, "
        f"lig_thr={ligand_expr_threshold}, rec_thr={receptor_expr_threshold}, lr_thr={lr_score_threshold}"
    )

    spot_names = adata.obs_names.tolist()
    spot_cell_expr_array = cell_full_expr.values
    spot_cell_names = cell_full_expr.index.tolist()
    gene_names = cell_full_expr.columns.tolist()
    spot_cell_name_to_idx = {name: idx for idx, name in enumerate(spot_cell_names)}
    cell_types = composition.columns.tolist()

    print(f"LR scores:          celltypes={len(cell_types)}, spot-cells={len(spot_cell_names)}")

    gene_name_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_names)}
    if hvg_genes is not None:
        allowed_gene_indices = {
            gene_name_to_idx[gene.upper()]
            for gene in hvg_genes
            if gene.upper() in gene_name_to_idx
        }
        print(f"LR scores:          comm genes (HVG) = {len(allowed_gene_indices)}/{len(gene_names)}")
    else:
        allowed_gene_indices = None
        print("LR scores:          comm genes (ALL)")

    filtered_gene_indices = _build_filtered_gene_indices(gene_names, gene_name_to_idx)
    active_ligand_sets, active_receptor_sets, total_counts = _build_active_gene_sets(
        spot_cell_names=spot_cell_names,
        spot_cell_name_to_idx=spot_cell_name_to_idx,
        spot_cell_expr_array=spot_cell_expr_array,
        filtered_gene_indices=filtered_gene_indices,
        allowed_gene_indices=allowed_gene_indices,
        ligand_expr_threshold=ligand_expr_threshold,
        receptor_expr_threshold=receptor_expr_threshold,
    )

    valid_lr_pairs, valid_lr_pairs_by_ligand = _build_valid_lr_pairs(lr_pairs, gene_name_to_idx)
    print(f"LR scores:          valid LR pairs = {len(valid_lr_pairs)}/{len(lr_pairs)}")

    composition_values = composition.to_numpy(copy=False)
    spot_cell_entries: List[List[Tuple[str, str, np.ndarray, float, set[int], set[int]]]] = []
    for spot_idx, spot_barcode in enumerate(spot_names):
        entries: List[Tuple[str, str, np.ndarray, float, set[int], set[int]]] = []
        cell_in_spot = np.where(composition_values[spot_idx] > 1e-6)[0]
        for cell_idx in cell_in_spot:
            celltype = cell_types[cell_idx]
            spot_cell_key = f"{spot_barcode}_{celltype}"
            row_idx = spot_cell_name_to_idx.get(spot_cell_key)
            if row_idx is None:
                continue
            entries.append(
                (
                    celltype,
                    spot_cell_key,
                    spot_cell_expr_array[row_idx, :],
                    total_counts[spot_cell_key],
                    active_ligand_sets[spot_cell_key],
                    active_receptor_sets[spot_cell_key],
                )
            )
        spot_cell_entries.append(entries)

    comm_event_records = []
    total_pairs = 0
    spots_with_cells = 0
    spots_without_cells = 0
    allow_same_celltype_comm = bool(getattr(args, "allow_same_celltype_comm", False))
    same_celltype_skipped = 0

    for i in tqdm(range(n_spots), desc="Computing LR scores", leave=False):
        spot_i_barcode = spot_names[i]
        source_entries = spot_cell_entries[i]
        if not source_entries:
            spots_without_cells += 1
            continue

        spots_with_cells += 1
        neighbor_js = np.where(knn_mask[i] == 1)[0]
        if neighbor_js.size == 0:
            continue

        diffs = spot_coords[neighbor_js] - spot_coords[i]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        keep_mask = dists <= lr_comm_distance_threshold
        neighbor_js = neighbor_js[keep_mask]
        dists = dists[keep_mask]

        for j, dist_ij in zip(neighbor_js, dists):
            total_pairs += 1
            spot_j_barcode = spot_names[j]
            target_entries = spot_cell_entries[j]
            if not target_entries:
                continue

            for celltype_i, spot_cell_i_key, cell_i_expr, total_i, active_lig_set, _ in source_entries:
                if total_i <= 0 or not active_lig_set:
                    continue

                for celltype_j, spot_cell_j_key, cell_j_expr, total_j, _, active_rec_set in target_entries:
                    if (not allow_same_celltype_comm) and (celltype_i == celltype_j):
                        same_celltype_skipped += 1
                        continue
                    if total_j <= 0 or not active_rec_set:
                        continue

                    normalization_factor = 1e4 / np.sqrt(total_i * total_j)
                    for lig_idx in active_lig_set:
                        ligand_pairs = valid_lr_pairs_by_ligand.get(lig_idx)
                        if not ligand_pairs:
                            continue

                        lig_val = cell_i_expr[lig_idx]
                        if lig_val <= 0:
                            continue

                        for _, rec_indices, ligand, receptor in ligand_pairs:
                            if not all(rec_idx in active_rec_set for rec_idx in rec_indices):
                                continue

                            rec_product = np.prod(cell_j_expr[list(rec_indices)])
                            score = np.log1p(np.sqrt(lig_val * rec_product) * normalization_factor)
                            if score >= lr_score_threshold:
                                comm_event_records.append(
                                    [
                                        spot_i_barcode,
                                        spot_j_barcode,
                                        spot_cell_i_key,
                                        spot_cell_j_key,
                                        ligand,
                                        receptor,
                                        score,
                                        1,
                                        float(dist_ij),
                                    ]
                                )

    print(
        "LR scores:          "
        f"events={len(comm_event_records)}, neighbor_pairs={total_pairs}, "
        f"spots_with_cells={spots_with_cells}/{n_spots}, "
        f"same_type={'ON' if allow_same_celltype_comm else 'OFF'}"
    )
    if (not allow_same_celltype_comm) and same_celltype_skipped:
        print(f"LR scores:          same-type skipped={same_celltype_skipped}")

    csv_path = os.path.join(output_dir, "lr_scores.csv")
    df = pd.DataFrame(
        comm_event_records,
        columns=["spot_i", "spot_j", "cell_i", "cell_j", "ligand", "receptor", "comm_score", "in_knn", "distance"],
    )
    df.to_csv(csv_path, index=False)
    print(
        "LR scores saved:    "
        f"{csv_path} (events={len(df)}, "
        f"spot_pairs={df.groupby(['spot_i', 'spot_j']).ngroups}, "
        f"cell_pairs={df.groupby(['cell_i', 'cell_j']).ngroups})"
    )

    graph_data = {
        "coords": spot_coords,
        "composition": composition,
        "knn_mask": knn_mask,
    }
    return knn_mask, csv_path, graph_data


if __name__ == "__main__":
    print("This is an LR scores calculation module. Import and use the calculate_lr_scores() function.")
