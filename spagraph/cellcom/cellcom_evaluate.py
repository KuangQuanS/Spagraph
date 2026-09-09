#!/usr/bin/env python3
"""
Cell-Cell Communication Evaluation Module

This module provides evaluation functions for analyzing cell-cell communication
results from trained HeteroGAT models, including attention score analysis,
LR pair statistics, and model-based communication prediction.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from .relation_ranker import rank_associated_attention


def evaluate_lr_candidate_scores(
    records: List[Dict[str, Any]],
    lr_id_to_pair: Dict[int, Tuple[str, str]],
    output_dir: str,
    min_ranking_edges: int = 10,
    score_column: str = "candidate_logit",
) -> Optional[pd.DataFrame]:
    """Write diagnostic scores from the optional LR candidate head.

    A candidate may occur in several overlapping center-spot subgraphs. Exact
    spatial/cell/LR events are deduplicated before pair statistics are formed.
    The resulting neural score is produced by the LR candidate head rather
    than copied from a shared aggregate communication edge.
    """
    if not records:
        print("LR candidate stats: no candidate scores collected")
        return None
    frame = pd.DataFrame.from_records(records)
    required = {
        "src_spot_barcode", "dst_spot_barcode", "source_cell", "target_cell",
        "lr_id", "lr_score", score_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing LR candidate columns: {sorted(missing)}")

    keys = [
        "src_spot_barcode", "dst_spot_barcode", "source_cell", "target_cell",
        "lr_id",
    ]
    aggregation = {
        "lr_score": ("lr_score", "mean"),
        score_column: (score_column, "mean"),
        "n_subgraph_occurrences": (score_column, "count"),
    }
    # Retain both neural outputs when available so absolute logits and
    # matched-control gaps can be compared from the same trained checkpoint.
    for diagnostic_column in ("candidate_logit", "candidate_gap"):
        if (
            diagnostic_column in frame.columns
            and diagnostic_column != score_column
        ):
            aggregation[diagnostic_column] = (diagnostic_column, "mean")
    if "n_matched_controls" in frame.columns:
        aggregation["n_matched_controls"] = ("n_matched_controls", "sum")
    unique = frame.groupby(keys, as_index=False).agg(**aggregation)
    unique["lr_pair"] = unique["lr_id"].map(
        lambda value: "_".join(lr_id_to_pair.get(int(value), (f"lr_{int(value)}",)))
    )
    unique = unique.loc[unique[score_column].notna()].copy()
    edge_path = os.path.join(output_dir, "lr_candidate_edge_statistics.csv")
    unique.sort_values(score_column, ascending=False).to_csv(
        edge_path, index=False
    )

    pair_rows = []
    for lr_pair, group in unique.groupby("lr_pair", sort=False):
        scores = group[score_column].to_numpy(dtype=float)
        if scores.size == 0:
            continue
        pair_rows.append(
            {
                "lr_pair": lr_pair,
                "supporting_unique_edges": int(scores.size),
                "candidate_score_mean": float(scores.mean()),
                "candidate_score_median": float(np.median(scores)),
                "candidate_score_std": float(scores.std()),
                "candidate_score_min": float(scores.min()),
                "candidate_score_max": float(scores.max()),
                "candidate_score_column": score_column,
                "total_lr_score": float(group["lr_score"].sum()),
                "n_source_spots": int(group["src_spot_barcode"].nunique()),
                "n_target_spots": int(group["dst_spot_barcode"].nunique()),
                "eligible_for_ranking": scores.size >= int(min_ranking_edges),
                "score_source": (
                    "pair_specific_lr_candidate_gap"
                    if score_column == "candidate_gap"
                    else "pair_specific_lr_candidate_head"
                ),
            }
        )
    if not pair_rows:
        print("LR candidate stats: no candidates with matched controls")
        return None
    pair_df = pd.DataFrame(pair_rows).sort_values("candidate_score_mean", ascending=False)
    pair_path = os.path.join(output_dir, "lr_candidate_pair_diagnostics.csv")
    pair_df.to_csv(pair_path, index=False)
    print(
        "LR candidate diagnostics: "
        f"{pair_path} (events={len(unique)}, LR pairs={len(pair_df)})"
    )
    return pair_df


def evaluate_cell_communication(
    all_cc_attention_scores: List[torch.Tensor],
    all_edge_index_cc: List[torch.Tensor],
    all_edge_attr_cc: List[torch.Tensor],
    all_spot_indices: List[torch.Tensor],
    all_n_spots_sub: List[torch.Tensor],
    all_cell_names: List[str],
    all_cell_node_mappings: List[Dict[int, int]],
    output_dir: str,
    n_spots: int,
    n_cells: int,
    spot_names: List[str] = None,
    all_src_barcodes: List[List[str]] = None,
    all_dst_barcodes: List[List[str]] = None,
    export_unified: bool = False,
    export_filtered: bool = False,
    attention_threshold: float = 0.1,
    lr_support_by_edge: Optional[
        Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], float]]
    ] = None,
) -> None:
    """
    Evaluate cell-cell communication from trained model attention scores.

    Args:
        all_cc_attention_scores: List of attention scores from all batches
        all_edge_index_cc: List of edge indices from all batches
        all_edge_attr_cc: List of edge attributes from all batches
        all_spot_indices: List of spot indices from all batches
        all_n_spots_sub: List of n_spots_sub values from all batches
        all_cell_names: List of cell type names
        output_dir: Output directory for results
        n_spots: Number of spots in the dataset
        n_cells: Number of cell types
        attention_threshold: Threshold for filtering edges by attention score (default: 0.1)
    """
    print(f"\n{'='*60}\nStage 3.6: Evaluate (attention importance)\n{'='*60}")

    if not all_cc_attention_scores:
        print("WARNING: no cell-cell attention scores collected")
        return

    print(f"Attention tensors:  {len(all_cc_attention_scores)}")

    # 合并所有batch的注意力得分
    all_scores = torch.cat(all_cc_attention_scores, dim=0)  # [total_edges]
    all_edges = torch.cat(all_edge_index_cc, dim=1)  # [2, total_edges]
    all_attrs = torch.cat(all_edge_attr_cc, dim=0)  # [total_edges, 2] - [lr_score, lr_id]
    all_spots = torch.cat(all_spot_indices, dim=0)  # [total_edges] - center spot indices
    all_n_spots_sub_batch = torch.cat(all_n_spots_sub, dim=0)  # [total_edges] - n_spots_sub for each edge
    
    # 合并cell_node_mappings
    all_cell_node_mappings_flat = []
    for batch_idx, mapping in enumerate(all_cell_node_mappings):
        num_edges_in_batch = all_cc_attention_scores[batch_idx].shape[0]
        all_cell_node_mappings_flat.extend([mapping] * num_edges_in_batch)
    
    # 合并barcode列表
    if all_src_barcodes is not None and all_dst_barcodes is not None:
        all_src_barcodes_flat = [barcode for batch_barcodes in all_src_barcodes for barcode in batch_barcodes]
        all_dst_barcodes_flat = [barcode for batch_barcodes in all_dst_barcodes for barcode in batch_barcodes]
    else:
        all_src_barcodes_flat = None
        all_dst_barcodes_flat = None

    n_unique_spots = len(torch.unique(all_spots))
    print(f"Edges total:        {all_scores.shape[0]} (unique_spots={n_unique_spots})")

    # 注意：all_scores现在是1维的[total_edges]，直接使用即可
    avg_scores = all_scores  # [total_edges] - 已经是平均后的注意力得分
    
    if lr_support_by_edge:
        unique_edges = {}
        for idx in range(all_edges.size(1)):
            n_spots_sub = int(all_n_spots_sub_batch[idx].item())
            src_local = int(all_edges[0, idx].item()) - n_spots_sub
            dst_local = int(all_edges[1, idx].item()) - n_spots_sub
            mapping = all_cell_node_mappings_flat[idx]
            if src_local not in mapping or dst_local not in mapping:
                raise ValueError(f'Missing cell-node mapping for communication edge {idx}')
            src_cell_id = mapping[src_local]
            dst_cell_id = mapping[dst_local]
            if not (0 <= src_cell_id < len(all_cell_names) and 0 <= dst_cell_id < len(all_cell_names)):
                raise ValueError(f'Invalid cell identity for communication edge {idx}')
            if all_src_barcodes_flat is None or all_dst_barcodes_flat is None:
                raise ValueError('Spot barcodes are required for unique communication-edge export')

            edge_key = (
                all_src_barcodes_flat[idx],
                all_dst_barcodes_flat[idx],
                all_cell_names[src_cell_id],
                all_cell_names[dst_cell_id],
            )
            edge_stats = unique_edges.setdefault(
                edge_key,
                {
                    "attention_sum": 0.0,
                    "n_subgraph_occurrences": 0,
                    "aggregate_lr_score": float(all_attrs[idx, 0].item()),
                },
            )
            edge_stats["attention_sum"] += float(avg_scores[idx].item())
            edge_stats["n_subgraph_occurrences"] += 1

        edge_rows = []
        pair_edges = {}
        for edge_key, edge_stats in unique_edges.items():
            support = lr_support_by_edge.get(edge_key, {})
            if not support:
                raise ValueError(f'Missing complete LR support for communication edge {edge_key}')
            edge_attention = (
                edge_stats["attention_sum"] / edge_stats["n_subgraph_occurrences"]
            )
            if not np.isfinite(edge_attention):
                raise ValueError(f'Nonfinite attention for communication edge {edge_key}')
            if any(not np.isfinite(float(value)) or float(value) < 0 for value in support.values()):
                raise ValueError(f'Invalid LR expression support for communication edge {edge_key}')
            support_names = sorted(
                f"{ligand}_{receptor}" for ligand, receptor in support
            )
            edge_rows.append(
                {
                    "src_spot_barcode": edge_key[0],
                    "dst_spot_barcode": edge_key[1],
                    "source_cell": edge_key[2],
                    "target_cell": edge_key[3],
                    "aggregate_lr_score": edge_stats["aggregate_lr_score"],
                    "edge_attention": edge_attention,
                    "n_subgraph_occurrences": edge_stats["n_subgraph_occurrences"],
                    "supporting_lr_count": len(support),
                    "supporting_lr_pairs": ";".join(support_names),
                    "supporting_lr_scores": json.dumps({
                        f"{ligand}_{receptor}": float(value)
                        for (ligand, receptor), value in sorted(support.items())
                    }, sort_keys=True),
                }
            )
            for lr_pair, lr_score in support.items():
                pair_edges.setdefault(lr_pair, []).append(
                    {
                        "edge_attention": edge_attention,
                        "lr_score": lr_score,
                        "src_spot": edge_key[0],
                        "dst_spot": edge_key[1],
                    }
                )

        if not edge_rows:
            raise ValueError('No valid communication edges available for export')
        edge_df = pd.DataFrame(edge_rows).sort_values(
            "edge_attention", ascending=False
        )
        edge_path = os.path.join(output_dir, "communication_edge_statistics.csv")
        edge_df.to_csv(edge_path, index=False)

        min_ranking_edges = 10
        pair_rows = []
        for (ligand, receptor), records in pair_edges.items():
            attention = np.asarray([record["edge_attention"] for record in records])
            lr_scores = np.asarray([record["lr_score"] for record in records])
            pair_rows.append(
                {
                    "lr_pair": f"{ligand}_{receptor}",
                    "supporting_unique_edges": len(records),
                    "associated_edge_attention_mean": float(attention.mean()),
                    "associated_edge_attention_median": float(np.median(attention)),
                    "associated_edge_attention_std": float(attention.std()),
                    "associated_edge_attention_min": float(attention.min()),
                    "associated_edge_attention_max": float(attention.max()),
                    "total_lr_score": float(lr_scores.sum()),
                    "n_source_spots": len({record["src_spot"] for record in records}),
                    "n_target_spots": len({record["dst_spot"] for record in records}),
                    "eligible_for_ranking": len(records) >= min_ranking_edges,
                }
            )

        pair_df = pd.DataFrame(pair_rows)
        pair_df["avg_attention_score"] = pair_df[
            "associated_edge_attention_mean"
        ]
        pair_df["std_attention_score"] = pair_df[
            "associated_edge_attention_std"
        ]
        pair_df["occurrence_count"] = pair_df["supporting_unique_edges"]
        pair_df["score_source"] = "shared_attention_associated_lr_support"
        pair_df = rank_associated_attention(pair_df)
        pair_path = os.path.join(output_dir, "lr_pair_statistics.csv")
        pair_df.to_csv(pair_path, index=False)
        print(
            "Unique edge stats:   "
            f"{edge_path} (edges={len(edge_df)}, LR pairs={len(pair_df)}, "
            f"ranking_min_edges={min_ranking_edges})"
        )
    # Aggregate-edge representative LR IDs cannot identify individual LR scores.
    # Stop before the historical exports, including filtered variants.
    if export_unified or export_filtered:
        print("Legacy representative-LR exports are disabled; use communication_edge_statistics.csv")
    if not lr_support_by_edge:
        raise ValueError("Complete lr_support_by_edge is required for valid LR ranking")
    return
def plot_dgi_loss(dgi_train_losses, dgi_val_losses=None, output_dir: str = None, epochs: int = None) -> None:
    """
    Plot and save DGI pretraining loss curve.

    Args:
        dgi_train_losses: List of DGI pretraining training losses for each epoch
        dgi_val_losses: List of DGI pretraining validation losses for each epoch (optional)
        output_dir: Output directory for the plot
        epochs: Total number of epochs (optional, will use len(dgi_train_losses) if not provided)
    """
    # ✅ 使用实际训练的epoch数，而不是预设的epochs参数
    actual_epochs = len(dgi_train_losses)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), dgi_train_losses, label="DGI Train Loss", linewidth=2, marker='o', color='orange')
    
    if dgi_val_losses is not None and len(dgi_val_losses) > 0:
        plt.plot(range(1, len(dgi_val_losses) + 1), dgi_val_losses, label="DGI Val Loss", linewidth=2, marker='s', color='red', linestyle='--')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"DGI Pretraining Loss Curve (Trained {actual_epochs} epochs)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir is not None:
        dgi_loss_curve_path = os.path.join(output_dir, "dgi_loss_curve.png")
        plt.savefig(dgi_loss_curve_path, dpi=150)
        plt.close()
        print(f"DGI loss curve saved: {dgi_loss_curve_path}")
    else:
        plt.show()


def plot_training_loss(train_losses: List[float], val_losses: List[float] = None, output_dir: str = None, epochs: int = None) -> None:
    """
    Plot and save training loss curve.

    Args:
        train_losses: List of training losses for each epoch
        val_losses: List of validation losses for each epoch (optional)
        output_dir: Output directory for the plot
        epochs: Total number of epochs (optional, will use len(train_losses) if not provided)
    """
    # ✅ 使用实际训练的epoch数，而不是预设的epochs参数
    actual_epochs = len(train_losses)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_losses, label="Training Loss", linewidth=2, marker='o')
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linewidth=2, marker='s', linestyle='--', color='red')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"HeteroGAT Training Loss Curve (Trained {actual_epochs} epochs)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir is not None:
        loss_curve_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_curve_path, dpi=150)
        plt.close()
        print(f"Loss curve saved:   {loss_curve_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # This module is meant to be imported, not run directly
    print("This is an evaluation module. Import and use the evaluate_cell_communication() function.")
