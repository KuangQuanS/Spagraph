#!/usr/bin/env python3
"""
Cell-Cell Communication Evaluation Module

This module provides evaluation functions for analyzing cell-cell communication
results from trained HeteroGAT models, including attention score analysis,
LR pair statistics, and model-based communication prediction.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any


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
    export_unified: bool = True,
    attention_threshold: float = 0.1
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
    
    # ========== 应用边过滤策略：Attention Threshold ==========
    # 保留注意力得分高于阈值的边，去除假阳性
    keep_mask = (avg_scores >= attention_threshold)
    print(
        "Filter by attention: "
        f"thr={attention_threshold}, kept={int(keep_mask.sum())}/{len(avg_scores)} "
        f"({keep_mask.sum()/len(avg_scores)*100:.1f}%)"
    )

    # ✅ 保存过滤前的完整数据，用于生成 model_based_comm_path
    all_scores_full = avg_scores.clone()  # 保存过滤前的完整注意力得分
    all_edges_full = all_edges.clone()    # 保存过滤前的完整边索引
    all_attrs_full = all_attrs.clone()    # 保存过滤前的完整边属性
    all_spots_full = all_spots.clone()    # 保存过滤前的完整spot索引
    all_n_spots_sub_full = all_n_spots_sub_batch.clone()  # 保存过滤前的完整n_spots_sub

    # 应用过滤
    filtered_scores = avg_scores[keep_mask]
    filtered_edges = all_edges[:, keep_mask]
    filtered_attrs = all_attrs[keep_mask]
    filtered_spots = all_spots[keep_mask]

    # 加载LR对映射
    lr_mapping_path = os.path.join(output_dir, "lr_pair_mapping.txt")
    lr_id_to_pair = {}
    if os.path.exists(lr_mapping_path):
        with open(lr_mapping_path, 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                lr_id, ligand, receptor = line.strip().split('\t')
                lr_id_to_pair[int(lr_id)] = (ligand, receptor)
        print(f"LR mapping loaded:  {len(lr_id_to_pair)} pairs")
    else:
        print(f"WARNING: missing LR mapping file: {lr_mapping_path}")

    # 统计每个LR对的得分（按spot聚合）
    lr_spot_scores = {}  # {(center_spot, lr_id): [attention_scores]}
    # 初始化统计计数器
    processed_edges = 0
    skipped_no_lr = 0

    for i in range(all_edges.size(1)):
        center_spot_idx = all_spots[i].item()
        lr_score = all_attrs[i, 0].item()
        lr_id = int(all_attrs[i, 1].item())
        attention_score = avg_scores[i].item()

        if lr_id >= 0:  # 有效的LR对
            # 收集每个spot-lr对的得分
            key = (center_spot_idx, lr_id)
            if key not in lr_spot_scores:
                lr_spot_scores[key] = []
            lr_spot_scores[key].append(attention_score)
            processed_edges += 1
        else:
            skipped_no_lr += 1

    print(f"LR scores used:     processed_edges={processed_edges}, skipped_no_lr={skipped_no_lr}")

    # 计算每个spot-lr对的平均得分
    spot_lr_avg_scores = {}
    for key, scores in lr_spot_scores.items():
        spot_lr_avg_scores[key] = np.mean(scores)

    # ========== 统计LR对的出现次数和得分 ==========

    # 按LR ID统计
    lr_id_stats = {}
    for (spot_idx, lr_id), score in spot_lr_avg_scores.items():
        if lr_id not in lr_id_stats:
            lr_id_stats[lr_id] = {
                'count': 0,
                'scores': [],
                'spots': set()
            }
        lr_id_stats[lr_id]['count'] += 1
        lr_id_stats[lr_id]['scores'].append(score)
        lr_id_stats[lr_id]['spots'].add(spot_idx)

    # 计算统计量并转换为LR对名称
    lr_pair_summary = []
    for lr_id, stats in lr_id_stats.items():
        if lr_id in lr_id_to_pair:
            ligand, receptor = lr_id_to_pair[lr_id]
            lr_pair_name = f"{ligand}_{receptor}"
        else:
            lr_pair_name = f"lr_{lr_id}"

        lr_pair_summary.append({
            'lr_pair': lr_pair_name,
            'lr_id': lr_id,
            'occurrence_count': stats['count'],
            'avg_attention_score': np.mean(stats['scores']),
            'std_attention_score': np.std(stats['scores']),
            'min_attention_score': np.min(stats['scores']),
            'max_attention_score': np.max(stats['scores']),
            'n_spots': len(stats['spots'])
        })

    # 保存LR对统计结果
    lr_stats_path = os.path.join(output_dir, "lr_pair_statistics.csv")
    with open(lr_stats_path, 'w') as f:
        f.write("lr_pair,lr_id,occurrence_count,avg_attention_score,std_attention_score,min_attention_score,max_attention_score,n_spots\n")
        for item in sorted(lr_pair_summary, key=lambda x: x['occurrence_count'], reverse=True):
            f.write(f"{item['lr_pair']},{item['lr_id']},{item['occurrence_count']},"
                   f"{item['avg_attention_score']:.6f},{item['std_attention_score']:.6f},"
                   f"{item['min_attention_score']:.6f},{item['max_attention_score']:.6f},"
                   f"{item['n_spots']}\n")

    print(f"LR stats saved:     {lr_stats_path} (unique_lr_pairs={len(lr_pair_summary)})")

    top_k = 10
    print(f"\nTop {top_k} LR pairs by occurrence:")
    for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['occurrence_count'], reverse=True)[:top_k]):
        print(
            f"  {i+1}. {item['lr_pair']}: n={item['occurrence_count']}, "
            f"avg_attention={item['avg_attention_score']:.4f}, spots={item['n_spots']}"
        )

    print(f"\nTop {top_k} LR pairs by attention:")
    for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['avg_attention_score'], reverse=True)[:top_k]):
        print(f"  {i+1}. {item['lr_pair']}: avg_attention={item['avg_attention_score']:.4f}, n={item['occurrence_count']}")

    # ========== 生成统一的通讯结果 ==========
    print("\nExport communication CSVs...")

    # 生成单一CSV文件，包含所有需要的列
    unified_comm_path = os.path.join(output_dir, "lr_communication.csv")
    if export_unified:
        with open(unified_comm_path, 'w') as f:
            f.write("src_spot_barcode,dst_spot_barcode,source_cell,target_cell,lr_pair,original_lr_score,attention_score\n")

            # 使用完整数据，先计算哪些边是真正的cell-cell边
            src_nodes_full = all_edges_full[0]
            dst_nodes_full = all_edges_full[1]
            n_spots_sub_arr = all_n_spots_sub_full

            # mask where both src and dst are cell nodes (their node indices >= n_spots_sub for that edge)
            cell_cell_mask = (src_nodes_full >= n_spots_sub_arr) & (dst_nodes_full >= n_spots_sub_arr)
            total_cell_cell_edges = int(cell_cell_mask.sum().item())
            print(f"Unified export:     cell-cell edges={total_cell_cell_edges}/{all_edges_full.size(1)}")

            generated_rows = 0

            # Show examples for debugging if nothing gets generated
            first_bad_examples = 0

            # If there are no edges matching the per-edge n_spots_sub rule (rare), try a fallback strategy
            if total_cell_cell_edges == 0:
                try:
                    min_n_spots_sub = int(n_spots_sub_arr.min().item())
                    print(f"WARNING: fallback to global min n_spots_sub={min_n_spots_sub} for cell-cell masking")
                    cell_cell_mask = (src_nodes_full >= min_n_spots_sub) & (dst_nodes_full >= min_n_spots_sub)
                    total_cell_cell_edges = int(cell_cell_mask.sum().item())
                    print(f"Unified export:     fallback cell-cell edges={total_cell_cell_edges}/{all_edges_full.size(1)}")
                except Exception:
                    print("WARNING: fallback strategy failed")

            # If still zero, perform a permissive mapping where we try both subtraction and non-subtraction
            if total_cell_cell_edges == 0:
                print("WARNING: no cell-cell edges found; trying permissive mapping")
                candidate_indices = []
                for idx in range(all_edges_full.size(1)):
                    src_idx = int(src_nodes_full[idx].item())
                    dst_idx = int(dst_nodes_full[idx].item())
                    n_sp = int(n_spots_sub_arr[idx].item())
                    # Try both schemes
                    for subtract in (True, False):
                        s_idx = src_idx - n_sp if subtract else src_idx
                        d_idx = dst_idx - n_sp if subtract else dst_idx
                        if 0 <= s_idx < n_cells and 0 <= d_idx < n_cells:
                            candidate_indices.append(idx)
                            break

                cell_cell_mask = torch.zeros_like(src_nodes_full, dtype=torch.bool)
                if candidate_indices:
                    cell_cell_mask[candidate_indices] = True
                    total_cell_cell_edges = int(len(candidate_indices))
                    print(f"Unified export:     permissive cell-cell edges={total_cell_cell_edges}/{all_edges_full.size(1)}")
            cell_cell_indices = torch.where(cell_cell_mask)[0].tolist()
            skipped_out_of_range = 0
            skipped_missing_mapping = 0

            for idx in cell_cell_indices:
                src_idx = int(src_nodes_full[idx].item())
                dst_idx = int(dst_nodes_full[idx].item())
                center_spot_idx = int(all_spots_full[idx].item())
                n_spots_sub = int(n_spots_sub_arr[idx].item())
                cell_node_mapping = all_cell_node_mappings_flat[idx]

                # Compute cell indices relative to subgraph
                src_cell_local_idx = src_idx - n_spots_sub
                dst_cell_local_idx = dst_idx - n_spots_sub

                if not isinstance(cell_node_mapping, dict):
                    skipped_missing_mapping += 1
                    continue

                # Use mapping to get cell type ids
                if src_cell_local_idx in cell_node_mapping and dst_cell_local_idx in cell_node_mapping:
                    src_cell_type_id = cell_node_mapping[src_cell_local_idx]
                    dst_cell_type_id = cell_node_mapping[dst_cell_local_idx]
                    
                    # Map cell type ids to cell names
                    if src_cell_type_id < len(all_cell_names) and dst_cell_type_id < len(all_cell_names):
                        src_cell = all_cell_names[src_cell_type_id]
                        dst_cell = all_cell_names[dst_cell_type_id]
                    else:
                        skipped_out_of_range += 1
                        continue
                else:
                    skipped_missing_mapping += 1
                    continue

                lr_score = float(all_attrs_full[idx, 0].item())
                lr_id = int(all_attrs_full[idx, 1].item())
                attention_score = float(all_scores_full[idx].item())
                edge_logits = attention_score

                # Get LR pair name
                if lr_id in lr_id_to_pair:
                    ligand, receptor = lr_id_to_pair[lr_id]
                    lr_pair_name = f"{ligand}_{receptor}"
                else:
                    lr_pair_name = f"lr_{lr_id}"

                # Map spot barcodes
                if all_src_barcodes_flat is not None and all_dst_barcodes_flat is not None and idx < len(all_src_barcodes_flat):
                    src_barcode = all_src_barcodes_flat[idx]
                    dst_barcode = all_dst_barcodes_flat[idx]
                elif spot_names is not None and center_spot_idx < len(spot_names):
                    src_barcode = spot_names[center_spot_idx]
                    dst_barcode = spot_names[center_spot_idx]
                else:
                    src_barcode = str(center_spot_idx)
                    dst_barcode = str(center_spot_idx)

                f.write(f"{src_barcode},{dst_barcode},{src_cell},{dst_cell},{lr_pair_name},{lr_score:.6f},{attention_score:.6f}\n")
                generated_rows += 1

            if skipped_out_of_range or skipped_missing_mapping:
                skipped_total = skipped_out_of_range + skipped_missing_mapping
                print(
                    f"Unified export:     skipped={skipped_total} (out_of_range={skipped_out_of_range}, missing_mapping={skipped_missing_mapping})"
                )
            print(f"Unified CSV saved:  {unified_comm_path} (rows={generated_rows})")

        # 生成按注意力阈值过滤后的通讯结果
        filtered_comm_path = os.path.join(output_dir, f"lr_communication_filtered_{attention_threshold}.csv")
        with open(filtered_comm_path, 'w') as f:
            f.write("src_spot_barcode,dst_spot_barcode,source_cell,target_cell,lr_pair,original_lr_score,attention_score\n")

            # 过滤后的数据
            src_nodes_f = filtered_edges[0]
            dst_nodes_f = filtered_edges[1]
            n_spots_sub_f = all_n_spots_sub_full[keep_mask]

            cell_cell_mask_f = (src_nodes_f >= n_spots_sub_f) & (dst_nodes_f >= n_spots_sub_f)
            total_cell_cell_edges_f = int(cell_cell_mask_f.sum().item())
            print(f"Filtered export:    cell-cell edges={total_cell_cell_edges_f}/{filtered_edges.size(1)}")

            keep_indices = keep_mask.nonzero(as_tuple=False).view(-1).tolist()
            filtered_cell_node_mappings = [all_cell_node_mappings_flat[i] for i in keep_indices]
            if all_src_barcodes_flat is not None and all_dst_barcodes_flat is not None:
                filtered_src_barcodes = [all_src_barcodes_flat[i] for i in keep_indices]
                filtered_dst_barcodes = [all_dst_barcodes_flat[i] for i in keep_indices]
            else:
                filtered_src_barcodes = None
                filtered_dst_barcodes = None

            generated_rows = 0
            cell_cell_indices_f = torch.where(cell_cell_mask_f)[0].tolist()
            for idx in cell_cell_indices_f:
                src_idx = int(src_nodes_f[idx].item())
                dst_idx = int(dst_nodes_f[idx].item())
                n_spots_sub = int(n_spots_sub_f[idx].item())
                center_spot_idx = int(filtered_spots[idx].item())
                cell_node_mapping = filtered_cell_node_mappings[idx]

                src_cell_local_idx = src_idx - n_spots_sub
                dst_cell_local_idx = dst_idx - n_spots_sub
                if src_cell_local_idx not in cell_node_mapping or dst_cell_local_idx not in cell_node_mapping:
                    continue

                src_cell_type_id = cell_node_mapping[src_cell_local_idx]
                dst_cell_type_id = cell_node_mapping[dst_cell_local_idx]
                if src_cell_type_id >= len(all_cell_names) or dst_cell_type_id >= len(all_cell_names):
                    continue

                src_cell = all_cell_names[src_cell_type_id]
                dst_cell = all_cell_names[dst_cell_type_id]

                lr_score = float(filtered_attrs[idx, 0].item())
                lr_id = int(filtered_attrs[idx, 1].item())
                attention_score = float(filtered_scores[idx].item())

                if lr_id in lr_id_to_pair:
                    ligand, receptor = lr_id_to_pair[lr_id]
                    lr_pair_name = f"{ligand}_{receptor}"
                else:
                    lr_pair_name = f"lr_{lr_id}"

                if filtered_src_barcodes is not None and filtered_dst_barcodes is not None and idx < len(filtered_src_barcodes):
                    src_barcode = filtered_src_barcodes[idx]
                    dst_barcode = filtered_dst_barcodes[idx]
                elif spot_names is not None and center_spot_idx < len(spot_names):
                    src_barcode = spot_names[center_spot_idx]
                    dst_barcode = spot_names[center_spot_idx]
                else:
                    src_barcode = str(center_spot_idx)
                    dst_barcode = str(center_spot_idx)

                f.write(f"{src_barcode},{dst_barcode},{src_cell},{dst_cell},{lr_pair_name},{lr_score:.6f},{attention_score:.6f}\n")
                generated_rows += 1

        print(f"Filtered CSV saved: {filtered_comm_path}")
    else:
        print("Export skipped:     export_unified=False")

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
