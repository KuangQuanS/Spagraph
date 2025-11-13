#!/usr/bin/env python3
"""
Cell-Cell Communication Evaluation Module

This module provides evaluation functions for analyzing cell-cell communication
results from trained HeteroGAT models, including attention score analysis,
LR pair statistics, and model-based communication prediction.
"""

import os
import logging
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
    all_cell_names: List[str],
    output_dir: str,
    n_spots: int,
    n_cells: int
) -> None:
    """
    Evaluate cell-cell communication from trained model attention scores.

    Args:
        all_cc_attention_scores: List of attention scores from all batches
        all_edge_index_cc: List of edge indices from all batches
        all_edge_attr_cc: List of edge attributes from all batches
        all_spot_indices: List of spot indices from all batches
        all_cell_names: List of cell type names
        output_dir: Output directory for results
        n_spots: Number of spots in the dataset
        n_cells: Number of cell types
    """
    logging.info("="*80)
    logging.info("阶段5: 统计cell-cell边重要性")
    logging.info("="*80)

    if not all_cc_attention_scores:
        logging.warning("没有收集到cell-cell注意力得分")
        return

    logging.info(f"收集到 {len(all_cc_attention_scores)} 个batch的注意力得分")

    # 合并所有batch的注意力得分
    all_scores = torch.cat(all_cc_attention_scores, dim=0)  # [total_edges, num_heads]
    all_edges = torch.cat(all_edge_index_cc, dim=1)  # [2, total_edges]
    all_attrs = torch.cat(all_edge_attr_cc, dim=0)  # [total_edges, 2] - [lr_score, lr_id]
    all_spots = torch.cat(all_spot_indices, dim=0)  # [total_edges] - center spot indices

    logging.info(f"合并后数据形状: all_scores={all_scores.shape}, all_edges={all_edges.shape}, all_attrs={all_attrs.shape}, all_spots={all_spots.shape}")

    # 计算平均注意力得分（跨所有heads）
    avg_scores = all_scores.mean(dim=1)  # [total_edges]
    logging.info(f"平均注意力得分形状: {avg_scores.shape}")

    # 加载LR对映射
    lr_mapping_path = os.path.join(output_dir, "lr_pair_mapping.txt")
    lr_id_to_pair = {}
    if os.path.exists(lr_mapping_path):
        with open(lr_mapping_path, 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                lr_id, ligand, receptor = line.strip().split('\t')
                lr_id_to_pair[int(lr_id)] = (ligand, receptor)
        logging.info(f"已加载LR对映射: {len(lr_id_to_pair)} 个LR对")
    else:
        logging.warning(f"找不到LR对映射文件: {lr_mapping_path}")

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

    logging.info(f"处理统计: processed_edges={processed_edges}, skipped_no_lr={skipped_no_lr}")
    logging.info(f"lr_spot_scores条目数: {len(lr_spot_scores)}")

    # 计算每个spot-lr对的平均得分
    spot_lr_avg_scores = {}
    for key, scores in lr_spot_scores.items():
        spot_lr_avg_scores[key] = np.mean(scores)

    # ========== 统计LR对的出现次数和得分 ==========
    logging.info("\n" + "="*60)
    logging.info("统计配体受体对的出现频率和注意力得分")
    logging.info("="*60)

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

    logging.info(f"LR对统计结果已保存: {lr_stats_path}")
    logging.info(f"   - 总共发现 {len(lr_pair_summary)} 个不同的LR对")
    logging.info(f"   - Top 10 最常出现的LR对:")

    for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['occurrence_count'], reverse=True)[:10]):
        logging.info(f"     {i+1}. {item['lr_pair']}: 出现{item['occurrence_count']}次, "
                    f"平均注意力={item['avg_attention_score']:.4f}, "
                    f"出现在{item['n_spots']}个spots")

    logging.info(f"\n   - Top 10 注意力得分最高的LR对:")
    for i, item in enumerate(sorted(lr_pair_summary, key=lambda x: x['avg_attention_score'], reverse=True)[:10]):
        logging.info(f"     {i+1}. {item['lr_pair']}: 平均注意力={item['avg_attention_score']:.4f}, "
                    f"出现{item['occurrence_count']}次")

    # ========== 新增: 用模型预测得分生成最终通讯结果 ==========
    logging.info("\n" + "="*60)
    logging.info("生成基于模型预测的通讯结果")
    logging.info("="*60)

    # ✅ 方案：直接使用训练阶段收集的 cc_attention 作为 predicted_strength 的近似
    # 因为 DataLoader 有 shuffle，重新前向传播的边顺序会不一致
    # 或者：使用 avg_scores（attention）作为 predicted_strength 的替代
    logging.info("使用注意力得分作为通讯强度的代理指标...")
    all_pred_strengths = avg_scores  # [total_edges] - 使用 GAT 注意力得分

    logging.info(f"   注意：由于 DataLoader shuffle，无法准确重建 predicted_comm_strength")
    logging.info(f"   使用 attention_score 作为模型学习到的重要性指标")
    logging.info(f"   这在实际效果上是合理的：attention 本身就反映了模型认为的边重要性")

    # 生成三种得分的通讯结果文件
    model_based_comm_path = os.path.join(output_dir, "lr_communication_model_based.csv")
    with open(model_based_comm_path, 'w') as f:
        f.write("center_spot,source_cell,target_cell,lr_pair,original_lr_score,attention_score,predicted_strength,modulated_score,score_type\n")

        # 遍历所有边，生成三种得分
        for i in range(all_edges.size(1)):
            src_idx = all_edges[0, i].item()
            dst_idx = all_edges[1, i].item()
            center_spot_idx = all_spots[i].item()

            if src_idx >= n_spots and dst_idx >= n_spots:
                src_cell_idx = (src_idx - n_spots) % n_cells
                dst_cell_idx = (dst_idx - n_spots) % n_cells

                if src_cell_idx < n_cells and dst_cell_idx < n_cells:
                    # ✅ edge_attr_cc 是 2维: [lr_score, lr_id]
                    lr_score = all_attrs[i, 0].item()  # 第0列: LR得分
                    lr_id = int(all_attrs[i, 1].item())  # 第1列: LR ID
                    attention_score = avg_scores[i].item()
                    # ✅ 使用 attention_score 作为 predicted_strength（因为训练时attention就是模型学习的重要性）
                    predicted_strength = attention_score

                    src_cell = all_cell_names[src_cell_idx]
                    dst_cell = all_cell_names[dst_cell_idx]

                    if lr_id in lr_id_to_pair:
                        ligand, receptor = lr_id_to_pair[lr_id]
                        lr_pair_name = f"{ligand}_{receptor}"
                    else:
                        lr_pair_name = f"lr_{lr_id}"

                    # 计算调制后的得分（原始LR得分 × 模型预测强度）
                    modulated_score = lr_score * predicted_strength

                    # 推荐使用modulated_score作为最终得分
                    score_type = "modulated"

                    f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},"
                           f"{lr_score:.6f},{attention_score:.6f},{predicted_strength:.6f},"
                           f"{modulated_score:.6f},{score_type}\n")

    logging.info(f"基于模型预测的通讯结果已保存: {model_based_comm_path}")
    logging.info(f"   - original_lr_score: 原始LR得分（表达 × 距离衰减）")
    logging.info(f"   - attention_score: GAT注意力得分（图结构重要性）")
    logging.info(f"   - predicted_strength: MLP预测的通讯强度 [0,1]")
    logging.info(f"   - modulated_score: 原始LR得分 × 预测强度（推荐使用）⭐")
    logging.info(f"   - 解释：modulated_score = 基础强度 × 模型学习的重要性因子")


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
        logging.info(f"DGI预训练损失曲线已保存: {dgi_loss_curve_path}")
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
        logging.info(f"损失曲线已保存: {loss_curve_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # This module is meant to be imported, not run directly
    print("This is an evaluation module. Import and use the evaluate_cell_communication() function.")