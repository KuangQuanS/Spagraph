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
    all_n_spots_sub: List[torch.Tensor],
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
        all_n_spots_sub: List of n_spots_sub values from all batches
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
        logging.warning(f"all_cc_attention_scores长度: {len(all_cc_attention_scores)}")
        logging.warning(f"all_edge_index_cc长度: {len(all_edge_index_cc)}")
        logging.warning(f"all_edge_attr_cc长度: {len(all_edge_attr_cc)}")
        logging.warning(f"all_spot_indices长度: {len(all_spot_indices)}")
        return

    logging.info(f"收集到 {len(all_cc_attention_scores)} 个spots的注意力得分")

    # 合并所有batch的注意力得分
    all_scores = torch.cat(all_cc_attention_scores, dim=0)  # [total_edges]
    all_edges = torch.cat(all_edge_index_cc, dim=1)  # [2, total_edges]
    all_attrs = torch.cat(all_edge_attr_cc, dim=0)  # [total_edges, 5] - [lr_score, lr_id, is_important, exist_logits, rate_pred]
    all_spots = torch.cat(all_spot_indices, dim=0)  # [total_edges] - center spot indices
    all_n_spots_sub_batch = torch.cat(all_n_spots_sub, dim=0)  # [total_edges] - n_spots_sub for each edge

    logging.info(f"合并后数据形状: all_scores={all_scores.shape}, all_edges={all_edges.shape}, all_attrs={all_attrs.shape}, all_spots={all_spots.shape}")
    n_unique_spots = len(torch.unique(all_spots))
    logging.info(f"数据统计: 总边数={all_scores.shape[0]}, 唯一spot数={n_unique_spots}, 平均每spot边数={all_scores.shape[0]/n_unique_spots:.1f}")
    logging.info(f"   - 唯一源细胞类型数={torch.unique(all_edges[0]).shape[0]}, 唯一目标细胞类型数={torch.unique(all_edges[1]).shape[0]}")

    # 注意：all_scores现在是1维的[total_edges]，直接使用即可
    avg_scores = all_scores  # [total_edges] - 已经是平均后的注意力得分
    
    # ✅ 提取双头预测结果
    exist_logits = all_attrs[:, 3]  # [total_edges] - 边存在性logits
    rate_pred = all_attrs[:, 4]  # [total_edges] - 边强度预测
    p_exist = torch.sigmoid(exist_logits)  # [total_edges] - 边存在概率
    
    logging.info(f"注意力得分形状: {avg_scores.shape}")
    logging.info(f"边存在性概率分布: min={p_exist.min():.3f}, mean={p_exist.mean():.3f}, max={p_exist.max():.3f}")
    logging.info(f"边存在性概率 > 0.5: {(p_exist > 0.5).sum()}/{len(p_exist)} ({(p_exist > 0.5).sum()/len(p_exist)*100:.1f}%)")

    # ========== 应用边过滤策略：Top-K per source node ==========
    # 为每个源节点（spot-cell对）保留top-k条边，去除假阳性
    logging.info("\n" + "="*60)
    logging.info("应用Top-K边过滤策略")
    logging.info("="*60)

    edge_topk = 5  # 每个源节点最多保留5条边
    keep_mask = torch.zeros(len(p_exist), dtype=torch.bool)

    # 按源节点分组
    unique_sources = torch.unique(all_edges[0])
    for src_node in unique_sources:
        src_edges_mask = (all_edges[0] == src_node)
        src_p_exist = p_exist[src_edges_mask]

        k = min(edge_topk, src_p_exist.size(0))
        if k > 0:
            # 获取top-k边的索引
            topk_values, topk_indices = torch.topk(src_p_exist, k)

            # 将全局索引中对应的位置标记为保留
            src_global_indices = torch.where(src_edges_mask)[0]
            keep_mask[src_global_indices[topk_indices]] = True

    # 应用过滤
    filtered_scores = avg_scores[keep_mask]
    filtered_edges = all_edges[:, keep_mask]
    filtered_attrs = all_attrs[keep_mask]
    filtered_spots = all_spots[keep_mask]
    filtered_p_exist = p_exist[keep_mask]
    filtered_rate_pred = rate_pred[keep_mask]

    logging.info(f"过滤前边数: {len(p_exist)}")
    logging.info(f"过滤后边数: {keep_mask.sum()} ({keep_mask.sum()/len(p_exist)*100:.1f}% 保留)")
    logging.info(f"平均每个源节点保留: {keep_mask.sum()/len(unique_sources):.1f} 条边")

    # ✅ 保存过滤前的完整数据，用于生成 model_based_comm_path
    all_scores_full = avg_scores.clone()  # 保存过滤前的完整注意力得分
    all_edges_full = all_edges.clone()    # 保存过滤前的完整边索引
    all_attrs_full = all_attrs.clone()    # 保存过滤前的完整边属性
    all_spots_full = all_spots.clone()    # 保存过滤前的完整spot索引
    all_n_spots_sub_full = all_n_spots_sub_batch.clone()  # 保存过滤前的完整n_spots_sub

    # ✅ 使用过滤后的数据替换原始数据（用于其他分析）
    avg_scores = filtered_scores
    all_edges = filtered_edges
    all_attrs = filtered_attrs
    all_spots = filtered_spots

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
    
    # ========== 保存清洁图边结果（带p_exist和rate_pred）==========
    cleaned_edges_path = os.path.join(output_dir, "lr_communication_edge_attention.csv")
    with open(cleaned_edges_path, 'w') as f:
        f.write("center_spot,source_cell,target_cell,lr_pair,lr_id,lr_score,is_important_label,p_exist,rate_pred,attention_score\n")
        
        for i in range(all_edges.size(1)):
            src_idx = all_edges[0, i].item()
            dst_idx = all_edges[1, i].item()
            center_spot_idx = all_spots[i].item()
            
            # ✅ 修复：将全局节点ID转换为细胞类型索引
            # 在子图中，细胞节点的索引是从 n_spots_sub 开始的
            # 但我们需要知道每个边的 n_spots_sub 值
            n_spots_sub = all_n_spots_sub_full[i].item()  # 从完整数据中获取
            
            if src_idx >= n_spots_sub and dst_idx >= n_spots_sub:
                src_cell_idx = src_idx - n_spots_sub
                dst_cell_idx = dst_idx - n_spots_sub
                
                if src_cell_idx >= 0 and dst_cell_idx >= 0 and src_cell_idx < n_cells and dst_cell_idx < n_cells:
                    lr_score = all_attrs[i, 0].item()
                    lr_id = int(all_attrs[i, 1].item())
                    is_important = int(all_attrs[i, 2].item())
                    p_exist_val = filtered_p_exist[i].item()
                    rate_pred_val = filtered_rate_pred[i].item()
                    attention_score = avg_scores[i].item()
                    
                    src_cell = all_cell_names[src_cell_idx]
                    dst_cell = all_cell_names[dst_cell_idx]
                    
                    if lr_id in lr_id_to_pair:
                        ligand, receptor = lr_id_to_pair[lr_id]
                        lr_pair_name = f"{ligand}_{receptor}"
                    else:
                        lr_pair_name = f"lr_{lr_id}"
                    
                    f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},{lr_id},"
                           f"{lr_score:.6f},{is_important},{p_exist_val:.6f},{rate_pred_val:.6f},{attention_score:.6f}\n")
    
    logging.info(f"清洁图边结果已保存: {cleaned_edges_path}")
    logging.info(f"   - 保留边数: {all_edges.size(1)}")
    logging.info(f"   - 包含列: lr_score, is_important_label, p_exist, rate_pred, attention_score")
    
    logging.info(f"\n   - Top 10 最常出现的LR对:")
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

    all_pred_strengths = all_scores_full  # [total_edges] - 使用完整的 GAT 注意力得分

    # 生成三种得分的通讯结果文件
    model_based_comm_path = os.path.join(output_dir, "lr_communication_model_based.csv")
    with open(model_based_comm_path, 'w') as f:
        f.write("center_spot,source_cell,target_cell,lr_pair,original_lr_score,edge_logits,adjusted_score,score_type\n")

        # 遍历所有边（使用完整数据），生成三种得分
        generated_rows = 0
        for i in range(all_edges_full.size(1)):
            src_idx = all_edges_full[0, i].item()
            dst_idx = all_edges_full[1, i].item()
            center_spot_idx = all_spots_full[i].item()
            n_spots_sub = all_n_spots_sub_full[i].item()

            # ✅ 修复：在子图中，细胞节点的索引是从 n_spots_sub 开始的
            if src_idx >= n_spots_sub and dst_idx >= n_spots_sub:
                src_cell_idx = src_idx - n_spots_sub
                dst_cell_idx = dst_idx - n_spots_sub

                if src_cell_idx >= 0 and dst_cell_idx >= 0 and src_cell_idx < n_cells and dst_cell_idx < n_cells:
                    # 获取边属性
                    lr_score = all_attrs_full[i, 0].item()
                    lr_id = int(all_attrs_full[i, 1].item())
                    exist_logits = all_attrs_full[i, 3].item()
                    
                    # 计算三种得分
                    original_lr_score = lr_score
                    edge_logits = exist_logits
                    adjusted_score = original_lr_score * (1 + np.tanh(edge_logits))
                    
                    # 获取细胞名称
                    src_cell = all_cell_names[src_cell_idx]
                    dst_cell = all_cell_names[dst_cell_idx]
                    
                    # 获取LR对名称
                    if lr_id in lr_id_to_pair:
                        ligand, receptor = lr_id_to_pair[lr_id]
                        lr_pair_name = f"{ligand}_{receptor}"
                    else:
                        lr_pair_name = f"lr_{lr_id}"
                    
                    # 写入三种得分
                    f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},{original_lr_score:.6f},{edge_logits:.6f},{adjusted_score:.6f},original\n")
                    f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},{original_lr_score:.6f},{edge_logits:.6f},{adjusted_score:.6f},logits\n")
                    f.write(f"{center_spot_idx},{src_cell},{dst_cell},{lr_pair_name},{original_lr_score:.6f},{edge_logits:.6f},{adjusted_score:.6f},adjusted\n")
                    
                    generated_rows += 3
                else:
                    if generated_rows == 0 and i < 5:  # 只打印前5个不符合条件的边
                        logging.warning(f"边 {i} 细胞索引超出范围: src_cell_idx={src_cell_idx}, dst_cell_idx={dst_cell_idx}, n_cells={n_cells}")
            else:
                if generated_rows == 0 and i < 5:  # 只打印前5个不符合条件的边
                    logging.warning(f"边 {i} 节点索引不符合条件: src_idx={src_idx}, dst_idx={dst_idx}, n_spots_sub={n_spots_sub}")
        
        logging.info(f"生成数据行数: {generated_rows}")

    logging.info(f"基于模型预测的通讯结果已保存: {model_based_comm_path}")
    logging.info(f"   - 总边数: {all_edges_full.size(1)}")

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