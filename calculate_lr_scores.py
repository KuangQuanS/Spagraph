#!/usr/bin/env python3
"""
LR Scores Calculation Module

This module provides functions for calculating KNN neighborhoods and LR communication scores
for spatial transcriptomics data analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from typing import List, Tuple, Dict, Any, Optional


def calculate_lr_scores(
    spot_coords: np.ndarray,
    composition: Optional[pd.DataFrame],
    args: Any,
    adata: Any,
    cell_full_expr: pd.DataFrame,
    lr_pairs: List[Tuple[str, str]],
    output_dir: str,
    n_neighbors: int = 20
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Calculate KNN neighborhoods and LR communication scores.

    Args:
        spot_coords: Spot coordinates array [n_spots, 2]
        composition: Cell composition matrix [n_spots, n_cells]
        args: Arguments object containing various parameters
        adata: ST data object (AnnData)
        cell_full_expr: Cell full expression matrix [n_cells, n_genes]
        lr_pairs: List of LR pairs [(ligand, receptor), ...]
        output_dir: Output directory for results
        n_neighbors: Number of neighbors for KNN graph construction

    Returns:
        Tuple of (knn_mask, csv_path, graph_data)
    """
    logging.info("="*80)
    logging.info("阶段3.5: 预计算KNN邻域和LR通讯得分")
    logging.info("="*80)

    # ✅ 调试：检查关键变量状态
    logging.info(f"spot_coords类型: {type(spot_coords)}, 形状: {spot_coords.shape if spot_coords is not None else 'None'}")
    logging.info(f"composition类型: {type(composition)}, 形状: {composition.shape if composition is not None else 'None'}")
    logging.info(f"args.output_dir: {args.output_dir}")

    # 获取spots数量
    n_spots = spot_coords.shape[0]

    # 构建KNN邻域（用于空间图构建，不限制距离）
    logging.info(f"构建KNN图: {n_spots} spots, {n_neighbors} neighbors")
    knn = kneighbors_graph(spot_coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    knn_mask = knn.toarray()  # [n_spots, n_spots] - 用于空间图构建
    
    # ✅ 构建LR通信邻域mask（允许更大范围的通信）
    # 使用更大的距离阈值，允许细胞间通信在更远距离发生
    lr_comm_distance_threshold = 200.0  # 200μm - 允许更远距离的LR通信
    lr_comm_mask = np.zeros((n_spots, n_spots), dtype=bool)
    
    for i in range(n_spots):
        for j in range(n_spots):
            if i != j:
                dist = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 +
                              (spot_coords[i, 1] - spot_coords[j, 1])**2)
                if dist <= lr_comm_distance_threshold:
                    lr_comm_mask[i, j] = True

    # ✅ 计算并保存LR通讯得分矩阵（使用 output_dir）
    logging.info("开始计算LR通讯得分...")
    logging.info(f"   - 活跃基因筛选: normalize_total(1e4) > {args.mean_expr_threshold}")
    logging.info(f"   - 通讯得分过滤阈值: {args.mean_expr_threshold} (与活跃基因阈值一致)")
    logging.info(f"   - 通讯得分计算: 配体×受体表达相乘（无距离衰减）")

    # ✅ 调试：检查KNN mask状态
    logging.info(f"   - KNN mask形状: {knn_mask.shape}")
    logging.info(f"   - KNN mask中非零元素数: {np.count_nonzero(knn_mask)}")
    logging.info(f"   - 平均每个spot的邻居数: {np.count_nonzero(knn_mask) / n_spots:.2f}")
    logging.info(f"   - LR通信mask中非零元素数: {np.count_nonzero(lr_comm_mask)}")
    logging.info(f"   - LR通信平均每个spot的潜在邻居数: {np.count_nonzero(lr_comm_mask) / n_spots:.2f}")

    # ✅ 准备数据结构
    spot_names = adata.obs_names.tolist()
    spot_cell_expr_array = cell_full_expr.values  # [n_spot_cells, n_genes]
    spot_cell_names = cell_full_expr.index.tolist()  # ['spot_barcode_celltype', ...]
    gene_names_in_npz = cell_full_expr.columns.tolist()
    
    # 构建快速查询字典: spot_cell_name -> array_row_idx
    spot_cell_name_to_idx = {name: idx for idx, name in enumerate(spot_cell_names)}
    
    # 提取实际存在的细胞类型（从composition的列名）
    cell_types = composition.columns.tolist()
    
    logging.info(f"   - Spot数量: {n_spots}")
    logging.info(f"   - 细胞类型数: {len(cell_types)}")
    logging.info(f"   - Spot-cell总数: {len(spot_cell_names)}")

    # ========== 优化1：预构建基因索引字典 ==========
    # 避免在循环中重复查找基因索引
    gene_name_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_names_in_npz)}

    # ========== 过滤不参与通讯的基因 ==========
    logging.info("过滤不参与通讯的基因...")

    # 定义过滤规则
    filtered_genes = set()
    for gene_name in gene_names_in_npz:
        gene_upper = gene_name.upper()
        # 1. 线粒体基因 (MT-)
        if gene_upper.startswith('MT-'):
            filtered_genes.add(gene_name)
        # 2. 血红蛋白基因 (HB)
        elif gene_upper.startswith('HB'):
            filtered_genes.add(gene_name)
        # 3. 假基因 (包含 'PSEUDO', '-AS', 'LOC')
        elif 'PSEUDO' in gene_upper or gene_upper.endswith('-AS1') or gene_upper.startswith('LOC'):
            filtered_genes.add(gene_name)
        # 4. 核糖体蛋白基因 (RPS, RPL) - 可选
        # elif gene_upper.startswith('RPS') or gene_upper.startswith('RPL'):
        #     filtered_genes.add(gene_name)

    logging.info(f"   - 过滤基因数: {len(filtered_genes)}/{len(gene_names_in_npz)}")
    logging.info(f"   - 线粒体基因: {sum(1 for g in filtered_genes if g.upper().startswith('MT-'))}")
    logging.info(f"   - 血红蛋白基因: {sum(1 for g in filtered_genes if g.upper().startswith('HB'))}")
    logging.info(f"   - 假基因: {sum(1 for g in filtered_genes if 'PSEUDO' in g.upper() or g.upper().endswith('-AS1') or g.upper().startswith('LOC'))}")

    # ========== 筛选每个细胞中的活跃基因（表达值过滤）==========
    logging.info("筛选每个细胞中的活跃基因（基于表达阈值）...")
    cell_active_genes_ligand = {}  # spot_cell_key -> set of active ligand gene indices
    cell_active_genes_receptor = {}  # spot_cell_key -> set of active receptor gene indices

    mean_expr_threshold = args.mean_expr_threshold  # 表达阈值
    
    # 过滤掉不参与通讯的基因索引（提前计算）
    filtered_gene_indices = set()
    for gene_name in filtered_genes:
        if gene_name in gene_names_in_npz:
            gene_idx = gene_names_in_npz.index(gene_name)
            filtered_gene_indices.add(gene_idx)

    # 遍历每个具体的细胞，计算其活跃基因
    for spot_cell_name in spot_cell_names:
        idx = spot_cell_name_to_idx[spot_cell_name]
        cell_expr = spot_cell_expr_array[idx, :]  # [n_genes]
        
        # 对单个细胞进行normalize
        total_count = cell_expr.sum()
        if total_count > 0:
            cell_expr_normalized = cell_expr / total_count * 1e4  # [n_genes]
        else:
            cell_expr_normalized = cell_expr  # 表达全为0的情况
        
        # 筛选活跃基因：归一化后的表达值 > threshold
        active_mask = cell_expr_normalized > mean_expr_threshold
        active_gene_indices = set(np.where(active_mask)[0]) - filtered_gene_indices
        
        # 配体和受体使用相同的活跃基因集合
        cell_active_genes_ligand[spot_cell_name] = active_gene_indices
        cell_active_genes_receptor[spot_cell_name] = active_gene_indices

    logging.info(f"   - 处理了 {len(spot_cell_names)} 个细胞")
    logging.info(f"   - 平均每个细胞活跃基因数: {np.mean([len(genes) for genes in cell_active_genes_ligand.values()]):.1f}")

    # ========== 优化2：预处理LR对，构建索引映射 ==========
    # 将LR对转换为索引对，并过滤掉不存在的基因
    valid_lr_pairs = []
    for ligand, receptor in lr_pairs:
        ligand_upper = ligand.upper()
        lig_idx = gene_name_to_idx.get(ligand_upper)

        if lig_idx is None:
            continue

        # 处理联合受体
        receptor_genes = [r.strip() for r in receptor.split('_')]
        rec_indices = []
        found_all = True

        for receptor_gene in receptor_genes:
            receptor_upper = receptor_gene.upper()
            rec_idx = gene_name_to_idx.get(receptor_upper)
            if rec_idx is None:
                found_all = False
                break
            rec_indices.append(rec_idx)

        if found_all:
            valid_lr_pairs.append((lig_idx, rec_indices, ligand, receptor))

    logging.info(f"   - 有效LR对: {len(valid_lr_pairs)}/{len(lr_pairs)}")

    # 初始化通讯事件记录
    comm_event_records = []

    # ========== 优化3：使用进度条和批量处理 ==========
    logging.info("   - 开始遍历LR通信邻居对（基于KNN空间邻居）...")
    total_pairs = 0
    spots_with_cells = 0
    spots_without_cells = 0
    same_celltype_skipped = 0  # 统计跳过的同类型细胞对

    # 遍历所有潜在的LR通信邻居对（使用knn_mask限制为空间邻居）
    for i in range(n_spots):
        spot_i_barcode = spot_names[i]

        # 获取spot i的cell composition
        composition_i = composition.iloc[i].values
        cell_in_i = np.where(composition_i > 1e-6)[0]

        if len(cell_in_i) == 0:
            spots_without_cells += 1
            continue

        spots_with_cells += 1

        for j in range(n_spots):
            if knn_mask[i, j] == 0:  # 只在KNN邻居之间计算LR通讯
                continue

            total_pairs += 1
            spot_j_barcode = spot_names[j]

            # 获取spot j的cell composition
            composition_j = composition.iloc[j].values
            cell_in_j = np.where(composition_j > 1e-6)[0]

            if len(cell_in_j) == 0:
                continue

            # 遍历cell对，计算LR通讯
            for cell_i_idx in cell_in_i:
                # ✅ 获取细胞类型名称（从composition的列名）
                celltype_i = cell_types[cell_i_idx]
                spot_cell_i_key = f"{spot_i_barcode}_{celltype_i}"
                
                # 查询该spot-cell是否存在
                if spot_cell_i_key not in spot_cell_name_to_idx:
                    continue
                
                idx_i = spot_cell_name_to_idx[spot_cell_i_key]
                cell_i_expr = spot_cell_expr_array[idx_i, :]

                for cell_j_idx in cell_in_j:
                    # ✅ 获取细胞类型名称
                    celltype_j = cell_types[cell_j_idx]
                    spot_cell_j_key = f"{spot_j_barcode}_{celltype_j}"
                    
                    # 查询该spot-cell是否存在
                    if spot_cell_j_key not in spot_cell_name_to_idx:
                        continue
                    
                    idx_j = spot_cell_name_to_idx[spot_cell_j_key]
                    cell_j_expr = spot_cell_expr_array[idx_j, :]

                    # ✅ 跳过相同细胞类型之间的通讯
                    if celltype_i == celltype_j:
                        same_celltype_skipped += 1
                        continue

                    # ========== 优化4：向量化LR得分计算 ==========
                    # 使用预处理的LR索引，避免重复查找
                    for lig_idx, rec_indices, ligand, receptor in valid_lr_pairs:
                        # 检查配体基因是否在源细胞中活跃
                        if lig_idx not in cell_active_genes_ligand[spot_cell_i_key]:
                            continue

                        # 检查所有受体基因是否在目标细胞中活跃
                        receptor_active = all(rec_idx in cell_active_genes_receptor[spot_cell_j_key] for rec_idx in rec_indices)
                        if not receptor_active:
                            continue

                        # 获取配体和受体的表达值
                        lig_val = cell_i_expr[lig_idx]
                        rec_vals = cell_j_expr[rec_indices]

                        # 计算受体乘积（联合受体取乘积）
                        rec_product = np.prod(rec_vals)

                        # ✅ 计算归一化通讯得分（基于各细胞的总表达量标准化）
                        total_i = cell_i_expr.sum()
                        total_j = cell_j_expr.sum()
                        if total_i > 0 and total_j > 0:
                            # 归一化因子：1e4 / sqrt(total_i * total_j)
                            normalization_factor = 1e4 / np.sqrt(total_i * total_j)
                            score = np.sqrt(lig_val * rec_product) * normalization_factor
                        else:
                            score = 0

                        # ✅ 过滤低于阈值的通讯事件
                        if score >= args.mean_expr_threshold:
                            # ✅ 计算距离，用于后续伪标签生成
                            distance = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 +
                                              (spot_coords[i, 1] - spot_coords[j, 1])**2)
                            
                            # ✅ 先记录是否在 KNN mask 内
                            in_knn = 1 if knn_mask[i, j] == 1 else 0
                            
                            comm_event_records.append([
                                spot_i_barcode, spot_j_barcode, 
                                spot_cell_i_key, spot_cell_j_key,  # ✅ 使用完整的spot_cell名称
                                ligand, receptor, score, in_knn, distance
                            ])

        # 每处理100个spot打印一次进度
        if (i + 1) % 100 == 0:
            logging.info(f"   - 已处理 {i+1}/{n_spots} spots, 发现 {len(comm_event_records)} 个通讯事件")

    logging.info(f"计算完成: {len(comm_event_records)} 个LR通讯事件")
    logging.info(f"   - Spots with cells: {spots_with_cells}/{n_spots}")
    logging.info(f"   - Spots without cells: {spots_without_cells}/{n_spots}")
    logging.info(f"   - 处理的邻居对: {total_pairs}")
    logging.info(f"   - 跳过同类型细胞对: {same_celltype_skipped}")

    # ✅ 如果设置了阈值，输出过滤统计
    if args.mean_expr_threshold > 0:
        logging.info(f"   - 通讯得分阈值过滤: score >= {args.mean_expr_threshold}")

    # ✅ 使用 output_dir 保存 LR 通讯得分
    csv_path = os.path.join(output_dir, "lr_scores.csv")

    df = pd.DataFrame(
        comm_event_records,
        columns=['spot_i', 'spot_j', 'cell_i', 'cell_j', 'ligand', 'receptor', 'comm_score', 'in_knn', 'distance']
    )
    
    df.to_csv(csv_path, index=False)
    logging.info(f"\nLR通讯得分已保存到: {csv_path}")
    logging.info(f"   - 总事件数: {len(df)}")
    logging.info(f"   - Spot对数: {df.groupby(['spot_i', 'spot_j']).ngroups}")
    logging.info(f"   - Cell对数: {df.groupby(['cell_i', 'cell_j']).ngroups}")

    # 准备graph_data字典（只包含坐标和composition）
    graph_data = {
        'coords': spot_coords,
        'composition': composition,
        'knn_mask': knn_mask,  # 传入预计算的KNN邻接矩阵
    }

    return knn_mask, csv_path, graph_data


if __name__ == '__main__':
    # This module is meant to be imported, not run directly
    print("This is an LR scores calculation module. Import and use the calculate_lr_scores() function.")