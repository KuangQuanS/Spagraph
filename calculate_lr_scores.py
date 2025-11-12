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
    cell_expr: pd.DataFrame,
    lr_pairs: List[Tuple[str, str]],
    spot_cell_expr_npz_path: str,
    output_dir: str
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Calculate KNN neighborhoods and LR communication scores.

    Args:
        spot_coords: Spot coordinates array [n_spots, 2]
        composition: Cell composition matrix [n_spots, n_cells]
        args: Arguments object containing various parameters
        adata: ST data object (AnnData)
        cell_expr: Cell expression matrix [n_cells, n_genes]
        lr_pairs: List of LR pairs [(ligand, receptor), ...]
        spot_cell_expr_npz_path: Path to spot-cell expression NPZ file
        output_dir: Output directory for results

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

    # 构建KNN邻域
    N = len(spot_coords)
    n_neighbors = args.n_spot_neighbors
    logging.info(f"准备构建KNN: N={N} spots, k={n_neighbors} neighbors")

    # 构建KNN图
    logging.info(f"构建KNN图: {N} spots, {n_neighbors} neighbors")
    knn = kneighbors_graph(spot_coords, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    knn_mask = knn.toarray()  # [N, N]

    # 添加物理距离限制
    distance_threshold = 200.0  # 200μm
    for i in range(N):
        for j in range(N):
            if knn_mask[i, j] == 1:
                dist = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 +
                              (spot_coords[i, 1] - spot_coords[j, 1])**2)
                if dist > distance_threshold:
                    knn_mask[i, j] = 0

    logging.info(f"应用距离过滤: 最大 {distance_threshold}μm")

    # ✅ 使用 output_dir 保存 KNN mask
    knn_npz_path = os.path.join(output_dir, "knn_mask.npz")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        knn_npz_path,
        knn_mask=knn_mask
    )

    logging.info(f"KNN邻接矩阵已保存到: {knn_npz_path}")

    # ✅ 计算并保存LR通讯得分矩阵（使用 output_dir）
    logging.info("开始计算LR通讯得分...")
    logging.info(f"   - 活跃基因筛选: 表达比例≥10% 且 normalize_total(1e4) > {args.mean_expr_threshold}")
    logging.info(f"   - 通讯得分过滤阈值: {args.lr_comm_score_threshold}")
    logging.info(f"   - 距离衰减参数sigma: {args.lr_distance_sigma}")

    # ✅ 调试：检查KNN mask状态
    logging.info(f"   - KNN mask形状: {knn_mask.shape}")
    logging.info(f"   - KNN mask中非零元素数: {np.count_nonzero(knn_mask)}")
    logging.info(f"   - 平均每个spot的邻居数: {np.count_nonzero(knn_mask) / N:.2f}")

    # 加载spot-cell表达数据
    npz_data = np.load(spot_cell_expr_npz_path, allow_pickle=True)
    spot_cell_expr_array = npz_data['spot_cell_expr']
    cell_names_in_npz = list(npz_data['cell_names'])
    gene_names_in_npz = list(npz_data['gene_names'])

    # 构建查询字典: {(spot_idx, cell_idx): array_row_idx}
    spot_names = adata.obs_names.tolist()
    cell_names_list = cell_expr.index.tolist()
    spot_cell_expr_index = {}

    for idx, name in enumerate(cell_names_in_npz):
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            spot_name, cell_name = parts
            try:
                cell_id = cell_names_list.index(cell_name)
                spot_id = spot_names.index(spot_name)
                spot_cell_expr_index[(spot_id, cell_id)] = idx
            except ValueError:
                pass

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

    # ========== 新增：筛选每个细胞类型中的活跃基因 ==========
    # 活跃基因定义：表达比例≥10% 且 normalize_total(1e4) > threshold
    logging.info("筛选每个细胞类型中的活跃基因...")
    cell_active_genes_ligand = {}  # cell_name -> set of active ligand gene indices
    cell_active_genes_receptor = {}  # cell_name -> set of active receptor gene indices

    expr_proportion_threshold = 0.1  # 10%的细胞中表达
    mean_expr_threshold = args.mean_expr_threshold  # ✅ 使用超参数，配体和受体统一阈值

    for cell_idx, cell_name in enumerate(cell_names_list):
        # 收集该细胞类型的所有表达数据
        cell_exprs = []
        for spot_idx in range(N):
            spot_name = spot_names[spot_idx]
            cell_combined_name = f"{spot_name}_{cell_name}"
            if cell_combined_name in cell_names_in_npz:
                array_idx = cell_names_in_npz.index(cell_combined_name)
                cell_exprs.append(spot_cell_expr_array[array_idx])

        if not cell_exprs:
            cell_active_genes_ligand[cell_name] = set()
            cell_active_genes_receptor[cell_name] = set()
            continue

        cell_expr_matrix = np.array(cell_exprs)  # [n_cells_of_this_type, n_genes]

        # ✅ 归一化 - 每个细胞normalize到总和为10000
        cell_totals = cell_expr_matrix.sum(axis=1, keepdims=True)  # [n_cells, 1]
        cell_totals[cell_totals == 0] = 1  # 避免除以0
        cell_expr_normalized = cell_expr_matrix / cell_totals * 1e4  # [n_cells, n_genes]

        # 计算每个基因的统计信息
        # 1. 表达比例：在多少比例的细胞中表达 (原始expr > 0)
        expr_proportion = np.mean(cell_expr_matrix > 0, axis=0)  # [n_genes]

        # 2. 标准化后的平均表达：mean(normalized_expr)
        mean_expr = np.mean(cell_expr_normalized, axis=0)  # [n_genes]

        # 3. 过滤掉不参与通讯的基因
        filtered_gene_indices = set()
        for gene_name in filtered_genes:
            if gene_name in gene_names_in_npz:
                gene_idx = gene_names_in_npz.index(gene_name)
                filtered_gene_indices.add(gene_idx)

        # ✅ 筛选活跃配体基因：表达比例≥10% 且 平均表达 > threshold（统一阈值）
        active_mask_ligand = (expr_proportion >= expr_proportion_threshold) & \
                             (mean_expr > mean_expr_threshold)
        active_ligand_indices = set(np.where(active_mask_ligand)[0]) - filtered_gene_indices

        # ✅ 筛选活跃受体基因：表达比例≥10% 且 平均表达 > threshold（统一阈值）
        active_mask_receptor = (expr_proportion >= expr_proportion_threshold) & \
                               (mean_expr > mean_expr_threshold)
        active_receptor_indices = set(np.where(active_mask_receptor)[0]) - filtered_gene_indices

        cell_active_genes_ligand[cell_name] = active_ligand_indices
        cell_active_genes_receptor[cell_name] = active_receptor_indices

        logging.info(f"   - {cell_name}: 配体={len(active_ligand_indices)}, 受体={len(active_receptor_indices)} "
                    f"(过滤前: {np.sum(active_mask_ligand)}/{np.sum(active_mask_receptor)})")

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

    n_cells = len(cell_names_list)

    # ========== 优化3：使用进度条和批量处理 ==========
    logging.info("   - 开始遍历KNN邻居对...")
    total_pairs = 0
    spots_with_cells = 0
    spots_without_cells = 0
    same_celltype_skipped = 0  # 统计跳过的同类型细胞对

    # 遍历所有KNN邻居对
    for i in range(N):
        spot_i_barcode = spot_names[i]  # 获取spot i的barcode

        # 获取spot i的cell composition
        composition_i = composition.iloc[i].values
        cell_in_i = np.where(composition_i > 1e-6)[0]

        if len(cell_in_i) == 0:
            spots_without_cells += 1
            continue

        spots_with_cells += 1

        for j in range(N):
            if knn_mask[i, j] == 0:  # 不是邻居，跳过
                continue

            total_pairs += 1
            spot_j_barcode = spot_names[j]  # 获取spot j的barcode

            # 获取spot j的cell composition
            composition_j = composition.iloc[j].values
            cell_in_j = np.where(composition_j > 1e-6)[0]

            if len(cell_in_j) == 0:
                continue

            # 遍历cell对，计算LR通讯
            for cell_i_idx in cell_in_i:
                idx_i = spot_cell_expr_index.get((i, cell_i_idx))
                if idx_i is None:
                    continue

                cell_i_expr = spot_cell_expr_array[idx_i, :]
                cell_i_name = cell_names_list[cell_i_idx]  # 获取cell名称
                cell_i = f"{spot_i_barcode}_{cell_i_name}"  # 组合成cell名称

                for cell_j_idx in cell_in_j:
                    idx_j = spot_cell_expr_index.get((j, cell_j_idx))
                    if idx_j is None:
                        continue

                    cell_j_expr = spot_cell_expr_array[idx_j, :]
                    cell_j_name = cell_names_list[cell_j_idx]  # 获取cell名称
                    cell_j = f"{spot_j_barcode}_{cell_j_name}"  # 组合成cell名称

                    # ✅ 跳过相同细胞类型之间的通讯
                    if cell_i_name == cell_j_name:
                        same_celltype_skipped += 1
                        continue

                    # ========== 优化4：向量化LR得分计算 ==========
                    # 使用预处理的LR索引，避免重复查找
                    for lig_idx, rec_indices, ligand, receptor in valid_lr_pairs:
                        # 检查配体基因是否在源细胞中活跃（配体阈值 > 0.5）
                        if lig_idx not in cell_active_genes_ligand[cell_i_name]:
                            continue

                        # 检查所有受体基因是否在目标细胞中活跃（受体阈值 > 0.3）
                        receptor_active = all(rec_idx in cell_active_genes_receptor[cell_j_name] for rec_idx in rec_indices)
                        if not receptor_active:
                            continue

                        # 获取配体和受体的表达值
                        lig_val = cell_i_expr[lig_idx]
                        rec_vals = cell_j_expr[rec_indices]

                        # 计算受体乘积（联合受体取乘积）
                        rec_product = np.prod(rec_vals)

                        # 计算spot间距离权重
                        distance = np.sqrt((spot_coords[i, 0] - spot_coords[j, 0])**2 +
                                            (spot_coords[i, 1] - spot_coords[j, 1])**2)
                        distance_weight = np.exp(-distance / args.lr_distance_sigma)

                        # 计算通讯得分：几何平均数 × 距离权重
                        score = np.sqrt(lig_val * rec_product) * distance_weight

                        # ✅ 过滤低于阈值的通讯事件
                        if score >= args.lr_comm_score_threshold:
                            comm_event_records.append([
                                spot_i_barcode, spot_j_barcode, cell_i, cell_j, ligand, receptor, score
                            ])

        # 每处理100个spot打印一次进度
        if (i + 1) % 100 == 0:
            logging.info(f"   - 已处理 {i+1}/{N} spots, 发现 {len(comm_event_records)} 个通讯事件")

    logging.info(f"计算完成: {len(comm_event_records)} 个LR通讯事件")
    logging.info(f"   - Spots with cells: {spots_with_cells}/{N}")
    logging.info(f"   - Spots without cells: {spots_without_cells}/{N}")
    logging.info(f"   - 处理的邻居对: {total_pairs}")
    logging.info(f"   - 跳过同类型细胞对: {same_celltype_skipped}")

    # ✅ 如果设置了阈值，输出过滤统计
    if args.lr_comm_score_threshold > 0:
        logging.info(f"   - 通讯得分阈值过滤: score >= {args.lr_comm_score_threshold}")

    # ✅ 使用 output_dir 保存 LR 通讯得分
    csv_path = os.path.join(output_dir, "lr_scores.csv")

    df = pd.DataFrame(
        comm_event_records,
        columns=['spot_i', 'spot_j', 'cell_i', 'cell_j', 'ligand', 'receptor', 'comm_score']
    )
    df.to_csv(csv_path, index=False)
    logging.info(f"LR通讯得分已保存到: {csv_path}")
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