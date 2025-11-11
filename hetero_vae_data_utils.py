import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import csv
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.neighbors import NearestNeighbors


class STHeteroSubgraphDataset:
    """空间转录组异构图子图数据集 - 以每个spot为中心构建k邻域子图"""
    
    def __init__(self, st_h5ad_path: str, cluster_expr: pd.DataFrame, 
                 cell_expr: pd.DataFrame, cell_full_expr: pd.DataFrame,
                 graph_data: Dict, lr_pairs: List[Tuple[str, str]],
                 k_neighbors: int = 10,
                 expr_threshold: float = 1.0,
                 spot_cell_expr_npz_path: Optional[str] = None,
                 load_lr_scores_csv: Optional[str] = None,
                 min_comm_edges: int = 1,
                 device: str = 'cpu'):
        """
        初始化数据集
        Args:
            st_h5ad_path: ST数据路径
            cluster_expr: Cluster marker基因表达量DataFrame [n_clusters, n_marker_genes]
            cell_expr: Cell marker基因表达量DataFrame [n_cells, n_marker_genes]
            cell_full_expr: Cell全基因表达量DataFrame [n_cells, n_all_genes]
            graph_data: 图数据（包含coords, composition, knn_mask等）
            lr_pairs: [(ligand, receptor), ...] 配体受体对列表
            k_neighbors: 邻域spot数量
            expr_threshold: LR配体受体的表达量阈值 (default: 1.0)
            spot_cell_expr_npz_path: 预构建的spot-cell-gene表达矩阵路径
            load_lr_scores_csv: 加载预先计算的LR通讯得分CSV路径
                               CSV格式: spot_i, spot_j, cell_i, cell_j, ligand, receptor, comm_score
            min_comm_edges: 最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)
            device: 设备
        """
        self.min_comm_edges = min_comm_edges
        self.cluster_expr = cluster_expr
        self.cell_expr = cell_expr
        self.cell_full_expr = cell_full_expr
        self.graph_data = graph_data
        self.device = device
        self.k_neighbors = k_neighbors
        self.lr_pairs = lr_pairs
        self.expr_threshold = expr_threshold
        
        # LR通讯结果字典 - 用于快速查询（从CSV加载）
        self.lr_scores_dict = {}  # {(spot_i, spot_j, ct_i, ct_j, ligand, receptor): score, ...}
        
        # LR对编码映射
        self.lr_pair_to_id = {}  # {(ligand, receptor): lr_id}
        self.lr_id_to_pair = {}  # {lr_id: (ligand, receptor)}
        
        # 加载ST数据
        self.adata = sc.read_h5ad(st_h5ad_path)
        self.n_spots = self.adata.n_obs
        
        # 获取spot坐标（用于构建邻域）
        self.coords = graph_data.get('coords', None)
        if self.coords is None:
            raise ValueError("coords必须在graph_data中提供")
        
        # 获取composition矩阵
        self.composition = graph_data.get('composition', None)
        if self.composition is None:
            raise ValueError("composition必须在graph_data中提供")
        
        # 获取预计算的KNN邻接矩阵
        self.knn_mask = graph_data.get('knn_mask', None)
        
        # 获取ST中与cluster表达量匹配的基因
        self.genes = cluster_expr.columns.tolist()
        self.st_X = self.adata[:, self.genes].X
        if hasattr(self.st_X, 'toarray'):
            self.st_X = self.st_X.toarray()
        
        # 从knn_mask构建knn_indices
        self.knn_indices = []
        for i in range(self.n_spots):
            neighbors = np.where(self.knn_mask[i] == 1)[0]
            # 确保每个spot都有固定数量的邻居，邻居不够时重复现有邻居来填充
            if len(neighbors) >= self.k_neighbors:
                neighbor_indices = neighbors[:self.k_neighbors]
            elif len(neighbors) > 0:
                # 邻居不够，重复现有邻居来填充
                n_repeats = (self.k_neighbors + len(neighbors) - 1) // len(neighbors)  # 向上取整
                repeated_neighbors = np.tile(neighbors, n_repeats)
                neighbor_indices = repeated_neighbors[:self.k_neighbors]
            else:
                # 没有邻居，用自己填充
                neighbor_indices = np.full(self.k_neighbors, i, dtype=int)
            self.knn_indices.append(neighbor_indices)
        self.knn_indices = np.array(self.knn_indices)  # [n_spots, k_neighbors]
        
        # marker基因表达用于embedding计算
        self.cell_marker_expr = cell_expr
        
        # 加载预构建的spot-cell-gene表达矩阵
        self.spot_cell_expr = None
        self.cell_names = None
        self.gene_names = None
        self.spot_cell_expr_index = {}  # {(spot_id, cell_id): row_idx}
        
        if spot_cell_expr_npz_path is not None:
            self._load_spot_cell_expr(spot_cell_expr_npz_path)
        else:
            print("[Warning] spot_celltype_expr_npz_path not provided.")
        
        # 加载预先计算的LR通讯得分（如果提供）
        if load_lr_scores_csv is not None:
            self._load_lr_scores_from_csv(load_lr_scores_csv)
        
        # ========== 预计算基因索引映射，避免每次都查找 ==========
        self.marker_gene_to_idx = {}
        for gene_name in self.cell_marker_expr.columns:
            if gene_name in self.gene_names:
                self.marker_gene_to_idx[gene_name] = self.gene_names.index(gene_name)
        
        # ========== 确保spot和cell基因顺序一致 ==========
        # 创建从cluster_expr基因顺序到cell_marker_expr基因顺序的映射
        self.spot_gene_order_to_cell = []
        for gene_name in self.cell_marker_expr.columns:
            if gene_name in self.genes:
                self.spot_gene_order_to_cell.append(self.genes.index(gene_name))
            else:
                # 如果cell marker基因不在cluster基因中，用-1表示缺失
                self.spot_gene_order_to_cell.append(-1)
        
        print(f"[Optimized] Pre-computed gene index mapping for {len(self.marker_gene_to_idx)} marker genes")
        print(f"[Aligned] Spot and cell gene orders aligned to {len(self.spot_gene_order_to_cell)} marker genes")
    
    def _load_spot_cell_expr(self, npz_path: str):
        """
        加载预构建的spot-cell-gene表达矩阵
        
        Args:
            npz_path: npz文件路径，包含：
                - spot_cell_expr: [n_spot_cells, n_genes] 2D数组，行名为"spot_name_cell_name"
                - cell_names: cell名称列表，格式为["spot1_c1", "spot1_c2", ..., "spot2_c1", ...]
                - gene_names: 基因名称列表
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"[Warning] Spot-cell expression file not found: {npz_path}")
            return
        
        print(f"[Loading] Spot-cell expression from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        
        # 支持向后兼容：新版本使用'spot_cell_expr'，旧版本使用'spot_celltype_expr'
        if 'spot_cell_expr' in data:
            self.spot_cell_expr = data['spot_cell_expr']  # [n_spot_cells, n_genes]
            self.cell_names = list(data['cell_names'])  # ["spot1_c1", "spot1_c2", ...]
        else:
            raise KeyError("NPZ file must contain either 'spot_cell_expr' or 'spot_celltype_expr' key")
        
        self.gene_names = list(data['gene_names'])
        
        # 构建快速查询字典：{(spot_name, cell_id): row_idx_in_array}
        self.spot_cell_expr_index = {}
        for idx, name in enumerate(self.cell_names):
            parts = name.rsplit('_', 1)  # 从右边分割，分离出spot_name和cell_name
            if len(parts) == 2:
                spot_name, cell_name = parts
                # 查找cell_id
                try:
                    cell_id = self.cell_expr.index.tolist().index(cell_name)
                    # 查找spot的全局索引
                    spot_id = self.adata.obs_names.tolist().index(spot_name)
                    self.spot_cell_expr_index[(spot_id, cell_id)] = idx
                except ValueError:
                    pass
        
        print(f"[Loaded] Spot-cell expression shape: {self.spot_cell_expr.shape}")
        print(f"[Loaded] Cells: {len(self.cell_names)} entries")
        print(f"[Loaded] Genes: {len(self.gene_names)}")
    
    def _load_lr_scores_from_csv(self, csv_path: str):
        """
        从CSV文件加载预先计算的LR通讯得分
        
        Args:
            csv_path: CSV文件路径
                     CSV格式: spot_i, spot_j, cell_i, cell_j, ligand, receptor, comm_score
                     其中 spot_i/spot_j 是spot的barcode
                     cell_i/cell_j 是 "barcode_celltype" 格式
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[Warning] LR scores CSV file not found: {csv_path}")
            return
        
        print(f"[Loading] LR communication scores from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 获取spot名称列表和cell名称列表
        spot_names = self.adata.obs_names.tolist()
        cell_names_list = self.cell_expr.index.tolist()
        
        # 构建查询字典：{(spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx, ligand, receptor): score}
        lr_id_counter = 0
        for idx, row in df.iterrows():
            spot_i_barcode = row['spot_i']
            spot_j_barcode = row['spot_j']
            cell_i = row['cell_i']  # "barcode_cell"
            cell_j = row['cell_j']  # "barcode_cell"
            
            # 从cell_i/cell_j中提取cell名称
            # cell_i格式: "barcode_cell"
            cell_i_name = cell_i.rsplit('_', 1)[-1] if '_' in cell_i else None
            cell_j_name = cell_j.rsplit('_', 1)[-1] if '_' in cell_j else None
            
            if cell_i_name is None or cell_j_name is None:
                continue
            
            try:
                # 转换为索引
                spot_i_idx = spot_names.index(spot_i_barcode)
                spot_j_idx = spot_names.index(spot_j_barcode)
                cell_i_idx = cell_names_list.index(cell_i_name)
                cell_j_idx = cell_names_list.index(cell_j_name)
                
                ligand = row['ligand']
                receptor = row['receptor']
                
                # 为LR对分配唯一ID
                lr_pair = (ligand, receptor)
                if lr_pair not in self.lr_pair_to_id:
                    self.lr_pair_to_id[lr_pair] = lr_id_counter
                    self.lr_id_to_pair[lr_id_counter] = lr_pair
                    lr_id_counter += 1
                
                key = (spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx, ligand, receptor)
                self.lr_scores_dict[key] = float(row['comm_score'])
            except ValueError:
                # 如果barcode或cell名称找不到，跳过这条记录
                continue
        
        print(f"[Loaded] Total LR score entries: {len(self.lr_scores_dict)}")
        print(f"[Loaded] Unique LR pairs: {len(self.lr_pair_to_id)}")
        
        # 保存LR对映射到文件
        lr_mapping_path = csv_path.parent / "lr_pair_mapping.txt"
        with open(lr_mapping_path, 'w') as f:
            f.write("lr_id\tligand\treceptor\n")
            for lr_id, (ligand, receptor) in self.lr_id_to_pair.items():
                f.write(f"{lr_id}\t{ligand}\t{receptor}\n")
        print(f"[Saved] LR pair mapping to {lr_mapping_path}")
        
        # ========== 预处理LR查询优化 ==========
        # 构建按spot-cell对分组的LR键索引，避免每次都遍历所有LR对
        self.lr_keys_by_spot_pair = {}  # {(spot_i, spot_j, cell_i, cell_j): [key1, key2, ...]}
        for key in self.lr_scores_dict.keys():
            spot_i, spot_j, cell_i, cell_j, ligand, receptor = key
            pair_key = (spot_i, spot_j, cell_i, cell_j)
            if pair_key not in self.lr_keys_by_spot_pair:
                self.lr_keys_by_spot_pair[pair_key] = []
            self.lr_keys_by_spot_pair[pair_key].append(key)
        
        print(f"[Optimized] LR keys grouped by {len(self.lr_keys_by_spot_pair)} spot-cell pairs")
        
        # ========== 预过滤：只保留有足够通讯边的spot ==========
        self._filter_valid_spots(self.min_comm_edges)
    
    def _filter_valid_spots(self, min_comm_edges: int):
        """
        预先过滤掉没有通讯边（或通讯边太少）的spot
        
        Args:
            min_comm_edges: 最小通讯边数阈值（默认1，即至少有1条通讯边）
        """
        print(f"\n[Filtering] 检查每个spot的通讯边数量...")
        valid_spots = []
        spot_comm_counts = []
        
        for spot_idx in range(self.n_spots):
            # 统计该spot作为中心点的通讯边数
            comm_edge_count = 0
            
            # 获取邻居
            neighbor_indices = self.knn_indices[spot_idx]
            
            # 获取中心spot的cell composition
            composition_center = self.composition.iloc[spot_idx].values
            cells_in_center = np.where(composition_center > 1e-6)[0]
            
            if len(cells_in_center) == 0:
                spot_comm_counts.append(0)
                continue
            
            # 检查与每个邻居的通讯
            for neighbor_idx in neighbor_indices:
                if self.knn_mask is not None and not self.knn_mask[spot_idx, neighbor_idx]:
                    continue
                
                composition_neighbor = self.composition.iloc[neighbor_idx].values
                cells_in_neighbor = np.where(composition_neighbor > 1e-6)[0]
                
                if len(cells_in_neighbor) == 0:
                    continue
                
                # 检查cell-cell通讯
                for cell_i in cells_in_center:
                    for cell_j in cells_in_neighbor:
                        pair_key = (spot_idx, neighbor_idx, cell_i, cell_j)
                        if pair_key in self.lr_keys_by_spot_pair:
                            comm_edge_count += len(self.lr_keys_by_spot_pair[pair_key])
            
            spot_comm_counts.append(comm_edge_count)
            if comm_edge_count >= min_comm_edges:
                valid_spots.append(spot_idx)
        
        self.valid_spot_indices = np.array(valid_spots)
        self.spot_comm_counts = np.array(spot_comm_counts)
        
        print(f"[Filtering] 统计结果:")
        print(f"   - 总spot数: {self.n_spots}")
        print(f"   - 有通讯边的spot: {len(valid_spots)} ({len(valid_spots)/self.n_spots*100:.1f}%)")
        print(f"   - 无通讯边的spot: {self.n_spots - len(valid_spots)} ({(self.n_spots-len(valid_spots))/self.n_spots*100:.1f}%)")
        print(f"   - 平均通讯边数: {np.mean(spot_comm_counts):.1f}")
        print(f"   - 中位数通讯边数: {np.median(spot_comm_counts):.1f}")
        print(f"   - 最大通讯边数: {np.max(spot_comm_counts)}")
        
        if len(valid_spots) < self.n_spots * 0.5:
            print(f"[Warning] 超过50%的spot没有通讯边，请检查LR通讯得分阈值设置")
    
    def __len__(self):
        # 返回有效spot数量，而不是全部spot数量
        return len(self.valid_spot_indices)
    
    def __getitem__(self, idx):
        """获取单个子图样本 - 固定celltype数量的异构子图
        
        返回：
            dict包含：
            - center_spot_idx: 中心spot全局索引
            - subgraph_spot_indices: 子图中所有spot的全局索引 [center, neighbor1, ..., neighbork]
            - expr_raw: [k+1, n_marker_genes] 所有spot的marker基因原始表达谱（顺序与cell一致）
            - cell_expr_raw: [(k+1)*n_cells, n_marker_genes] 所有cell的原始marker基因表达谱
            - cell_full_expr: [n_cells, n_all_genes] 所有cell的全基因表达（用于LR计算）
            - edge_index_ss: [2, num_edges] spot-spot边
            - edge_attr_ss: [num_edges] spot-spot权重
            - edge_index_sc: [2, num_edges] spot-cell边（固定9个cell节点）
            - edge_attr_sc: [num_edges] spot-cell权重（由composition决定，无则为0）
            - edge_index_cc: [2, num_edges] cell-cell边
            - edge_attr_cc: [num_edges] cell-cell权重
            - coords_subgraph: [k+1, 2] 子图中spot的坐标
            - composition_subgraph: [k+1, n_cells] 子图中spot的composition（完整9个）
        """
        # ========== Step 1: 获取邻域spot（从有效spot列表中映射）==========
        # idx是有效spot列表中的索引，需要映射到原始spot索引
        center_idx = self.valid_spot_indices[idx]
        neighbor_indices = self.knn_indices[center_idx]
        subgraph_spot_indices = np.concatenate([[center_idx], neighbor_indices])
        n_spots_sub = len(subgraph_spot_indices)
        
        # ========== Step 2: 获取spot表达量（原始表达谱，只保留marker基因，顺序与cell一致） ==========
        expr_vecs = self.st_X[subgraph_spot_indices].astype(np.float32)  # [k+1, n_marker_genes_cluster]
        
        # 重新排列基因顺序，使其与cell marker基因顺序一致
        expr_marker = np.zeros((expr_vecs.shape[0], len(self.spot_gene_order_to_cell)), dtype=np.float32)
        for i, gene_idx in enumerate(self.spot_gene_order_to_cell):
            if gene_idx >= 0:
                expr_marker[:, i] = expr_vecs[:, gene_idx]
            # 如果基因缺失，保持为0
        
        expr_raw = torch.tensor(expr_marker, dtype=torch.float32)  # [k+1, n_marker_genes] 直接返回原始表达谱
        
        # ========== Step 3: 获取cell表达量（优化版本）==========
        cell_expr_raw_list = []
        n_cells = len(self.cell_marker_expr)
        
        # 预先获取所有需要的基因位置
        marker_gene_positions = []
        for gene_name in self.cell_marker_expr.columns:
            pos = self.marker_gene_to_idx.get(gene_name, -1)
            marker_gene_positions.append(pos)
        
        # 对子图中每个spot的每个cell获取表达向量
        for spot_global_idx in subgraph_spot_indices:
            for cell_id in range(n_cells):
                # 查找该spot该cell的表达向量
                idx_in_array = self.spot_cell_expr_index.get((spot_global_idx, cell_id))
                
                if idx_in_array is not None:
                    # 从spot-cell表达矩阵中提取完整表达
                    spot_cell_full_expr = self.spot_cell_expr[idx_in_array, :]
                    
                    # 只取marker基因 - 使用预计算的位置
                    cell_marker_expr = np.zeros(len(marker_gene_positions), dtype=np.float32)
                    for gene_idx, gene_pos in enumerate(marker_gene_positions):
                        if gene_pos >= 0:
                            cell_marker_expr[gene_idx] = spot_cell_full_expr[gene_pos]
                else:
                    # 如果找不到，用全局cell marker表达作为备选
                    cell_marker_expr = self.cell_marker_expr.iloc[cell_id].values.astype(np.float32)
                
                cell_expr_raw_list.append(cell_marker_expr)
        
        cell_expr_raw = torch.tensor(
            np.array(cell_expr_raw_list), dtype=torch.float32
        )  # [(k+1)*n_cells, n_marker_genes]
        
        # ========== Step 4: 获取composition矩阵 ==========
        coords_sub = self.coords[subgraph_spot_indices]
        composition_subgraph_full = self.composition.iloc[subgraph_spot_indices].values.astype(np.float32)
        # [k+1, n_cells]
        
        # ========== Step 5: 构建相似度边（Spot-Spot + Spot-Cell） ==========
        # 统一处理所有表示相似度的边：物理距离和权重都代表某种相似度
        n_cells = self.cell_expr.shape[0]
        edge_index_like_list = []
        edge_attr_like_list = []
        
        # 5a. Spot-Spot边（基于物理坐标KNN距离）
        for i in range(n_spots_sub):
            for j in range(n_spots_sub):
                if i != j:
                    dist = np.linalg.norm(coords_sub[i] - coords_sub[j])
                    sigma = 50.0
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                    if weight > 1e-6:
                        edge_index_like_list.append([i, j])
                        edge_attr_like_list.append(weight)
        
        # 5b. Spot-Cell边（基于composition权重）
        # 节点编号方案：
        # Spot节点: 0 ~ n_spots_sub-1
        # Cell节点: n_spots_sub ~ n_spots_sub + n_spots_sub*n_cells - 1
        #   Spot_i的cell_j节点编号 = n_spots_sub + i*n_cells + j
        
        for spot_id in range(n_spots_sub):
            for cell_id in range(n_cells):
                comp = composition_subgraph_full[spot_id, cell_id]
                weight = np.sqrt(comp)  # 权重转换
                if weight > 1e-6:
                    # Spot_i的Cell_j节点编号 = n_spots_sub + i*n_cells + j
                    cell_node_id = n_spots_sub + spot_id * n_cells + cell_id
                    edge_index_like_list.append([spot_id, cell_node_id])
                    edge_attr_like_list.append(weight)
        
        edge_index_like = torch.tensor(
            np.array(edge_index_like_list).T if edge_index_like_list else np.array([[], []]).astype(int),
            dtype=torch.long
        )
        edge_attr_like = torch.tensor(
            edge_attr_like_list if edge_attr_like_list else [],
            dtype=torch.float32
        )

        # ========== Step 7: 构建Celltype-Celltype边 ==========
        # 只计算中心点发出的通讯边，避免重复（其他点作为中心点时会计算）
        # 注意：LR通讯得分已经预计算在CSV中，直接从字典查询

        edge_index_cc_list = []
        edge_attr_cc_list = []  # 现在包含两个特征：[lr_score, lr_id]

        # 只计算中心点（i_local=0）与其他点的通讯
        center_local_idx = 0
        center_global_idx = subgraph_spot_indices[center_local_idx]
        composition_center = composition_subgraph_full[center_local_idx]
        cells_in_center = np.where(composition_center > 1e-6)[0]

        for j_local in range(1, n_spots_sub):  # 从1开始，跳过自己
            spot_j_global = subgraph_spot_indices[j_local]
            
            # 直接使用knn_mask检查邻居关系，避免预处理knn_indices的重复填充问题
            if self.knn_mask is not None:
                if not self.knn_mask[center_global_idx, spot_j_global]:
                    continue
            else:
                # 如果没有knn_mask，回退到使用knn_indices检查
                center_neighbors_set = set(self.knn_indices[center_global_idx])
                if spot_j_global not in center_neighbors_set:
                    continue
                    
            composition_j = composition_subgraph_full[j_local]
            cells_in_j = np.where(composition_j > 1e-6)[0]
            
            # 计算中心点cell到邻居点cell的通讯边
            for cell_i in cells_in_center:
                for cell_j in cells_in_j:
                    # 优化：直接查询预计算的LR得分，避免遍历所有LR对
                    pair_key = (center_global_idx, spot_j_global, cell_i, cell_j)
                    lr_keys = self.lr_keys_by_spot_pair.get(pair_key, [])
                    
                    total_lr_score = 0.0
                    lr_ids = []
                    for key in lr_keys:
                        spot_i, spot_j, ct_i, ct_j, ligand, receptor = key
                        total_lr_score += self.lr_scores_dict[key]
                        lr_id = self.lr_pair_to_id[(ligand, receptor)]
                        lr_ids.append(lr_id)
                    
                    if total_lr_score > 1e-6 and lr_ids:
                        # 使用第一个LR ID作为代表（如果有多个LR对，取平均得分但用第一个ID）
                        lr_id = lr_ids[0]
                        
                        cell_i_node_id = n_spots_sub + center_local_idx * n_cells + cell_i
                        cell_j_node_id = n_spots_sub + j_local * n_cells + cell_j
                        edge_index_cc_list.append([cell_i_node_id, cell_j_node_id])
                        # 边特征：[lr_score, lr_id]
                        edge_attr_cc_list.append([total_lr_score, lr_id])
                        
        edge_index_cc = torch.tensor(
            np.array(edge_index_cc_list).T if edge_index_cc_list else np.array([[], []]).astype(int),
            dtype=torch.long
        )
        edge_attr_cc = torch.tensor(
            edge_attr_cc_list if edge_attr_cc_list else [],
            dtype=torch.float32
        )
        
        # ========== Step 8: 整理返回数据 ==========
        coords_subgraph = torch.tensor(coords_sub, dtype=torch.float32)
        composition_subgraph = torch.tensor(composition_subgraph_full, dtype=torch.float32)
        
        return {
            'center_spot_idx': center_idx,
            'subgraph_spot_indices': subgraph_spot_indices,
            'n_spots_sub': n_spots_sub,
            'n_cells': n_cells,
            'expr_raw': expr_raw,  # [k+1, n_genes] 原始spot表达谱
            'cell_expr_raw': cell_expr_raw,  # [(k+1)*n_cells, n_marker_genes] 原始cell表达谱
            'edge_index_like': edge_index_like,  # 相似度边 (spot-spot + spot-cell)
            'edge_attr_like': edge_attr_like,
            'edge_index_cc': edge_index_cc,  # cell-cell通讯边
            'edge_attr_cc': edge_attr_cc,
            'coords_subgraph': coords_subgraph,
            'composition_subgraph': composition_subgraph,
        }


def hetero_subgraph_collate_fn(batch):
    """
    自定义 collate_fn 处理异构子图批次
    
    返回堆叠的多个子图，而不是合并成一个大图
    每个subgraph保持独立，通过列表或堆叠张量返回
    
    Args:
        batch: 列表，每个元素是 STHeteroSubgraphDataset 返回的字典
    
    Returns:
        batch_dict: 包含堆叠后数据的字典
    """
    batch_size = len(batch)
    n_cells = batch[0]['n_cells']
    n_spots_sub = batch[0]['n_spots_sub']
    
    # 堆叠表达量
    expr_raw_batch = torch.stack([sample['expr_raw'] for sample in batch], dim=0)  # [B, k+1, n_marker_genes]
    cell_expr_raw_batch = torch.stack([sample['cell_expr_raw'] for sample in batch], dim=0)  # [B, (k+1)*n_cells, n_marker_genes]
    
    # 边保持为列表（因为边数可能不同）
    edge_index_like_list = [sample['edge_index_like'] for sample in batch]  # list of [2, E_i]
    edge_attr_like_list = [sample['edge_attr_like'] for sample in batch]    # list of [E_i]
    edge_index_cc_list = [sample['edge_index_cc'] for sample in batch]      # list of [2, E_i]
    edge_attr_cc_list = [sample['edge_attr_cc'] for sample in batch]        # list of [E_i, 2] - [lr_score, lr_id]
    
    batch_dict = {
        'batch_size': batch_size,
        'n_spots_sub': n_spots_sub,
        'n_cells': n_cells,
        'center_spot_idx': [sample['center_spot_idx'] for sample in batch],
        'expr_raw': expr_raw_batch,  # [B, k+1, n_marker_genes]
        'cell_expr_raw': cell_expr_raw_batch,  # [B, (k+1)*n_cells, n_marker_genes]
        'edge_index_like': edge_index_like_list,  # list of [2, E_i]
        'edge_attr_like': edge_attr_like_list,    # list of [E_i]
        'edge_index_cc': edge_index_cc_list,      # list of [2, E_i]
        'edge_attr_cc': edge_attr_cc_list,        # list of [E_i]
    }
    
    return batch_dict
