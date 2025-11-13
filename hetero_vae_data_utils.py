import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.neighbors import NearestNeighbors
import random

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        """
        Args:
            patience: 在多少个epoch内没有改善就停止
            min_delta: 最小改善阈值
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, val_loss):
        """
        Args:
            epoch: 当前epoch
            val_loss: 当前损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            # 损失有显著改善
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 损失没有显著改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置cudnn，使其确定性运行
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 移除所有已有的 handler（避免重复打印）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件handler，写日志到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建控制台handler，兼容tqdm
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 配置tqdm logger避免冲突
    tqdm_logger = logging.getLogger('tqdm')
    tqdm_logger.setLevel(logging.WARNING)  # 只显示警告和错误
    tqdm_logger.addHandler(console_handler)
    tqdm_logger.propagate = False  # 防止向上传播

class STHeteroSubgraphDataset:
    """空间转录组异构图子图数据集 - 以每个spot为中心构建k邻域子图"""
    
    def __init__(self, st_h5ad_path: str, cluster_expr: pd.DataFrame, 
                 cell_expr: pd.DataFrame, cell_full_expr: pd.DataFrame,
                 graph_data: Dict, lr_pairs: List[Tuple[str, str]],
                 k_neighbors: int = 10,
                 expr_threshold: float = 1.0,
                 load_lr_scores_csv: Optional[str] = None,
                 min_comm_edges: int = 1,
                 valid_cell_types: Optional[List[str]] = None,
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
            load_lr_scores_csv: 加载预先计算的LR通讯得分CSV路径
                               CSV格式: spot_i, spot_j, cell_i, cell_j, ligand, receptor, comm_score
            min_comm_edges: 最小通讯边数阈值，少于此值的spot将被过滤 (default: 1)
            valid_cell_types: 有效的细胞类型列表，用于过滤NPZ数据（可选）
            device: 设备
        """
        self.valid_cell_types = valid_cell_types
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
        
        # 用于过滤的辅助字典
        self.lr_keys_by_spot_pair = {}  # {(spot_i, spot_j, ct_i, ct_j): [keys]}
        
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
        
        # 从knn_mask构建knn_indices（允许变长邻居）
        # ✅ 修改：不再强制固定邻居数量，允许每个spot有不同数量的邻居
        self.knn_indices = []
        self.knn_neighbor_counts = []  # 记录每个spot实际的邻居数量
        
        for i in range(self.n_spots):
            neighbors = np.where(self.knn_mask[i] == 1)[0]
            
            if len(neighbors) > 0:
                # 保留所有在距离阈值内的邻居（不超过k_neighbors）
                neighbor_indices = neighbors[:self.k_neighbors] if len(neighbors) > self.k_neighbors else neighbors
            else:
                # 如果没有邻居，至少保留一个（避免孤立节点）
                # 找到距离最近的spot作为邻居
                neighbor_indices = np.array([i])  # 暂时用自己，后续会在构建边时处理
            
            self.knn_indices.append(neighbor_indices)
            self.knn_neighbor_counts.append(len(neighbor_indices))
        
        # ✅ 注意：knn_indices 现在是变长的list，不能直接转为array
        # self.knn_indices 保持为 list of arrays
        
        # marker基因表达用于embedding计算
        self.cell_marker_expr = cell_expr
        
        # 使用cell_expr作为细胞表达数据
        # 假设所有spot的相同细胞类型有相同的表达
        self.spot_cell_expr = cell_expr.values  # [n_cells, n_marker_genes]
        self.cell_names = cell_expr.index.tolist()
        self.gene_names = cell_expr.columns.tolist()
        self.spot_cell_expr_index = {}

        # 每个细胞类型在所有spot中共享表达
        spot_names = self.adata.obs_names.tolist()
        cell_names_list = self.cell_expr.index.tolist()
        for spot_idx in range(self.n_spots):
            for cell_idx in range(len(cell_names_list)):
                if self.composition.iloc[spot_idx, cell_idx] > 1e-6:
                    self.spot_cell_expr_index[(spot_idx, cell_idx)] = cell_idx
        
        if load_lr_scores_csv is not None:
            self._load_lr_scores_from_csv(load_lr_scores_csv)
        else:
            raise ValueError("必须提供load_lr_scores_csv")
        
        # 过滤有效spot
        self._filter_valid_spots(self.min_comm_edges)
        
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
    

    def _load_lr_scores_from_csv(self, csv_path: str):
        """从CSV文件加载预先计算的LR通讯得分"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到LR得分CSV文件: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 获取spot名称列表和cell名称列表
        spot_names = self.adata.obs_names.tolist()
        cell_names_list = self.cell_expr.index.tolist()
        
        # 构建查询字典：{(spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx, ligand, receptor): score}
        lr_id_counter = 1  # 从1开始，避免与相似度边的ID=0冲突
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
                
                # 同时更新lr_keys_by_spot_pair
                spot_pair_key = (spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx)
                if spot_pair_key not in self.lr_keys_by_spot_pair:
                    self.lr_keys_by_spot_pair[spot_pair_key] = []
                self.lr_keys_by_spot_pair[spot_pair_key].append(key)
            except ValueError:
                # 如果barcode或cell名称找不到，跳过这条记录
                continue
        
        print(f"[Loaded] Total LR score entries: {len(self.lr_scores_dict)}")
        print(f"[Loaded] Unique LR pairs: {len(self.lr_pair_to_id)}")
    
    def _filter_valid_spots(self, min_comm_edges: int):
        """预过滤掉没有通讯边或通讯边太少的spot"""
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
    
    def __len__(self):
        # 返回有效spot数量，如果没有过滤则返回全部spot数量
        if hasattr(self, 'valid_spot_indices'):
            return len(self.valid_spot_indices)
        else:
            return self.n_spots
    
    def __getitem__(self, idx):
        """获取单个子图样本"""
        # ========== Step 1: 获取邻域spot（从有效spot列表中映射）==========
        # idx是有效spot列表中的索引，需要映射到原始spot索引
        if hasattr(self, 'valid_spot_indices'):
            center_idx = self.valid_spot_indices[idx]
        else:
            center_idx = idx
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
        
        # ========== Step 3: 只为实际存在的cell创建节点和获取表达量 ==========
        cell_expr_raw_list = []
        n_cell_types = len(self.cell_marker_expr)  # 所有cell类型数
        
        # 预先获取所有需要的基因位置
        marker_gene_positions = []
        for gene_name in self.cell_marker_expr.columns:
            pos = self.marker_gene_to_idx.get(gene_name, -1)
            marker_gene_positions.append(pos)
        
        # ✅ 新增：记录每个spot实际存在的cell及其节点映射
        # spot_cell_mapping: {(spot_local_idx, cell_type_id): cell_node_local_idx}
        spot_cell_mapping = {}
        cell_node_counter = 0  # 动态分配 cell 节点编号
        
        # 遍历子图中的每个 spot
        composition_subgraph_full = self.composition.iloc[subgraph_spot_indices].values.astype(np.float32)
        
        for spot_local_idx, spot_global_idx in enumerate(subgraph_spot_indices):
            # 找出该 spot 实际存在的 cell 类型 (composition > 1e-6)
            composition_spot = composition_subgraph_full[spot_local_idx]
            cells_in_spot = np.where(composition_spot > 1e-6)[0]
            
            for cell_type_id in cells_in_spot:
                # 记录这个 cell 节点的映射
                spot_cell_mapping[(spot_local_idx, cell_type_id)] = cell_node_counter
                cell_node_counter += 1
                
                # 获取该 spot-cell 的表达向量
                idx_in_array = self.spot_cell_expr_index.get((spot_global_idx, cell_type_id))
                
                if idx_in_array is not None:
                    # 从 spot-cell 表达矩阵中提取完整表达
                    spot_cell_full_expr = self.spot_cell_expr[idx_in_array, :]
                    
                    # 只取 marker 基因
                    cell_marker_expr = np.zeros(len(marker_gene_positions), dtype=np.float32)
                    for gene_idx, gene_pos in enumerate(marker_gene_positions):
                        if gene_pos >= 0:
                            cell_marker_expr[gene_idx] = spot_cell_full_expr[gene_pos]
                else:
                    # 如果找不到，用全局 cell marker 表达作为备选
                    cell_marker_expr = self.cell_marker_expr.iloc[cell_type_id].values.astype(np.float32)
                
                cell_expr_raw_list.append(cell_marker_expr)
        
        cell_expr_raw = torch.tensor(
            np.array(cell_expr_raw_list), dtype=torch.float32
        ) if cell_expr_raw_list else torch.empty((0, len(marker_gene_positions)), dtype=torch.float32)
        # [actual_n_cells, n_marker_genes] - 只包含实际存在的 cell 节点
        
        n_actual_cells = len(cell_expr_raw_list)  # 实际创建的 cell 节点数
        
        # ========== Step 4: 获取composition矩阵和坐标 ==========
        coords_sub = self.coords[subgraph_spot_indices]
        # composition_subgraph_full 已在 Step 3 中计算
        # [k+1, n_cell_types]
        
        # ========== Step 5: 构建相似度边（Spot-Spot + Spot-Cell） ==========
        # ✅ 新的节点编号方案：
        # Spot节点: 0 ~ n_spots_sub-1
        # Cell节点: n_spots_sub ~ n_spots_sub + n_actual_cells - 1 (动态分配)
        #   每个实际存在的 cell 按顺序分配节点编号
        
        edge_index_like_list = []
        edge_attr_like_list = []
        
        # 5a. Spot-Spot边（基于KNN关系 + 高斯权重）
        # ✅ 只为KNN邻居创建边，避免对所有spot对计算距离
        for i in range(n_spots_sub):
            spot_i_global = subgraph_spot_indices[i]
            for j in range(n_spots_sub):
                if i != j:
                    spot_j_global = subgraph_spot_indices[j]
                    
                    # 检查是否为KNN邻居（使用预计算的knn_mask）
                    if self.knn_mask[spot_i_global, spot_j_global] == 1:
                        # 计算高斯权重
                        dist = np.linalg.norm(coords_sub[i] - coords_sub[j])
                        sigma = 50.0
                        weight = np.exp(-dist**2 / (2 * sigma**2))
                        if weight > 1e-4:
                            edge_index_like_list.append([i, j])
                            edge_attr_like_list.append(weight)
        
        # 5b. Spot-Cell边（基于composition权重）
        # ✅ 只为实际存在的 cell 创建边
        for (spot_local_idx, cell_type_id), cell_node_local_idx in spot_cell_mapping.items():
            comp = composition_subgraph_full[spot_local_idx, cell_type_id]
            weight = np.sqrt(comp)  # 权重转换
            
            # Cell 节点的全局编号 = n_spots_sub + cell_node_local_idx
            cell_node_global_id = n_spots_sub + cell_node_local_idx
            edge_index_like_list.append([spot_local_idx, cell_node_global_id])
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
            
            # ✅ 计算中心点cell到邻居点cell的通讯边（使用动态节点映射）
            for cell_type_i in cells_in_center:
                # 查找中心 spot 的这个 cell 类型对应的节点编号
                cell_i_key = (center_local_idx, cell_type_i)
                if cell_i_key not in spot_cell_mapping:
                    continue
                cell_i_node_local_idx = spot_cell_mapping[cell_i_key]
                
                for cell_type_j in cells_in_j:
                    # 查找邻居 spot 的这个 cell 类型对应的节点编号
                    cell_j_key = (j_local, cell_type_j)
                    if cell_j_key not in spot_cell_mapping:
                        continue
                    cell_j_node_local_idx = spot_cell_mapping[cell_j_key]
                    
                    # 优化：直接查询预计算的LR得分，避免遍历所有LR对
                    pair_key = (center_global_idx, spot_j_global, cell_type_i, cell_type_j)
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
                        
                        # ✅ 使用动态分配的节点编号
                        cell_i_node_global_id = n_spots_sub + cell_i_node_local_idx
                        cell_j_node_global_id = n_spots_sub + cell_j_node_local_idx
                        edge_index_cc_list.append([cell_i_node_global_id, cell_j_node_global_id])
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
            'n_cells': n_actual_cells,  # ✅ 实际创建的 cell 节点数（动态）
            'n_cell_types': n_cell_types,  # 所有 cell 类型数（用于 composition 矩阵维度）
            'spot_cell_mapping': spot_cell_mapping,  # 节点映射字典
            'expr_raw': expr_raw,  # [k+1, n_genes] 原始spot表达谱
            'cell_expr_raw': cell_expr_raw,  # [n_actual_cells, n_marker_genes] 实际存在的cell表达谱
            'edge_index_like': edge_index_like,  # 相似度边 (spot-spot + spot-cell)
            'edge_attr_like': edge_attr_like,
            'edge_index_cc': edge_index_cc,  # cell-cell通讯边
            'edge_attr_cc': edge_attr_cc,
            'coords_subgraph': coords_subgraph,
            'composition_subgraph': composition_subgraph,  # [k+1, n_cell_types]
        }


def hetero_subgraph_collate_fn(batch):
    """自定义collate_fn处理异构子图批次"""
    batch_size = len(batch)
    
    # ✅ 每个样本的 spot 数量可能不同（因为邻居数量受距离阈值影响）
    n_spots_sub_list = [sample['n_spots_sub'] for sample in batch]
    
    # ✅ Spot 表达量也改为列表（维度可能不同）
    expr_raw_list = [sample['expr_raw'] for sample in batch]  # list of [n_spots_i, n_marker_genes]
    
    # ✅ Cell 表达量改为列表（因为每个 subgraph 的 cell 数量可能不同）
    cell_expr_raw_list = [sample['cell_expr_raw'] for sample in batch]  # list of [n_cells_i, n_marker_genes]
    
    # 边保持为列表（因为边数可能不同）
    edge_index_like_list = [sample['edge_index_like'] for sample in batch]  # list of [2, E_i]
    edge_attr_like_list = [torch.cat([sample['edge_attr_like'].unsqueeze(-1), torch.zeros_like(sample['edge_attr_like'].unsqueeze(-1))], dim=-1) for sample in batch]    # list of [E_i, 2] - [weight, 0]
    edge_index_cc_list = [sample['edge_index_cc'] for sample in batch]      # list of [2, E_i]
    edge_attr_cc_list = [sample['edge_attr_cc'] for sample in batch]        # list of [E_i, 2] - [lr_score, lr_id+1]
    
    # ✅ 收集每个样本的实际 cell 节点数
    n_cells_list = [sample['n_cells'] for sample in batch]
    
    batch_dict = {
        'batch_size': batch_size,
        'n_spots_sub': n_spots_sub_list,  # ✅ 改为列表，每个样本可能不同
        'n_cells': n_cells_list,  # ✅ 改为列表，每个样本可能不同
        'center_spot_idx': [sample['center_spot_idx'] for sample in batch],
        'spot_cell_mapping': [sample['spot_cell_mapping'] for sample in batch],  # ✅ 添加spot_cell_mapping
        'expr_raw': expr_raw_list,  # ✅ list of [n_spots_i, n_marker_genes]
        'cell_expr_raw': cell_expr_raw_list,  # ✅ list of [n_cells_i, n_marker_genes]
        'edge_index_like': edge_index_like_list,  # list of [2, E_i]
        'edge_attr_like': edge_attr_like_list,    # list of [E_i]
        'edge_index_cc': edge_index_cc_list,      # list of [2, E_i]
        'edge_attr_cc': edge_attr_cc_list,        # list of [E_i]
    }
    
    return batch_dict
