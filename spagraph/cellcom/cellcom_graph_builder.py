import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
from pathlib import Path
import random

class GraphAugmentor:
    """图增强器 - 用于构建腐败图"""
    
    def __init__(self, drop_edge_rate: float = 0.1, mask_node_rate: float = 0.1):
        """
        Args:
            drop_edge_rate: 边删除率
            mask_node_rate: 节点掩蔽率
        """
        self.drop_edge_rate = drop_edge_rate
        self.mask_node_rate = mask_node_rate
    
    def drop_edges(self, edge_index: np.ndarray, edge_attr: np.ndarray,
                   rate: float = None, strategy: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        删除边
        
        Args:
            edge_index: [2, n_edges]
            edge_attr: [n_edges] or [n_edges, d]
            rate: 删除率
            strategy: 'random' 随机删除, 'weak' 删除弱边（基于 edge_attr 的第一列）
        """
        if rate is None:
            rate = self.drop_edge_rate
        
        if edge_index.size == 0:
            return edge_index, edge_attr
        
        n_edges = edge_index.shape[1]
        
        if strategy == 'random':
            # 随机删除
            mask = np.random.binomial(1, 1 - rate, n_edges).astype(bool)
        elif strategy == 'weak':
            # 删除弱边：基于edge_attr的第一列
            if edge_attr is not None and edge_attr.size > 0:
                edge_weights = edge_attr if edge_attr.ndim == 1 else edge_attr[:, 0]
                threshold = np.percentile(edge_weights, rate * 100)
                mask = edge_weights >= threshold
            else:
                mask = np.random.binomial(1, 1 - rate, n_edges).astype(bool)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        edge_index_aug = edge_index[:, mask]
        edge_attr_aug = edge_attr[mask] if edge_attr is not None else None
        
        return edge_index_aug, edge_attr_aug
    
    def augment_graph(self, edge_index_like: np.ndarray, edge_attr_like: np.ndarray,
                     edge_index_cc: np.ndarray, edge_attr_cc: np.ndarray) -> Dict:
        """
        增强异构图（删除边）
        
        Args:
            edge_index_like: [2, E_like] 合并的spot-spot和spot-cell边
            edge_attr_like: [E_like] 对应的边属性
            edge_index_cc: [2, E_cc] cell-cell边
            edge_attr_cc: [E_cc, 2] 对应的边属性 [lr_score, lr_id]
        
        Returns:
            augmented_graph: 增强后的图数据
        """
        # 相似度边：随机删除
        ei_like_aug, ea_like_aug = self.drop_edges(edge_index_like, edge_attr_like, 0.1, strategy='random')
        
        # 通讯边：删除弱边
        ei_cc_aug, ea_cc_aug = self.drop_edges(edge_index_cc, edge_attr_cc, 0.2, strategy='weak')
        
        return {
            'edge_index_like': ei_like_aug,
            'edge_attr_like': ea_like_aug,
            'edge_index_cc': ei_cc_aug,
            'edge_attr_cc': ea_cc_aug,
        }


class HeteroGraphBuilder:
    """构建异构图的三种边类型"""
    
    def __init__(self, n_spot_neighbors: int = 10, spot_distance_sigma: float = 50.0,
                 composition_weight_mode: str = 'sqrt'):
        self.n_spot_neighbors = n_spot_neighbors
        self.spot_distance_sigma = spot_distance_sigma
        self.composition_weight_mode = composition_weight_mode
    
    def build_spot_spot_edges(self, coords: np.ndarray, 
                             edge_weight_mode: str = 'gaussian') -> Tuple[np.ndarray, np.ndarray]:
        """
        构建Spot-Spot边
        
        Args:
            coords: [n_spots, 2] 空间坐标
            edge_weight_mode: 'gaussian' 或 'knn'
        
        Returns:
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        """
        n_spots = coords.shape[0]
        
        # 计算距离矩阵
        distances = squareform(pdist(coords, metric='euclidean'))
        
        # kNN构建边
        knn = NearestNeighbors(n_neighbors=min(self.n_spot_neighbors + 1, n_spots), 
                               algorithm='ball_tree').fit(coords)
        _, indices = knn.kneighbors(coords)
        
        edges_list = []
        weights_list = []
        
        for i in range(n_spots):
            for j in indices[i]:
                if i != j:  # 排除自环
                    dist = distances[i, j]
                    # 高斯加权：exp(-d^2 / (2*sigma^2))
                    weight = np.exp(-dist**2 / (2 * self.spot_distance_sigma**2))
                    edges_list.append([i, j])
                    weights_list.append(weight)
        
        edge_index = np.array(edges_list).T  # [2, n_edges]
        edge_attr = np.array(weights_list)  # [n_edges]
        
        return edge_index, edge_attr
    
    def build_spot_celltype_edges(self, spot_celltype_composition: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建Spot-Celltype边
        
        Args:
            spot_celltype_composition: [n_spots, n_celltypes] 成分矩阵
        
        Returns:
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        """
        n_spots = spot_celltype_composition.shape[0]
        n_celltypes = spot_celltype_composition.shape[1]
        
        edges_list = []
        weights_list = []
        
        for spot_id in range(n_spots):
            for celltype_id in range(n_celltypes):
                comp = spot_celltype_composition.iloc[spot_id, celltype_id]
                
                # 权重转换
                if self.composition_weight_mode == 'sqrt':
                    weight = np.sqrt(comp)
                elif self.composition_weight_mode == 'log1p':
                    weight = np.log1p(comp)
                else:
                    weight = comp
                
                if weight > 1e-6:  # 忽略非常小的权重
                    edges_list.append([spot_id, n_spots + celltype_id])
                    weights_list.append(weight)
        
        edge_index = np.array(edges_list).T  # [2, n_edges]
        edge_attr = np.array(weights_list)  # [n_edges]
        
        return edge_index, edge_attr
    
    def _compute_lr_score(self, celltype_expr: pd.DataFrame, 
                         celltype_i: int, celltype_j: int,
                         ligand: str, receptor: str) -> float:
        """计算LR通讯得分，支持联合受体（用下划线分隔）"""
        ligand_upper = ligand.upper()
        
        # 查找配体基因表达
        lig_expr = None
        for col in celltype_expr.columns:
            if col.upper() == ligand_upper:
                lig_expr = celltype_expr.iloc[celltype_i][col]
                break
        
        if lig_expr is None or lig_expr < 1e-6:
            return 0.0
        
        # 处理联合受体
        receptor_genes = [r.strip() for r in receptor.split('_')]
        receptor_product = 1.0
        
        for receptor_gene in receptor_genes:
            receptor_upper = receptor_gene.upper()
            rec_expr = None
            
            for col in celltype_expr.columns:
                if col.upper() == receptor_upper:
                    rec_expr = celltype_expr.iloc[celltype_j][col]
                    break
            
            if rec_expr is None or rec_expr < 1e-6:
                return 0.0
            
            receptor_product *= rec_expr
        
        return float(lig_expr) * receptor_product
    
    def build_celltype_celltype_edges(self, celltype_expr: pd.DataFrame,
                                     lr_genes: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建Celltype-Celltype边
        
        Args:
            celltype_expr: [n_celltypes, n_genes] celltype表达矩阵
            lr_genes: [(ligand, receptor), ...] 配体受体对列表
        
        Returns:
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        """
        n_celltypes = celltype_expr.shape[0]
        
        edges_list = []
        weights_list = []
        
        # 为每个celltype对计算LR通讯
        for i in range(n_celltypes):
            for j in range(n_celltypes):
                if i != j:
                    # 计算从celltype i到celltype j的通讯强度
                    total_score = 0.0
                    
                    for ligand, receptor in lr_genes:
                        # i是配体来源，j是受体来源
                        score_ij = self._compute_lr_score(
                            celltype_expr, i, j,
                            ligand, receptor
                        )
                        total_score += score_ij
                    
                    if total_score > 1e-6:
                        edges_list.append([i, j])
                        weights_list.append(total_score)
        
        edge_index = np.array(edges_list).T if edges_list else np.array([[], []]).astype(int)
        edge_attr = np.array(weights_list) if weights_list else np.array([])
        
        return edge_index, edge_attr
    
    def build_lr_score_matrix(self, celltype_expr: pd.DataFrame,
                             lr_genes: List[Tuple[str, str]]) -> np.ndarray:
        """
        构建celltype-celltype的LR得分矩阵
        
        Args:
            celltype_expr: [n_celltypes, n_genes] celltype表达矩阵
            lr_genes: [(ligand, receptor), ...] 配体受体对列表
        
        Returns:
            lr_score_mat: [n_celltypes, n_celltypes] LR得分矩阵
        """
        n_celltypes = celltype_expr.shape[0]
        lr_score_mat = np.zeros((n_celltypes, n_celltypes))
        
        for i in range(n_celltypes):
            for j in range(n_celltypes):
                if i != j:
                    total_score = 0.0
                    for ligand, receptor in lr_genes:
                        score_ij = self._compute_lr_score(
                            celltype_expr, i, j,
                            ligand, receptor
                        )
                        total_score += score_ij
                    lr_score_mat[i, j] = total_score
        
        return lr_score_mat
    
    def build_celltype_neighbor_mask(self, coords: np.ndarray, 
                                    composition: pd.DataFrame,
                                    k_neighbors: int = 10) -> np.ndarray:
        """
        构建celltype近邻掩码矩阵 - 标记哪些celltype对在空间上是近邻的
        
        Args:
            coords: [n_spots, 2] spot坐标
            composition: [n_spots, n_celltypes] celltype成分矩阵
            k_neighbors: 邻近spot数量
        
        Returns:
            celltype_neighbor_mask: [n_celltypes, n_celltypes] 
                                   值为1表示存在近邻关系
        """
        n_celltypes = composition.shape[1]
        celltype_neighbor_mask = np.zeros((n_celltypes, n_celltypes))
        
        # 构建KNN索引
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, coords.shape[0]), 
                               algorithm='ball_tree').fit(coords)
        _, indices = knn.kneighbors(coords)
        
        # 对每个spot和其邻近spot，检查它们包含的celltype
        for spot_id in range(coords.shape[0]):
            neighbor_spots = indices[spot_id, 1:]  # 排除自己
            
            # 获取当前spot的celltype构成
            spot_composition = composition.iloc[spot_id].values
            celltype_ids_in_spot = np.where(spot_composition > 1e-6)[0]
            
            # 获取邻近spot的celltype构成
            for neighbor_id in neighbor_spots:
                neighbor_composition = composition.iloc[neighbor_id].values
                celltype_ids_in_neighbor = np.where(neighbor_composition > 1e-6)[0]
                
                # 标记这些celltype对之间存在近邻关系
                for ct_i in celltype_ids_in_spot:
                    for ct_j in celltype_ids_in_neighbor:
                        if ct_i != ct_j:
                            celltype_neighbor_mask[ct_i, ct_j] = 1
        
        return celltype_neighbor_mask
    
    def build_complete_graph(self, celltype_expr: pd.DataFrame,
                           lr_genes: List[Tuple[str, str]],
                           coords: Optional[np.ndarray] = None,
                           composition: Optional[pd.DataFrame] = None) -> Dict:
        """
        构建完整的异构图
        
        Args:
            celltype_expr: [n_celltypes, n_genes] celltype表达量（用于LR计算）
            lr_genes: 配体受体对列表
            coords: [n_spots, 2] spot坐标（可选）
            composition: [n_spots, n_celltypes] 成分矩阵（可选）
        
        Returns:
            graph_data: 包含所有边信息的字典
        """
        n_celltypes = celltype_expr.shape[0]
        
        if coords is None:
            raise ValueError("必须提供spot坐标")
        
        if composition is None:
            raise ValueError("必须提供celltype成分矩阵")
        
        n_spots = coords.shape[0]
        
        # 构建三种边
        ei_ss, ea_ss = self.build_spot_spot_edges(coords)
        ei_sc, ea_sc = self.build_spot_celltype_edges(composition)
        ei_cc, ea_cc = self.build_celltype_celltype_edges(celltype_expr, lr_genes)
        
        graph_data = {
            'n_spots': n_spots,
            'n_celltypes': n_celltypes,
            'coords': coords,
            'edge_index_ss': ei_ss,
            'edge_attr_ss': ea_ss,
            'edge_index_sc': ei_sc,
            'edge_attr_sc': ea_sc,
            'edge_index_cc': ei_cc,
            'edge_attr_cc': ea_cc,
            'celltype_expr': celltype_expr,
            'composition': composition,
        }
        
        return graph_data


def convert_to_torch_geometric(graph_data: Dict):
    """
    转换为PyTorch Geometric格式
    
    Args:
        graph_data: HeteroGraphBuilder.build_complete_graph() 的输出
    
    Returns:
        PyG HeteroData 对象
    """
    try:
        from torch_geometric.data import HeteroData
        
        data = HeteroData()
        
        # 节点
        data['spot'].num_nodes = graph_data['n_spots']
        data['celltype'].num_nodes = graph_data['n_celltypes']
        
        # Spot-Spot边
        if graph_data['edge_index_ss'].size > 0:
            ei_ss = torch.tensor(graph_data['edge_index_ss'], dtype=torch.long)
            ea_ss = torch.tensor(graph_data['edge_attr_ss'], dtype=torch.float)
            data['spot', 'communicates_with', 'spot'].edge_index = ei_ss
            data['spot', 'communicates_with', 'spot'].edge_attr = ea_ss
        
        # Spot-Celltype边
        if graph_data['edge_index_sc'].size > 0:
            ei_sc = torch.tensor(graph_data['edge_index_sc'], dtype=torch.long)
            ea_sc = torch.tensor(graph_data['edge_attr_sc'], dtype=torch.float)
            data['spot', 'composed_of', 'celltype'].edge_index = ei_sc
            data['spot', 'composed_of', 'celltype'].edge_attr = ea_sc
        
        # Celltype-Celltype边
        if graph_data['edge_index_cc'].size > 0:
            ei_cc = torch.tensor(graph_data['edge_index_cc'], dtype=torch.long)
            ea_cc = torch.tensor(graph_data['edge_attr_cc'], dtype=torch.float)
            data['celltype', 'interacts', 'celltype'].edge_index = ei_cc
            data['celltype', 'interacts', 'celltype'].edge_attr = ea_cc
        
        return data
    
    except ImportError:
        raise ImportError("需要安装torch_geometric: pip install torch-geometric")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class STHeteroSubgraphDataset:
    """空间转录组异构图子图数据集 - 以每个spot为中心构建k邻域子图"""
    
    def __init__(
        self,
        st_h5ad_path: str,
        cluster_expr: pd.DataFrame,
        cell_expr: pd.DataFrame,
        cell_full_expr: pd.DataFrame,
        graph_data: Dict,
        lr_pairs: List[Tuple[str, str]],
        k_neighbors: int = 10,
        expr_threshold: float = 1.0,
        load_lr_scores_csv: Optional[str] = None,
        min_comm_edges: int = 1,
        valid_cell_types: Optional[List[str]] = None,
        device: str = 'cpu',
        spot_cell_expr: Optional[pd.DataFrame] = None,
        adata: Optional[sc.AnnData] = None,
        spot_names: Optional[List[str]] = None,
        st_X: Optional[np.ndarray] = None,
        comm_by_spot_pair: Optional[Dict[Tuple[int, int], List[Tuple[int, int, float, int]]]] = None,
        lr_pair_to_id: Optional[Dict[Tuple[str, str], int]] = None,
        lr_id_to_pair: Optional[Dict[int, Tuple[str, str]]] = None,
    ):
        """
        初始化数据集
        Args:
            st_h5ad_path: ST数据路径
            cluster_expr: Cluster marker基因表达量DataFrame [n_clusters, n_marker_genes]
            cell_expr: Cell marker基因表达量DataFrame [n_cells, n_marker_genes]
            cell_full_expr: Cell全基因表达量DataFrame [n_cells, n_all_genes]
            spot_cell_expr: Spot-cell动态表达DataFrame [n_spot_cells, n_genes]
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
        self.spot_cell_expr_df = spot_cell_expr
        self.graph_data = graph_data
        self.device = device
        self.k_neighbors = k_neighbors
        self.lr_pairs = lr_pairs
        self.expr_threshold = expr_threshold
        
        # LR对编码映射
        self.lr_pair_to_id = {}  # {(ligand, receptor): lr_id}
        self.lr_id_to_pair = {}  # {lr_id: (ligand, receptor)}
        
        # 聚合后的通讯得分（按 spot+cell 对聚合所有 LR 事件）
        # {(spot_i, spot_j, ct_i, ct_j): [total_score, lr_id]}
        self.lr_scores_by_spot_pair = {}
        # 按 spot 对索引的通讯列表：{(spot_i, spot_j): [(ct_i, ct_j, total_score, lr_id), ...]}
        self.comm_by_spot_pair = {}
        
        # 加载ST数据
        if adata is not None:
            print("[Dataset] Using in-memory adata")
            self.adata = adata
        else:
            print(f"[Dataset] Using h5ad fallback: {st_h5ad_path}")
            self.adata = sc.read_h5ad(st_h5ad_path)
        self.adata.var_names_make_unique()
        self.spot_names = list(spot_names) if spot_names is not None else self.adata.obs_names.tolist()
        self.n_spots = len(self.spot_names)
        
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
        self.genes = list(dict.fromkeys(cluster_expr.columns.tolist()))
        # 过滤到 adata 中存在的基因
        self.genes = [g for g in self.genes if g in self.adata.var_names]
        # 对于激活基因，我们需要从adata中提取对应的表达
        if st_X is not None:
            self.st_X = st_X
        elif len(self.genes) != self.adata.n_vars:
            # 如果基因数量不匹配，说明使用的是激活基因子集
            self.st_X = self.adata[:, self.genes].X
        else:
            self.st_X = self.adata.X
            
        if hasattr(self.st_X, 'toarray'):
            self.st_X = self.st_X.toarray()
        if self.st_X.dtype != np.float32:
            self.st_X = self.st_X.astype(np.float32)
        
        # 从knn_mask构建knn_indices（允许变长邻居）
 
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
        

        # self.knn_indices 保持为 list of arrays
        
        # marker基因表达用于embedding计算
        self.cell_marker_expr = cell_expr
        self.cell_names = cell_expr.index.tolist()
        self.gene_names = cell_expr.columns.tolist()
        
        print("[Dataset] Loading communication index source")
        if comm_by_spot_pair is not None and lr_pair_to_id is not None and lr_id_to_pair is not None:
            print("[Dataset] Using in-memory communication cache")
            self.comm_by_spot_pair = comm_by_spot_pair
            self.lr_pair_to_id = lr_pair_to_id
            self.lr_id_to_pair = lr_id_to_pair
            self.lr_scores_by_spot_pair = None
            total_cell_pairs = sum(len(v) for v in self.comm_by_spot_pair.values())
            print(f"[Loaded] Spot pairs with communication: {len(self.comm_by_spot_pair)}")
            print(f"[Loaded] Spot-cell pairs with communication: {total_cell_pairs}")
            print(f"[Loaded] Unique LR pairs: {len(self.lr_pair_to_id)}")
        elif load_lr_scores_csv is not None:
            print(f"[Dataset] Using CSV fallback: {load_lr_scores_csv}")
            self._load_lr_scores_from_csv(load_lr_scores_csv)
        else:
            raise ValueError("必须提供load_lr_scores_csv")
        
        # DGI训练不需要过滤spot，直接使用所有spot

        # ========== 确保spot和cell基因顺序一致 ==========
        # 创建从cluster_expr基因顺序到cell_marker_expr基因顺序的映射
        print("[Dataset] Aligning marker matrices")
        self.spot_gene_order_to_cell = []
        for gene_name in self.cell_marker_expr.columns:
            if gene_name in self.genes:
                self.spot_gene_order_to_cell.append(self.genes.index(gene_name))
            else:
                # 如果cell marker基因不在cluster基因中，用-1表示缺失
                self.spot_gene_order_to_cell.append(-1)
        print(f"[Aligned] Spot/cell gene order: {len(self.spot_gene_order_to_cell)} genes")

        # ========== 预计算加速用的数组 ==========
        self.marker_genes = list(self.cell_marker_expr.columns)
        self.n_marker_genes = len(self.marker_genes)
        self.zero_cell_marker_expr = np.zeros(self.n_marker_genes, dtype=np.float32)

        self.composition_values = self.composition.values.astype(np.float32)
        self.cell_name_to_idx = {name: idx for idx, name in enumerate(self.cell_names)}
        self.cells_in_spot = [
            np.where(self.composition_values[i] > 1e-6)[0]
            for i in range(self.n_spots)
        ]
        
        # ========== 预计算Spot-Spot高斯权重矩阵（避免__getitem__中重复计算）==========
        print("[Dataset] Building spatial weight matrix")
        coords = self.coords
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # [n_spots, n_spots, 2]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # [n_spots, n_spots]
        sigma = 50.0
        self.spot_weight_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)).astype(np.float32)  # [n_spots, n_spots]

        # 预先对齐 spot 表达矩阵到 marker 基因顺序
        self.st_X_marker = np.zeros(
            (self.n_spots, len(self.spot_gene_order_to_cell)),
            dtype=np.float32
        )
        valid_positions = [
            (dst_idx, src_idx)
            for dst_idx, src_idx in enumerate(self.spot_gene_order_to_cell)
            if src_idx >= 0
        ]
        if valid_positions:
            dst_idx, src_idx = zip(*valid_positions)
            self.st_X_marker[:, np.array(dst_idx)] = self.st_X[:, np.array(src_idx)]

        # 预先对齐 spot-cell 动态表达到 marker 基因顺序
        if self.spot_cell_expr_df is None:
            raise ValueError("spot_cell_expr (动态 spot-cell 表达) 不能为空：cell 节点特征必须来自动态表达谱")

        self.spot_cell_row_idx = None
        self.spot_cell_marker_expr_values = None
        marker_expr_df = self.spot_cell_expr_df.reindex(columns=self.marker_genes, fill_value=0.0)
        self.spot_cell_marker_expr_values = marker_expr_df.values.astype(np.float32, copy=False)

        spot_name_to_idx = {name: idx for idx, name in enumerate(self.spot_names)}
        row_idx_matrix = np.full((self.n_spots, len(self.cell_names)), -1, dtype=np.int32)
        for row_idx, spot_cell_name in enumerate(marker_expr_df.index):
            parts = spot_cell_name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            spot_barcode, celltype = parts
            spot_idx = spot_name_to_idx.get(spot_barcode)
            cell_idx = self.cell_name_to_idx.get(celltype)
            if spot_idx is None or cell_idx is None:
                continue
            row_idx_matrix[spot_idx, cell_idx] = row_idx
        self.spot_cell_row_idx = row_idx_matrix
        print("[Dataset] Dataset initialization finished")
    

    def _load_lr_scores_from_csv(self, csv_path: str):
        """从CSV文件加载预先计算的LR通讯得分"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到LR得分CSV文件: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 获取spot名称列表和cell名称列表
        spot_name_to_idx = {name: idx for idx, name in enumerate(self.spot_names)}
        cell_name_to_idx = {name: idx for idx, name in enumerate(self.cell_names)}
        
        # 构建查询字典：{(spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx, ligand, receptor): score}
        lr_id_counter = 1  # 从1开始，避免与相似度边的ID=0冲突
        for row in df.itertuples(index=False):
            spot_i_barcode = row.spot_i
            spot_j_barcode = row.spot_j
            cell_i = row.cell_i  # "barcode_cell"
            cell_j = row.cell_j  # "barcode_cell"
            
            # 从cell_i/cell_j中提取cell名称
            # cell_i格式: "barcode_cell"
            cell_i_name = cell_i.rsplit('_', 1)[-1] if '_' in cell_i else None
            cell_j_name = cell_j.rsplit('_', 1)[-1] if '_' in cell_j else None
            
            if cell_i_name is None or cell_j_name is None:
                continue
            
            try:
                # 转换为索引
                spot_i_idx = spot_name_to_idx[spot_i_barcode]
                spot_j_idx = spot_name_to_idx[spot_j_barcode]
                cell_i_idx = cell_name_to_idx[cell_i_name]
                cell_j_idx = cell_name_to_idx[cell_j_name]
                
                ligand = row.ligand
                receptor = row.receptor
                
                # 为LR对分配唯一ID
                lr_pair = (ligand, receptor)
                if lr_pair not in self.lr_pair_to_id:
                    self.lr_pair_to_id[lr_pair] = lr_id_counter
                    self.lr_id_to_pair[lr_id_counter] = lr_pair
                    lr_id_counter += 1
                
                lr_id = self.lr_pair_to_id[lr_pair]

                # ✅ 聚合 comm_score（同一 spot/cell 对可能对应多个 LR 事件）
                comm_score = float(row.comm_score)
                spot_pair_key = (spot_i_idx, spot_j_idx, cell_i_idx, cell_j_idx)
                pair_data = self.lr_scores_by_spot_pair.get(spot_pair_key)
                if pair_data is None:
                    self.lr_scores_by_spot_pair[spot_pair_key] = [comm_score, lr_id]
                else:
                    pair_data[0] += comm_score
            except KeyError:
                # 如果barcode或cell名称找不到，跳过这条记录
                continue
        
        # 二级索引：按 spot 对分组，生成 edge 时只遍历存在通讯的 cell 对，避免 O(C^2) 穷举
        comm_by_spot_pair = {}
        for (spot_i, spot_j, ct_i, ct_j), (total_score, lr_id) in self.lr_scores_by_spot_pair.items():
            comm_by_spot_pair.setdefault((spot_i, spot_j), []).append((ct_i, ct_j, total_score, lr_id))
        self.comm_by_spot_pair = comm_by_spot_pair

        total_cell_pairs = sum(len(v) for v in self.comm_by_spot_pair.values())
        print(f"[Loaded] Spot pairs with communication: {len(self.comm_by_spot_pair)}")
        print(f"[Loaded] Spot-cell pairs with communication: {total_cell_pairs}")
        print(f"[Loaded] Unique LR pairs: {len(self.lr_pair_to_id)}")

        # 释放中间聚合字典，节省内存（生成边时使用 self.comm_by_spot_pair）
        self.lr_scores_by_spot_pair = None
    
    def __len__(self):
        # 返回所有spot数量（DGI训练不需要过滤）
        return self.n_spots
    
    def __getitem__(self, idx):
        """获取单个子图样本"""
        # ========== Step 1: 获取邻域spot（DGI训练使用所有spot）==========
        center_idx = idx  # 直接使用idx作为center spot索引
        neighbor_indices = self.knn_indices[center_idx]
        subgraph_spot_indices = np.concatenate([[center_idx], neighbor_indices])
        n_spots_sub = len(subgraph_spot_indices)
        
        # ========== Step 2: 获取spot表达量（原始表达谱，只保留marker基因，顺序与cell一致） ==========
        expr_marker = self.st_X_marker[subgraph_spot_indices]  # [k+1, n_marker_genes]
        expr_raw = torch.from_numpy(expr_marker)  # [k+1, n_marker_genes]
        
        # ========== Step 3: 只为实际存在的cell创建节点和获取表达量 ==========
        cell_expr_raw_list = []
        n_cell_types = len(self.cell_names)  # 所有cell类型数
        
        # ✅ 新增：记录每个spot实际存在的cell及其节点映射
        # spot_cell_mapping: {(spot_local_idx, cell_type_id): cell_node_local_idx}
        spot_cell_mapping = {}
        cell_node_counter = 0  # 动态分配 cell 节点编号
        
        # 遍历子图中的每个 spot
        composition_subgraph_full = self.composition_values[subgraph_spot_indices]
        
        for spot_local_idx, spot_global_idx in enumerate(subgraph_spot_indices):
            # 找出该 spot 实际存在的 cell 类型 (composition > 1e-6)
            cells_in_spot = self.cells_in_spot[spot_global_idx]
            
            for cell_type_id in cells_in_spot:
                # 记录这个 cell 节点的映射
                spot_cell_mapping[(spot_local_idx, cell_type_id)] = cell_node_counter
                cell_node_counter += 1
                
                row_idx = self.spot_cell_row_idx[spot_global_idx, cell_type_id]
                cell_marker_expr = (
                    self.spot_cell_marker_expr_values[row_idx]
                    if row_idx >= 0
                    else self.zero_cell_marker_expr
                )
                cell_expr_raw_list.append(cell_marker_expr)
        
        cell_expr_raw = (
            torch.from_numpy(np.array(cell_expr_raw_list, dtype=np.float32))
            if cell_expr_raw_list
            else torch.empty((0, self.n_marker_genes), dtype=torch.float32)
        )
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
        
        # 5a. Spot-Spot边（基于KNN关系 + 高斯权重）- 使用预计算的权重矩阵
        knn_sub = self.knn_mask[np.ix_(subgraph_spot_indices, subgraph_spot_indices)]  # [k+1, k+1]
        weight_sub = self.spot_weight_matrix[np.ix_(subgraph_spot_indices, subgraph_spot_indices)]  # [k+1, k+1]
        
        # 找到有效边：KNN邻居 & 权重 > 阈值 & 非自环
        valid_mask = (knn_sub == 1) & (weight_sub > 1e-4)
        np.fill_diagonal(valid_mask, False)  # 排除自环
        
        # 提取边索引和权重
        src_indices, dst_indices = np.where(valid_mask)
        weights = weight_sub[valid_mask]
        
        # 直接构建numpy数组，避免Python list开销
        if len(src_indices) > 0:
            edge_index_like_list = list(np.stack([src_indices, dst_indices], axis=1))
            edge_attr_like_list = list(weights)
        
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
        # ✅ 修改：计算子图内所有spot的cell之间的通讯（不只是中心spot）
        # 注意：LR通讯得分已经预计算在CSV中，直接从字典查询

        edge_index_cc_list = []
        edge_attr_cc_list = []  # 现在包含两个特征：[lr_score, lr_id]

        # ✅ 计算子图内所有spot的cell之间的通讯（有向边：配体细胞 -> 受体细胞）
        # 仅遍历“预计算里存在通讯”的 cell 对，避免对 cells_in_i × cells_in_j 做穷举
        for i_local, spot_i_global in enumerate(subgraph_spot_indices):
            for j_local, spot_j_global in enumerate(subgraph_spot_indices):
                comm_list = self.comm_by_spot_pair.get((spot_i_global, spot_j_global))
                if not comm_list:
                    continue

                for cell_type_i, cell_type_j, total_lr_score, lr_id in comm_list:
                    cell_i_node_local_idx = spot_cell_mapping.get((i_local, cell_type_i))
                    if cell_i_node_local_idx is None:
                        continue
                    cell_j_node_local_idx = spot_cell_mapping.get((j_local, cell_type_j))
                    if cell_j_node_local_idx is None:
                        continue

                    cell_i_node_global_id = n_spots_sub + cell_i_node_local_idx
                    cell_j_node_global_id = n_spots_sub + cell_j_node_local_idx
                    edge_index_cc_list.append([cell_i_node_global_id, cell_j_node_global_id])
                    edge_attr_cc_list.append([total_lr_score, lr_id])
                        
        edge_index_cc = torch.tensor(
            np.array(edge_index_cc_list).T if edge_index_cc_list else np.array([[], []]).astype(int),
            dtype=torch.long
        )
        edge_attr_cc = torch.tensor(
            edge_attr_cc_list if edge_attr_cc_list else [],
            dtype=torch.float32
        )
        
        # 如果通讯边数量少于阈值，直接丢弃该子图样本
        if edge_index_cc.size(1) < self.min_comm_edges:
            return None
        
        # ========== Step 8: 整理返回数据 ==========
        coords_subgraph = torch.tensor(coords_sub, dtype=torch.float32)
        composition_subgraph = torch.tensor(composition_subgraph_full, dtype=torch.float32)
        
        return {
            'center_spot_idx': center_idx,
            'subgraph_spot_indices': subgraph_spot_indices,  # 保留索引用于内部计算
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
    # 过滤掉没有通讯边的子图（__getitem__ 返回 None）
    batch = [sample for sample in batch if sample is not None]
    batch_size = len(batch)
    if batch_size == 0:
        return None
    
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
    edge_attr_cc_list = [sample['edge_attr_cc'] for sample in batch]        # list of [E_i, 2] - [lr_score, lr_id]
    
    # ✅ 收集每个样本的实际 cell 节点数
    n_cells_list = [sample['n_cells'] for sample in batch]
    
    batch_dict = {
        'batch_size': batch_size,
        'n_spots_sub': n_spots_sub_list,  # ✅ 改为列表，每个样本可能不同
        'n_cells': n_cells_list,  # ✅ 改为列表，每个样本可能不同
        'center_spot_idx': [sample['center_spot_idx'] for sample in batch],
        'spot_indices': [sample['subgraph_spot_indices'] for sample in batch],  # 保留索引用于内部计算
        'spot_cell_mapping': [sample['spot_cell_mapping'] for sample in batch],  # ✅ 添加spot_cell_mapping
        'expr_raw': expr_raw_list,  # ✅ list of [n_spots_i, n_marker_genes]
        'cell_expr_raw': cell_expr_raw_list,  # ✅ list of [n_cells_i, n_marker_genes]
        'edge_index_like': edge_index_like_list,  # list of [2, E_i]
        'edge_attr_like': edge_attr_like_list,    # list of [E_i]
        'edge_index_cc': edge_index_cc_list,      # list of [2, E_i]
        'edge_attr_cc': edge_attr_cc_list,        # list of [E_i]
    }
    
    return batch_dict


def hetero_subgraph_collate_fn_batched(batch):
    """将多个子图做 disjoint-union，真正 batch 化为一个大图（减少 Python 循环开销）"""
    batch = [sample for sample in batch if sample is not None]
    batch_size = len(batch)
    if batch_size == 0:
        return None

    n_spots_sub_list = [sample['n_spots_sub'] for sample in batch]
    n_cells_list = [sample['n_cells'] for sample in batch]
    spot_offsets = np.cumsum([0] + n_spots_sub_list[:-1]).astype(int).tolist()
    cell_offsets = np.cumsum([0] + n_cells_list[:-1]).astype(int).tolist()
    total_spots = int(sum(n_spots_sub_list))
    total_cells = int(sum(n_cells_list))

    expr_raw = torch.cat([sample['expr_raw'] for sample in batch], dim=0)
    if total_cells > 0:
        cell_expr_raw = torch.cat([sample['cell_expr_raw'] for sample in batch], dim=0)
    else:
        cell_expr_raw = torch.empty((0, expr_raw.size(1)), dtype=expr_raw.dtype)

    edge_index_like_parts = []
    edge_attr_like_parts = []
    edge_index_cc_parts = []
    edge_attr_cc_parts = []

    for sample, n_spots_sub, n_cells, spot_off, cell_off in zip(
        batch, n_spots_sub_list, n_cells_list, spot_offsets, cell_offsets
    ):
        # ---- like edges (spot-spot + spot-cell) ----
        ei_like = sample['edge_index_like']
        if ei_like.numel() > 0:
            src = ei_like[0] + spot_off
            dst = ei_like[1]
            dst_is_spot = dst < n_spots_sub
            dst = torch.where(
                dst_is_spot,
                dst + spot_off,
                (dst - n_spots_sub) + total_spots + cell_off
            )
            edge_index_like_parts.append(torch.stack([src, dst], dim=0))

            ea_like = sample['edge_attr_like']
            if ea_like.numel() > 0:
                edge_attr_like_parts.append(torch.stack([ea_like, torch.zeros_like(ea_like)], dim=1))

        # ---- cc edges (cell-cell) ----
        ei_cc = sample['edge_index_cc']
        if ei_cc.numel() > 0:
            src = (ei_cc[0] - n_spots_sub) + total_spots + cell_off
            dst = (ei_cc[1] - n_spots_sub) + total_spots + cell_off
            edge_index_cc_parts.append(torch.stack([src, dst], dim=0))

            ea_cc = sample['edge_attr_cc']
            if ea_cc.dim() == 1:
                ea_cc = ea_cc.view(-1, 2) if ea_cc.numel() > 0 else ea_cc.new_zeros((0, 2))
            if ea_cc.numel() > 0:
                edge_attr_cc_parts.append(ea_cc)

    edge_index_like = (
        torch.cat(edge_index_like_parts, dim=1)
        if edge_index_like_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr_like = (
        torch.cat(edge_attr_like_parts, dim=0)
        if edge_attr_like_parts
        else torch.empty((0, 2), dtype=torch.float32)
    )
    edge_index_cc = (
        torch.cat(edge_index_cc_parts, dim=1)
        if edge_index_cc_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr_cc = (
        torch.cat(edge_attr_cc_parts, dim=0)
        if edge_attr_cc_parts
        else torch.empty((0, 2), dtype=torch.float32)
    )

    return {
        'batch_size': batch_size,
        'n_spots_sub': n_spots_sub_list,
        'n_cells': n_cells_list,
        'expr_raw': expr_raw,
        'cell_expr_raw': cell_expr_raw,
        'edge_index_like': edge_index_like,
        'edge_attr_like': edge_attr_like,
        'edge_index_cc': edge_index_cc,
        'edge_attr_cc': edge_attr_cc,
    }
