import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import logging


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
            # ✅ 删除弱边：基于 edge_attr 的值（假设第一列是边权重/得分）
            if edge_attr is not None and edge_attr.size > 0:
                # 提取边权重
                if edge_attr.ndim == 1:
                    edge_weights = edge_attr
                else:
                    edge_weights = edge_attr[:, 0]  # 取第一列作为权重
                
                # 计算阈值：删除最弱的 rate% 的边
                threshold = np.percentile(edge_weights, rate * 100)
                mask = edge_weights >= threshold
            else:
                # 如果没有边属性，回退到随机删除
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
        # 相似度边：随机删除（这些边基于空间/表达相似度）
        ei_like_aug, ea_like_aug = self.drop_edges(edge_index_like, edge_attr_like, 0.1, strategy='random')
        
        # ✅ 通讯边：删除弱边（基于 LR 通讯得分）
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
        self.logger = logging.getLogger(__name__)
    
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
        
        self.logger.info(f"Spot-Spot edges: {edge_index.shape[1]}")
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
                
                # 权重转换：sqrt, log1p, 或直接使用
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
        
        self.logger.info(f"Spot-Celltype edges: {edge_index.shape[1]}")
        return edge_index, edge_attr
    
    def _compute_lr_score(self, celltype_expr: pd.DataFrame, 
                         celltype_i: int, celltype_j: int,
                         ligand: str, receptor: str) -> float:
        """
        计算从celltype_i到celltype_j的LR通讯得分
        
        支持联合受体（用下划线分隔的多个基因），例如：
        - 单受体: 'EGFR'
        - 联合受体: 'TGFbR1_TGFbR2' (两个基因都需要表达)
        
        Args:
            celltype_expr: [n_celltypes, n_genes] 表达矩阵
            celltype_i: 配体来源celltype索引
            celltype_j: 受体来源celltype索引
            ligand: 配体基因名
            receptor: 受体基因名（可能包含下划线分隔的多个基因）
        
        Returns:
            score: 配体和所有受体表达量的乘积
        """
        ligand_upper = ligand.upper()
        
        # 查找配体基因表达（在celltype_i中）
        lig_expr = None
        for col in celltype_expr.columns:
            if col.upper() == ligand_upper:
                lig_expr = celltype_expr.iloc[celltype_i][col]
                break
        
        if lig_expr is None or lig_expr < 1e-6:
            return 0.0
        
        # 处理受体（可能是联合受体，用下划线分隔）
        receptor_genes = [r.strip() for r in receptor.split('_')]
        receptor_product = 1.0
        
        for receptor_gene in receptor_genes:
            receptor_upper = receptor_gene.upper()
            rec_expr = None
            
            # 在celltype_j中查找受体基因
            for col in celltype_expr.columns:
                if col.upper() == receptor_upper:
                    rec_expr = celltype_expr.iloc[celltype_j][col]
                    break
            
            # 如果联合受体中任何一个基因找不到或不表达，返回0
            if rec_expr is None or rec_expr < 1e-6:
                return 0.0
            
            receptor_product *= rec_expr
        
        # LR通讯强度：配体表达 × 受体乘积
        score = float(lig_expr) * receptor_product
        return score
    
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
        
        self.logger.info(f"Celltype-Celltype edges: {edge_index.shape[1] if edge_index.size > 0 else 0}")
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
        
        self.logger.info(f"Celltype neighbor mask built: {np.sum(celltype_neighbor_mask) / 2:.0f} neighbor pairs")
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
        n_spots = coords.shape[0] if coords is not None else 100
        
        # 如果没有坐标，生成随机坐标
        if coords is None:
            coords = np.random.randn(n_spots, 2) * 100
            logging.warning(f"⚠️  未提供坐标，生成随机坐标: {coords.shape}")
        
        # 如果没有composition，生成随机
        if composition is None:
            composition = pd.DataFrame(
                np.random.dirichlet(np.ones(n_celltypes), n_spots),
                columns=celltype_expr.index
            )
            logging.warning(f"⚠️  未提供composition，生成随机composition: {composition.shape}")
        
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
        logging.warning("PyTorch Geometric not installed, returning dict format instead")
        return graph_data
