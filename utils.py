import os
import torch
import torch.nn as nn
import numpy as np
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import networkx as nx

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

    # 避免tqdm进度条被日志覆盖
    logging.getLogger('tqdm').addHandler(console_handler)

# def compute_rwpe(adj_matrix, steps=5, dim=32):
#     """
#     RWPE: Random Walk Positional Encoding
#     adj_matrix: numpy (N,N) 邻接矩阵
#     steps: k步随机游走
#     dim: RWPE编码维度
    
#     return: RWPE (N, dim)
#     """
#     N = adj_matrix.shape[0]
    
#     # 归一化邻接矩阵 (transition matrix)
#     deg = adj_matrix.sum(axis=1, keepdims=True)
#     deg[deg == 0] = 1e-6
#     P = adj_matrix / deg  # row-normalized
    
#     # 初始访问分布
#     rw = np.eye(N)
#     rw_features = []

#     # 迭代k步随机游走
#     for k in range(steps):
#         rw = rw @ P    # k-step random walk probability
#         rw_features.append(rw)

#     # 拼接不同步长的访问概率
#     RW_full = np.concatenate(rw_features, axis=-1)  # (N, N*k)
    
#     # 降维 -> RWPE dim (用SVD/PCA降到dim)
#     U, S, V = np.linalg.svd(RW_full)
#     RWPE = U[:, :dim] * S[:dim]  # (N, dim)

#     return torch.FloatTensor(RWPE)  # (N, dim)

# def random_walk_subgraph(adj_matrix, start_node=None, walk_len=50):
#     """
#     从随机起点做随机游走，采样子图节点
#     """
#     N = adj_matrix.shape[0]
#     if start_node is None:
#         start_node = np.random.randint(0, N)
    
#     visited = [start_node]
#     cur = start_node
    
#     for _ in range(walk_len):
#         neighbors = np.where(adj_matrix[cur] > 0)[0]
#         if len(neighbors) == 0:
#             break
#         cur = np.random.choice(neighbors)
#         visited.append(cur)
    
#     sub_nodes = np.unique(visited)
#     sub_adj = adj_matrix[np.ix_(sub_nodes, sub_nodes)]
#     return sub_adj, sub_nodes


# def build_graph_data_for_contrastive_learning(
#         coords,
#         fusion_emb,
#         lr_score_mat,
#         k=6,
#         lambda_dist=0.5,
#         rwpe_steps=5,
#         rwpe_dim=32,
#         drop_edge_rate=0.2,
#         noise_scale=0.1,
#         random_walk_len=50,
#         normalize_lr=True
#     ):
#     """
#     生成 Graphormer 输入：
#     G1 = DropEdge + RWPE (全图增强)
#     G2 = 随机游走采样子图 (局部增强)
#     """
#     N = coords.shape[0]

#     # ======== 1. KNN邻接 + 距离 ========
#     dist_mat = pairwise_distances(coords)
#     np.fill_diagonal(dist_mat, 1e-6)
#     knn_graph = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
#     adj_matrix = knn_graph.toarray().astype(float)
#     np.fill_diagonal(adj_matrix, 0)

#     # ======== 2. LR打分归一化 + Edge Bias ========
#     if normalize_lr:
#         lr_score_mat = (lr_score_mat - lr_score_mat.min()) / (lr_score_mat.max() - lr_score_mat.min() + 1e-8)
#     dist_norm = dist_mat / (dist_mat.std() + 1e-8)
#     edge_bias = np.log1p(lr_score_mat + 1e-6) - lambda_dist * dist_norm
#     edge_bias = edge_bias * (adj_matrix > 0) + (-1e9) * (adj_matrix == 0)

#     edge_bias_torch = torch.FloatTensor(edge_bias)

#     # ======== 3. RWPE编码（仅G1用） ========
#     RWPE = compute_rwpe(adj_matrix, steps=rwpe_steps, dim=rwpe_dim)  # (N, rwpe_dim)
#     fusion_emb_torch = torch.FloatTensor(fusion_emb)
#     node_features_g1 = torch.cat([fusion_emb_torch, RWPE], dim=-1)

#     # ======== 4. G1 = DropEdge增强视图 ========
#     adj_g1 = adj_matrix.copy()
#     drop_edges = np.random.rand(*adj_g1.shape) < drop_edge_rate
#     adj_g1[drop_edges] = 0
#     adj_mask_g1 = torch.tensor(adj_g1 > 0, dtype=torch.bool)
#     feat_g1 = node_features_g1 + noise_scale * torch.randn_like(node_features_g1)
#     G1 = {
#         "node_feat": feat_g1,
#         "edge_bias": edge_bias_torch,  # 保持全局bias
#         "adj_mask": adj_mask_g1
#     }

#     # ======== 5. G2 = 随机游走子图 ========
#     sub_adj, sub_nodes = random_walk_subgraph(adj_matrix, walk_len=random_walk_len)
#     sub_feat = fusion_emb_torch[sub_nodes] + noise_scale * torch.randn_like(fusion_emb_torch[sub_nodes])
#     sub_edge_bias = edge_bias[np.ix_(sub_nodes, sub_nodes)]
#     sub_edge_bias_torch = torch.FloatTensor(sub_edge_bias)
#     sub_adj_mask = torch.tensor(sub_adj > 0, dtype=torch.bool)
#     G2 = {
#         "node_feat": sub_feat,        # 不拼RWPE，直接原始特征
#         "edge_bias": sub_edge_bias_torch,
#         "adj_mask": sub_adj_mask
#     }

#     # ======== 6. G_origin 保留全图原始视图 ========
#     adj_mask_torch = torch.tensor(adj_matrix > 0, dtype=torch.bool)
#     G_origin = {
#         "node_feat": node_features_g1,  # 原始拼RWPE
#         "edge_bias": edge_bias_torch,
#         "adj_mask": adj_mask_torch
#     }

#     return G1, G2, G_origin, adj_matrix
