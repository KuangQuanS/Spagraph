"""K-nearest neighbors utilities for dynamic cluster representation"""

import numpy as np
import torch
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity


def precompute_knn_cells(
    spot_embeddings: np.ndarray,
    sc_cell_embeddings: np.ndarray,
    sc_cell_labels: np.ndarray,
    k_cells_per_cluster: int = 10
) -> np.ndarray:
    """预计算每个spot在每个cluster中的k个最近细胞
    
    这个函数应该在第一阶段（VAE训练）结束后调用，基于VAE学到的embeddings
    预计算好k-nearest cells的索引，供第二阶段（GAT训练）使用。
    
    核心逻辑：
    1. 对每个cluster，找到属于该cluster的所有细胞
    2. 对每个spot，计算与该cluster所有细胞的余弦相似度
    3. 选择top-k个最相似的细胞
    4. 保存这k个细胞的全局索引
    
    Args:
        spot_embeddings: [n_spots, embedding_dim] ST数据的VAE embedding
        sc_cell_embeddings: [n_cells, embedding_dim] SC数据的VAE embedding
        sc_cell_labels: [n_cells] 每个细胞的cluster标签（0, 1, ..., n_clusters-1）
        k_cells_per_cluster: 每个cluster选择多少个最近细胞
    
    Returns:
        knn_cell_indices: [n_spots, n_clusters, k] 每个spot在每个cluster中的k个最近细胞的全局索引
        
    Example:
        >>> # 在第一阶段结束后
        >>> knn_indices = precompute_knn_cells(
        ...     spot_embeddings=st_data_emb,
        ...     sc_cell_embeddings=sc_data_emb,
        ...     sc_cell_labels=sc_labels,
        ...     k_cells_per_cluster=10
        ... )
        >>> # 保存到artifacts
        >>> stage1_artifacts.knn_cell_indices = knn_indices
    """
    n_spots = spot_embeddings.shape[0]
    n_clusters = int(sc_cell_labels.max()) + 1
    k = k_cells_per_cluster
    
    # 初始化输出：[n_spots, n_clusters, k]
    # 用-1填充（表示无效索引，用于处理cluster细胞数<k的情况）
    knn_cell_indices = np.full((n_spots, n_clusters, k), -1, dtype=np.int64)
    
    print(f"Precomputing k-nearest cells for {n_spots} spots and {n_clusters} clusters...")
    
    # 对每个cluster单独处理
    for cluster_id in range(n_clusters):
        # 找到属于该cluster的所有细胞
        cluster_mask = (sc_cell_labels == cluster_id)
        cluster_cell_indices = np.where(cluster_mask)[0]  # 全局索引
        n_cluster_cells = len(cluster_cell_indices)
        
        if n_cluster_cells == 0:
            print(f"  Warning: Cluster {cluster_id} has no cells, skipping")
            continue
        
        # 该cluster的细胞embeddings
        cluster_cell_embeddings = sc_cell_embeddings[cluster_cell_indices]  # [n_cluster_cells, embedding_dim]
        
        # 计算每个spot与该cluster所有细胞的余弦相似度
        # similarity: [n_spots, n_cluster_cells]
        similarity = cosine_similarity(spot_embeddings, cluster_cell_embeddings)
        
        # 对每个spot，选择top-k个最相似的细胞
        actual_k = min(k, n_cluster_cells)  # 如果该cluster细胞数 < k，只选actual_k个
        
        # argsort默认升序，我们需要降序（最大的在前），所以取[-actual_k:]
        # topk_local_indices: [n_spots, actual_k] 在该cluster内的局部索引
        topk_local_indices = np.argpartition(similarity, -actual_k, axis=1)[:, -actual_k:]
        
        # 按相似度排序（可选，但更规范）
        for spot_idx in range(n_spots):
            local_indices = topk_local_indices[spot_idx]
            similarities = similarity[spot_idx, local_indices]
            sorted_order = np.argsort(similarities)[::-1]  # 降序
            topk_local_indices[spot_idx] = local_indices[sorted_order]
        
        # 转换为全局索引
        topk_global_indices = cluster_cell_indices[topk_local_indices]  # [n_spots, actual_k]
        
        # 存储结果
        knn_cell_indices[:, cluster_id, :actual_k] = topk_global_indices
        
        if n_cluster_cells < k:
            print(f"  Cluster {cluster_id}: only {n_cluster_cells} cells (< k={k}), padding with -1")
    
    print(f"Done! Precomputed k-nearest cells shape: {knn_cell_indices.shape}")
    return knn_cell_indices


def precompute_knn_cells_torch(
    spot_embeddings: torch.Tensor,
    sc_cell_embeddings: torch.Tensor,
    sc_cell_labels: torch.Tensor,
    k_cells_per_cluster: int = 10,
    batch_size: int = 500
) -> torch.Tensor:
    """PyTorch版本的预计算k-nearest cells（向量化优化，支持分批处理防止OOM）
    
    优化策略：
    1. 分批处理spots，避免大矩阵占满GPU内存
    2. 每个cluster的相似度计算向量化
    3. 使用torch.topk一次性获取top-k
    
    Args:
        spot_embeddings: [n_spots, embedding_dim] torch.Tensor（CPU或GPU）
        sc_cell_embeddings: [n_cells, embedding_dim] torch.Tensor（GPU）
        sc_cell_labels: [n_cells] torch.Tensor (long, GPU)
        k_cells_per_cluster: 每个cluster选择多少个最近细胞
        batch_size: 每批处理多少个spots（防止OOM）
    
    Returns:
        knn_cell_indices: [n_spots, n_clusters, k] torch.Tensor (long, CPU)
    """
    device = sc_cell_embeddings.device  # 使用sc数据的设备
    n_spots = spot_embeddings.shape[0]
    n_clusters = int(sc_cell_labels.max().item()) + 1
    k = k_cells_per_cluster
    
    # 将spot embeddings移到GPU（如果不在的话）
    if spot_embeddings.device != device:
        spot_embeddings = spot_embeddings.to(device)
    
    # 初始化输出（在CPU上，避免GPU OOM）
    knn_cell_indices = torch.full((n_spots, n_clusters, k), -1, dtype=torch.long)
    
    # 归一化embeddings（用于余弦相似度）
    spot_embeddings_norm = torch.nn.functional.normalize(spot_embeddings, p=2, dim=1)
    sc_cell_embeddings_norm = torch.nn.functional.normalize(sc_cell_embeddings, p=2, dim=1)
    
    # 对每个cluster单独处理
    for cluster_id in range(n_clusters):
        cluster_mask = (sc_cell_labels == cluster_id)
        cluster_cell_indices = torch.where(cluster_mask)[0]
        n_cluster_cells = len(cluster_cell_indices)
        
        if n_cluster_cells == 0:
            continue
        
        # 该cluster的细胞embeddings（保持在GPU上）
        cluster_cell_embeddings_norm = sc_cell_embeddings_norm[cluster_cell_indices]
        
        actual_k = min(k, n_cluster_cells)
        
        # ✅ 分批处理spots，避免大矩阵OOM
        for batch_start in range(0, n_spots, batch_size):
            batch_end = min(batch_start + batch_size, n_spots)
            batch_spots_norm = spot_embeddings_norm[batch_start:batch_end]  # [batch_size, embedding_dim]
            
            # 余弦相似度 = 归一化向量的点积
            similarity = torch.matmul(batch_spots_norm, cluster_cell_embeddings_norm.T)  # [batch_size, n_cluster_cells]
            
            # 选择top-k
            topk_similarities, topk_local_indices = torch.topk(similarity, actual_k, dim=1)  # [batch_size, actual_k]
            
            # 转换为全局索引
            topk_global_indices = cluster_cell_indices[topk_local_indices]  # [batch_size, actual_k]
            
            # 存储结果（移到CPU）
            knn_cell_indices[batch_start:batch_end, cluster_id, :actual_k] = topk_global_indices.cpu()
    
    return knn_cell_indices


def compute_cell_weights_mlp(
    spot_embeddings: torch.Tensor,
    cell_embeddings: torch.Tensor,
    mlp: torch.nn.Module
) -> torch.Tensor:
    """使用MLP计算细胞权重（用于Loss中的动态表达计算）
    
    Args:
        spot_embeddings: [n_spots, embedding_dim] 当前batch的spot embeddings
        cell_embeddings: [n_spots, k, embedding_dim] 每个spot的k个最近细胞的embeddings
        mlp: MLP模型，输入 [spot_emb || cell_emb]，输出标量权重
    
    Returns:
        cell_weights: [n_spots, k] softmax归一化后的权重（百分比，和为1）
    """
    n_spots, k, embedding_dim = cell_embeddings.shape
    
    # 扩展spot_embeddings: [n_spots, k, embedding_dim]
    spot_embeddings_expanded = spot_embeddings.unsqueeze(1).expand(-1, k, -1)
    
    # 拼接: [n_spots, k, 2*embedding_dim]
    combined = torch.cat([spot_embeddings_expanded, cell_embeddings], dim=-1)
    
    # 重塑为2D: [n_spots*k, 2*embedding_dim]
    combined_flat = combined.view(-1, 2 * embedding_dim)
    
    # MLP计算: [n_spots*k, 1]
    weights_flat = mlp(combined_flat)
    
    # 重塑回: [n_spots, k]
    weights = weights_flat.view(n_spots, k)
    
    # Softmax归一化（每个spot的k个细胞权重和为1）
    cell_weights = torch.nn.functional.softmax(weights, dim=1)
    
    return cell_weights
