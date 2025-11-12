import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class HeteroGATLayer(nn.Module):
    """异构图注意力层"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(out_dim, out_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.scaling_factor = math.sqrt(self.head_dim)
    
    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: [n_nodes, in_dim]
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        
        Returns:
            out: [n_nodes, out_dim]
        """
        n_nodes = node_feat.size(0)
        
        # 线性变换
        Q = self.query(node_feat)
        K = self.key(node_feat)
        V = self.value(node_feat)
        
        # Reshape为多头
        Q = Q.view(n_nodes, self.num_heads, self.head_dim)
        K = K.view(n_nodes, self.num_heads, self.head_dim)
        V = V.view(n_nodes, self.num_heads, self.head_dim)
        
        # 提取边
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        n_edges = len(src_idx)
        
        # 获取边对应的特征
        Q_dst = Q[dst_idx]
        K_src = K[src_idx]
        V_src = V[src_idx]
        
        # 计算注意力权重
        attn_logits = (Q_dst * K_src).sum(dim=-1) / self.scaling_factor
        attn_weights = torch.sigmoid(attn_logits)
        
        # 聚合
        out = torch.zeros(n_nodes, self.num_heads, self.head_dim,
                         device=node_feat.device, dtype=node_feat.dtype)
        
        for i in range(n_edges):
            dst = dst_idx[i]
            weight = attn_weights[i].unsqueeze(-1)
            out[dst] += weight * V_src[i]
        
        # 重塑
        out = out.view(n_nodes, self.out_dim)
        
        # 输出投影
        out = self.output_projection(out)
        out = self.output_dropout(out)
        
        return out


class HeteroGATEncoder(nn.Module):
    """多层异构图注意力编码器"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 128],
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        dims = [input_dim] + hidden_dims
        
        self.layers = nn.ModuleList([
            HeteroGATLayer(dims[i], dims[i+1], num_heads, dropout)
            for i in range(len(dims) - 1)
        ])
        
        self.activation = nn.ReLU()
        self.output_dim = hidden_dims[-1]
    
    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: [n_nodes, input_dim]
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        
        Returns:
            out: [n_nodes, hidden_dims[-1]]
        """
        x = node_feat
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x


class Readout(nn.Module):
    """图级别的readout操作"""
    
    def __init__(self, hidden_dim: int, readout_mode: str = 'mean'):
        super().__init__()
        self.readout_mode = readout_mode
        
        if readout_mode == 'gated':
            # 门控readout
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [n_nodes, hidden_dim]
        
        Returns:
            summary: [hidden_dim] - 图级别的summary向量
        """
        if self.readout_mode == 'mean':
            return node_embeddings.mean(dim=0)
        elif self.readout_mode == 'sum':
            return node_embeddings.sum(dim=0)
        elif self.readout_mode == 'gated':
            gate = self.gate(node_embeddings)
            proj = self.proj(node_embeddings)
            gated = gate * proj
            return gated.mean(dim=0)
        else:
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")


class Discriminator(nn.Module):
    """DGI判别器"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 双线性判别器
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
    
    def forward(self, node_embedding: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embedding: [hidden_dim] - 单个节点的嵌入
            summary: [hidden_dim] - 图级别的summary向量
        
        Returns:
            score: 标量 - 判别得分
        """
        # 双线性形式: h^T W s
        score = torch.matmul(node_embedding, torch.matmul(self.weight, summary))
        return score


class DGIPretrainModel(nn.Module):
    """DGI风格的自监督预训练模型"""
    
    def __init__(self, vae_encoder, vae_latent_dim: int = 64,
                 gat_hidden_dims: list = [256, 256, 128],
                 gat_heads: int = 4, gat_dropout: float = 0.1,
                 readout_mode: str = 'mean',
                 corruption_mode: str = 'feature_mask',
                 mask_ratio: float = 0.3,
                 noise_std: float = 0.1):
        """
        Args:
            vae_encoder: 预训练的VAE编码器
            vae_latent_dim: VAE潜在空间维度
            gat_hidden_dims: GAT隐藏层维度列表
            gat_heads: 注意力头数
            gat_dropout: Dropout概率
            readout_mode: Readout模式 ('mean', 'sum', 'gated')
            corruption_mode: 腐蚀模式 ('feature_mask', 'gaussian_noise', 'shuffle')
            mask_ratio: 特征遮掩比例（用于feature_mask模式）
            noise_std: 高斯噪声标准差（用于gaussian_noise模式）
        """
        super().__init__()
        
        self.vae_encoder = vae_encoder
        self.vae_latent_dim = vae_latent_dim
        self.corruption_mode = corruption_mode
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        
        # GAT编码器（共享）
        self.encoder = HeteroGATEncoder(vae_latent_dim, gat_hidden_dims, gat_heads, gat_dropout)
        
        # Readout
        self.readout = Readout(gat_hidden_dims[-1], readout_mode)
        
        # 判别器
        self.discriminator = Discriminator(gat_hidden_dims[-1])
    
    def corrupt_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        腐蚀节点特征
        
        Args:
            features: [n_nodes, feature_dim]
        
        Returns:
            corrupted_features: [n_nodes, feature_dim]
        """
        corrupted = features.clone()
        
        if self.corruption_mode == 'feature_mask':
            # 随机遮掩部分特征维度
            mask = torch.rand(features.shape, device=features.device) > self.mask_ratio
            corrupted = corrupted * mask.float()
        
        elif self.corruption_mode == 'gaussian_noise':
            # 添加高斯噪声
            noise = torch.randn_like(features) * self.noise_std
            corrupted = corrupted + noise
        
        elif self.corruption_mode == 'shuffle':
            # 打乱节点间的特征（行置换）
            perm = torch.randperm(features.size(0), device=features.device)
            corrupted = corrupted[perm]
        
        else:
            raise ValueError(f"Unknown corruption mode: {self.corruption_mode}")
        
        return corrupted
    
    def forward(self, expr_raw: torch.Tensor, cell_expr_raw: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            expr_raw: [k+1, n_genes] - Spot基因表达
            cell_expr_raw: [(k+1)*n_cells, n_genes] - Cell基因表达
            edge_index: [2, n_edges] - 图的边（包括空间相似度边和通讯边）
            edge_attr: [n_edges] - 边属性
        
        Returns:
            pos_scores: [n_nodes] - 正样本判别得分
            neg_scores: [n_nodes] - 负样本判别得分
            summary: [hidden_dim] - 图summary向量
        """
        # ========== 编码特征 ==========
        # VAE编码
        mu_spot, _ = self.vae_encoder(expr_raw)
        mu_cell, _ = self.vae_encoder(cell_expr_raw)
        
        # 拼接所有节点特征
        all_features = torch.cat([mu_spot, mu_cell], dim=0)  # [n_nodes, vae_latent_dim]
        n_nodes = all_features.size(0)
        
        # ========== 原始图编码 ==========
        node_embeddings = self.encoder(all_features, edge_index, edge_attr)  # [n_nodes, hidden_dim]
        
        # Readout得到summary向量
        summary = self.readout(node_embeddings)  # [hidden_dim]
        
        # ========== 腐蚀特征 ==========
        corrupted_features = self.corrupt_features(all_features)
        
        # ========== 腐蚀图编码 ==========
        corrupted_embeddings = self.encoder(corrupted_features, edge_index, edge_attr)  # [n_nodes, hidden_dim]
        
        # ========== 判别 ==========
        # 正样本：原始嵌入 + summary
        pos_scores = torch.zeros(n_nodes, device=node_embeddings.device)
        for i in range(n_nodes):
            pos_scores[i] = self.discriminator(node_embeddings[i], summary)
        
        # 负样本：腐蚀嵌入 + summary
        neg_scores = torch.zeros(n_nodes, device=node_embeddings.device)
        for i in range(n_nodes):
            neg_scores[i] = self.discriminator(corrupted_embeddings[i], summary)
        
        return pos_scores, neg_scores, summary
    
    def get_embeddings(self, expr_raw: torch.Tensor, cell_expr_raw: torch.Tensor,
                      edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        获取节点嵌入（不进行判别）
        
        Args:
            expr_raw: [k+1, n_genes]
            cell_expr_raw: [(k+1)*n_cells, n_genes]
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
        
        Returns:
            embeddings: [n_nodes, hidden_dim]
        """
        # VAE编码
        mu_spot, _ = self.vae_encoder(expr_raw)
        mu_cell, _ = self.vae_encoder(cell_expr_raw)
        
        # 拼接所有节点特征
        all_features = torch.cat([mu_spot, mu_cell], dim=0)
        
        # 编码
        embeddings = self.encoder(all_features, edge_index, edge_attr)
        
        return embeddings


def dgi_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    DGI损失函数
    
    Args:
        pos_scores: [n_nodes] - 正样本判别得分
        neg_scores: [n_nodes] - 负样本判别得分
    
    Returns:
        loss: 标量
    """
    # BCE损失
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
    
    loss = pos_loss + neg_loss
    
    return loss
