import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class SimpleViTEncoder(nn.Module):
    """简化的Vision Transformer编码器"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, 
                     W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, C * self.patch_size * self.patch_size)
        
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # 使用平均池化作为全局特征
        x = x.mean(dim=1)  # [B, hidden_dim]
        
        return x


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
                edge_attr: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            node_feat: [n_nodes, in_dim]
            edge_index: [2, n_edges]
            edge_attr: [n_edges] 或 [n_edges, 1]
            return_attention: 是否返回注意力得分
        
        Returns:
            out: [n_nodes, out_dim]
            attention_scores: [n_edges, num_heads] 如果return_attention=True，否则None
        """
        n_nodes = node_feat.size(0)
        
        # 线性变换
        Q = self.query(node_feat)  # [n_nodes, out_dim]
        K = self.key(node_feat)    # [n_nodes, out_dim]
        V = self.value(node_feat)  # [n_nodes, out_dim]
        
        # Reshape为多头
        Q = Q.view(n_nodes, self.num_heads, self.head_dim)  # [n_nodes, num_heads, head_dim]
        K = K.view(n_nodes, self.num_heads, self.head_dim)
        V = V.view(n_nodes, self.num_heads, self.head_dim)
        
        # 提取边
        src_idx = edge_index[0]  # [n_edges]
        dst_idx = edge_index[1]  # [n_edges]
        n_edges = len(src_idx)
        
        # 获取边对应的特征
        Q_dst = Q[dst_idx]  # [n_edges, num_heads, head_dim]
        K_src = K[src_idx]
        V_src = V[src_idx]
        
        # 计算注意力权重（未归一化）
        attn_logits = (Q_dst * K_src).sum(dim=-1) / self.scaling_factor  # [n_edges, num_heads]
        
        # 注意：不再进行softmax归一化，直接使用原始logits作为注意力得分
        # 这样可以保留原始的重要性评分，而不受概率分布约束
        
        # 聚合（使用未归一化的logits作为权重）
        out = torch.zeros(n_nodes, self.num_heads, self.head_dim,
                         device=node_feat.device, dtype=node_feat.dtype)
        
        for i in range(n_edges):
            dst = dst_idx[i]
            # 使用sigmoid将logits转换为0-1范围的权重，但保持相对重要性
            weight = torch.sigmoid(attn_logits[i]).unsqueeze(-1)  # [num_heads, 1]
            out[dst] += weight * V_src[i]  # [num_heads, head_dim]
        
        # 重塑
        out = out.view(n_nodes, self.out_dim)
        
        # 输出投影
        out = self.output_projection(out)
        out = self.output_dropout(out)
        
        # 返回注意力得分（如果需要）
        if return_attention:
            # 返回未归一化的原始注意力logits [n_edges, num_heads]
            # 这样可以保留完整的得分信息，用于后续的KL对齐
            return out, attn_logits
        else:
            return out, None


class HeteroGAT(nn.Module):
    """多层异构图注意力网络"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 128],
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        dims = [input_dim] + hidden_dims
        
        self.layers = nn.ModuleList([
            HeteroGATLayer(dims[i], dims[i+1], num_heads, dropout)
            for i in range(len(dims) - 1)
        ])
        
        self.activation = nn.ReLU()
    
    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            node_feat: [n_nodes, input_dim]
            edge_index: [2, n_edges]
            edge_attr: [n_edges]
            return_attention: 是否返回注意力得分
        
        Returns:
            out: [n_nodes, hidden_dims[-1]]
            attention_scores: [n_edges, num_heads] 如果return_attention=True，否则None
        """
        x = node_feat
        
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and return_attention:
                # 最后一层返回注意力得分
                x, attn_scores = layer(x, edge_index, edge_attr, return_attention=True)
            else:
                x, _ = layer(x, edge_index, edge_attr, return_attention=False)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x, attn_scores if return_attention else None


class FusionModule(nn.Module):
    """融合模态特征"""
    
    def __init__(self, image_dim: int, expr_dim: int, fusion_dim: int = 256):
        super().__init__()
        
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.expr_proj = nn.Linear(expr_dim, fusion_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, image_feat: torch.Tensor, expr_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feat: [B, image_dim]
            expr_feat: [B, expr_dim]
        
        Returns:
            fused: [B, fusion_dim]
        """
        img_proj = self.image_proj(image_feat)
        expr_proj = self.expr_proj(expr_feat)
        
        fused = self.fusion(torch.cat([img_proj, expr_proj], dim=1))
        
        return fused


class HeteroSTModel(nn.Module):
    """异构ST通讯模型 - 简化版（仅使用基因表达）"""
    
    def __init__(self, n_genes: int = None, vae_latent_dim: int = 64, vae_hidden_dim: int = 256,
                 image_dim: int = None, fusion_dim: int = 256, 
                 gat_layers: int = 3, gat_hidden_dims: list = None,
                 gat_heads: int = 4, gat_dropout: float = 0.1,
                 output_dim: int = 64, n_celltypes: int = None, vae_encoder = None):
        super().__init__()
        
        if gat_hidden_dims is None:
            gat_hidden_dims = [256, 256, 128]
        
        self.n_genes = n_genes
        self.vae_latent_dim = vae_latent_dim
        self.output_dim = output_dim
        self.gat_hidden_dims = gat_hidden_dims
        
        # VAE编码器（必需）
        if vae_encoder is None:
            raise ValueError("必须提供预训练的VAE编码器 (vae_encoder)")
        self.vae_encoder = vae_encoder
        
        # 异构GAT
        self.gat = HeteroGAT(vae_latent_dim, gat_hidden_dims, gat_heads, gat_dropout)
        
        # 输出层
        self.output_proj = nn.Linear(gat_hidden_dims[-1], output_dim)
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # 可学习的相似度边权重调节因子
        # α_ss: spot-spot边权重调节因子
        # α_sc: spot-cell边权重调节因子
        self.alpha_ss = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
        self.alpha_sc = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
    
    def forward(self, expr_raw: torch.Tensor,
                cell_expr_raw: torch.Tensor,
                edge_index_like: torch.Tensor, edge_attr_like: torch.Tensor,
                edge_index_cc: torch.Tensor, edge_attr_cc: torch.Tensor,
                return_attention: bool = False) -> Tuple:
        """
        Args:
            expr_raw: [k+1, n_genes] 原始Spot基因表达量
            cell_expr_raw: [(k+1)*n_cells, n_marker_genes] 原始Cell基因表达量
            edge_index_like: [2, n_edges_like] 相似度边 (spot-spot + spot-celltype)
            edge_attr_like: [n_edges_like]
            edge_index_cc: [2, n_edges_cc] celltype-celltype边
            edge_attr_cc: [n_edges_cc]
            return_attention: 是否返回cell-cell边的注意力得分
        
        Returns:
            spot_repr: [k+1, output_dim] spot表示
            cell_repr: [n_cells, output_dim] cell表示
            combined: [k+1+n_celltypes, output_dim] 组合表示
            spot_proj: [k+1, output_dim] spot投影（用于对比学习）
            cc_attention: [n_edges_cc, num_heads] cell-cell注意力得分（如果return_attention=True）
        """
        # VAE编码（使用预训练VAE编码器）
        # 使用提供的VAE编码器，只取mean
        mu, log_var = self.vae_encoder(expr_raw)
        spot_latent = mu  # [k+1, vae_latent_dim]
        
        mu_cell, log_var_cell = self.vae_encoder(cell_expr_raw)
        cell_latent = mu_cell  # [(k+1)*n_cells, vae_latent_dim]
        
        n_spots = spot_latent.size(0)
        n_cells_total = cell_latent.size(0)
        
        # Spot节点特征
        spot_feat = spot_latent  # [n_spots, vae_latent_dim]
        
        # Cell节点特征
        cell_feat = cell_latent  # [n_cells_total, vae_latent_dim]
        
        # 拼接所有节点特征
        all_feat = torch.cat([spot_feat, cell_feat], dim=0)  # [n_spots+n_cells_total, vae_latent_dim]
        
        # ========== 应用可学习的边权重调节 ==========
        # 对相似度边应用可学习的权重调节因子
        if edge_index_like.size(1) > 0:
            # 区分spot-spot和spot-cell边
            src_nodes = edge_index_like[0]  # [n_edges_like]
            dst_nodes = edge_index_like[1]  # [n_edges_like]
            
            # spot-spot边：src < n_spots and dst < n_spots
            ss_mask = (src_nodes < n_spots) & (dst_nodes < n_spots)
            # spot-cell边：src < n_spots and dst >= n_spots
            sc_mask = (src_nodes < n_spots) & (dst_nodes >= n_spots)
            
            # 应用softplus确保权重为正，并调节原始权重
            edge_attr_like_regulated = edge_attr_like.clone()
            edge_attr_like_regulated[ss_mask] = edge_attr_like[ss_mask] * torch.nn.functional.softplus(self.alpha_ss)
            edge_attr_like_regulated[sc_mask] = edge_attr_like[sc_mask] * torch.nn.functional.softplus(self.alpha_sc)
        else:
            edge_attr_like_regulated = edge_attr_like
        
        # GAT编码 - 相似度边
        if edge_index_like.size(1) > 0:
            like_repr, _ = self.gat(all_feat, edge_index_like, edge_attr_like_regulated, return_attention=False)
        else:
            like_repr = all_feat
        
        # Celltype-Celltype通讯（主要关注）
        if edge_index_cc.size(1) > 0:
            cc_repr, cc_attention = self.gat(all_feat, edge_index_cc, edge_attr_cc, return_attention=return_attention)
        else:
            cc_repr = all_feat
            cc_attention = None
        
        # 加权组合：Cell-Cell边权重更大
        combined = 0.4 * like_repr + 0.6 * cc_repr
        
        # 输出投影
        repr_out = self.output_proj(combined)  # [n_spots+n_cells_total, output_dim]
        
        # 分离spot和cell的表示
        spot_repr_out = repr_out[:n_spots]  # [n_spots, output_dim]
        cell_repr_out = repr_out[n_spots:]  # [n_cells_total, output_dim]
        
        # 对比学习投影
        spot_proj = self.projection_head(spot_repr_out)  # [n_spots, output_dim]
        
        if return_attention:
            return spot_repr_out, cell_repr_out, combined, spot_proj, cc_attention
        else:
            return spot_repr_out, cell_repr_out, combined, spot_proj, None
