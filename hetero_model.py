import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math

class EdgeAttentionLayer(nn.Module):
    """基于边特征的注意力层 - Edge Attention"""

    def __init__(self, edge_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 边特征编码器 - 将边特征转换为节点表示的更新
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重预测器 - 基于边特征预测注意力权重
        self.attention_predictor = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid()  # 输出[0,1]范围的注意力权重
        )

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

class EdgeAttentionLayer(nn.Module):
    """基于边特征的注意力层 - 简化版Edge Attention"""

    def __init__(self, edge_dim: int, hidden_dim: int, node_dim: int, num_heads: int = 4, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature  # 温度系数，用于控制注意力分布的尖锐程度

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 边特征编码器 - 将边特征转换为节点表示的更新
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重预测器 - 用于节点聚合（内部使用softmax归一化）
        self.attention_predictor = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )
        
        # ✅ 边强度预测器 - 用于评估边的重要性（输出raw logits）
        self.edge_strength_predictor = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 单头输出，预测边强度
        )

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        # ✅ 节点特征投影层（预先定义）
        self.node_proj = nn.Linear(node_dim, hidden_dim)

    def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor,
                node_feat: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            edge_attr: [n_edges, edge_dim] 边特征
            edge_index: [2, n_edges] 边索引
            node_feat: [n_nodes, node_dim] 节点特征（用于聚合）
            return_attention: 是否返回注意力得分

        Returns:
            out: [n_nodes, hidden_dim] 节点表示
            attention_scores: [n_edges, num_heads] 边注意力得分
        """
        n_edges = edge_attr.size(0)
        n_nodes = node_feat.size(0)

        # 1. 预测边强度（raw logits，用于监督学习和评估）
        edge_strength_logits = self.edge_strength_predictor(edge_attr).squeeze(-1)  # [n_edges]

        # 2. 计算注意力权重用于节点聚合（对每个目标节点的边做softmax归一化）
        attention_logits = self.attention_predictor(edge_attr)  # [n_edges, num_heads]
        # 使用edge-wise softmax进行归一化（对每个目标节点的所有入边）
        attention_weights = self._edge_softmax(attention_logits, edge_index[1])  # [n_edges, num_heads]

        # 3. 基于边特征计算边更新
        edge_updates = self.edge_encoder(edge_attr)  # [n_edges, hidden_dim]

        # Reshape为多头
        edge_updates = edge_updates.view(n_edges, self.num_heads, self.head_dim)  # [n_edges, num_heads, head_dim]

        # 4. 使用注意力权重聚合边信息到节点
        dst_idx = edge_index[1]  # [n_edges]

        # 扩展权重用于聚合
        attn_weights_expanded = attention_weights.unsqueeze(-1)  # [n_edges, num_heads, 1]
        weighted_updates = edge_updates * attn_weights_expanded  # [n_edges, num_heads, head_dim]

        # ✅ 节点特征投影（预先定义）
        node_proj = self.node_proj(node_feat)  # [n_nodes, hidden_dim]
        
        # 初始化输出（从投影后的节点特征开始）
        # 将节点特征reshape为多头格式
        out = node_proj.view(n_nodes, self.num_heads, self.head_dim)  # [n_nodes, num_heads, head_dim]

        # 聚合边更新到目标节点 - 使用非in-place版本避免梯度计算错误
        # 创建索引张量用于scatter_add
        dst_idx_expanded = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim)  # [n_edges, num_heads, head_dim]
        out = out.scatter_add(0, dst_idx_expanded, weighted_updates)  # ✅ 非in-place版本

        # 重塑
        out = out.view(n_nodes, self.hidden_dim)

        # 输出投影
        out = self.output_projection(out)
        out = self.output_dropout(out)

        # 返回边强度logits（用于监督学习和评估）
        if return_attention:
            return out, edge_strength_logits  # 返回raw logits，不是sigmoid后的
        else:
            return out, None
    
    def _edge_softmax(self, logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        对每个目标节点的所有入边做softmax归一化
        
        Args:
            logits: [n_edges, num_heads] 注意力logits
            edge_index: [n_edges] 目标节点索引
            
        Returns:
            attention_weights: [n_edges, num_heads] 归一化后的注意力权重
        """
        # 为每个目标节点计算softmax
        # 1. 找到logits的最大值（用于数值稳定性）
        max_logits = torch.zeros(edge_index.max() + 1, logits.size(1), 
                                  device=logits.device, dtype=logits.dtype)
        max_logits.scatter_reduce_(0, edge_index.unsqueeze(-1).expand_as(logits), 
                                   logits, reduce='amax', include_self=False)
        max_logits = max_logits[edge_index]  # [n_edges, num_heads]
        
        # 2. 计算exp(logits - max_logits)
        exp_logits = torch.exp(logits - max_logits)
        
        # 3. 计算每个目标节点的exp_logits之和
        sum_exp = torch.zeros(edge_index.max() + 1, logits.size(1),
                              device=logits.device, dtype=logits.dtype)
        sum_exp.scatter_add_(0, edge_index.unsqueeze(-1).expand_as(exp_logits), exp_logits)
        sum_exp = sum_exp[edge_index]  # [n_edges, num_heads]
        
        # 4. 归一化
        attention_weights = exp_logits / (sum_exp + 1e-8)
        
        return attention_weights
class EdgeAttentionNetwork(nn.Module):
    """多层边注意力网络"""

    def __init__(self, edge_dim: int, node_dim: int, hidden_dims: list = [256, 256, 128],
                 num_heads: int = 4, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()

        self.layers = nn.ModuleList([
            EdgeAttentionLayer(edge_dim, hidden_dims[0], node_dim, num_heads, dropout, temperature)
        ])
        
        # 后续层都使用相同的edge_dim（因为边特征不变）和递增的hidden_dim
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                EdgeAttentionLayer(edge_dim, hidden_dims[i+1], hidden_dims[i], num_heads, dropout, temperature)
            )

        self.activation = nn.ReLU()

    def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor,
                node_feat: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            edge_attr: [n_edges, edge_dim]
            edge_index: [2, n_edges]
            node_feat: [n_nodes, node_dim]
            return_attention: 是否返回注意力得分

        Returns:
            out: [n_nodes, hidden_dims[-1]]
            attention_scores: [n_edges, num_heads] 如果return_attention=True，否则None
        """
        x = node_feat

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1 and return_attention:
                # 最后一层返回注意力得分
                x, attn_scores = layer(edge_attr, edge_index, x, return_attention=True)
            else:
                x, _ = layer(edge_attr, edge_index, x, return_attention=False)
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
                 output_dim: int = 64, n_celltypes: int = None, vae_encoder = None,
                 temperature: float = 1.0):
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
        
        # ✅ Edge Attention网络 - 基于边特征的注意力
        # 注意：统一使用edge_dim=2，所有边都包含[score, id]
        # 对于相似度边：[weight, -1]（-1表示非通讯边）
        # 对于通讯边：[lr_score, lr_id]
        self.edge_attn_spatial = EdgeAttentionNetwork(edge_dim=2, node_dim=vae_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout, temperature=temperature)  # 空间相似度图
        self.edge_attn_comm = EdgeAttentionNetwork(edge_dim=2, node_dim=vae_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout, temperature=temperature)     # 通讯图
        
        # ✅ 备用投影层：当没有边时，将 VAE latent 投影到 GAT 输出维度
        self.fallback_proj = nn.Linear(vae_latent_dim, gat_hidden_dims[-1])
        
        # ✅ 融合层：融合空间表示和通讯表示
        self.fusion_layer = nn.Sequential(
            nn.Linear(gat_hidden_dims[-1] * 2, gat_hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(gat_dropout)
        )
        
        # 输出层
        self.output_proj = nn.Linear(gat_hidden_dims[-1], output_dim)
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # ✅ 通讯强度预测头（基于注意力机制）
        # 输入：注意力得分 + 节点表示
        self.comm_predictor = nn.Sequential(
            nn.Linear(1 + gat_hidden_dims[-1] * 2, gat_hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(gat_dropout),
            nn.Linear(gat_hidden_dims[-1], 1)
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
                return_attention: bool = False, edge_mask_ratio: float = 0.15) -> Tuple:
        """
        Args:
            expr_raw: [k+1, n_genes] 原始Spot基因表达量
            cell_expr_raw: [(k+1)*n_cells, n_marker_genes] 原始Cell基因表达量
            edge_index_like: [2, n_edges_like] 相似度边 (spot-spot + spot-celltype)
            edge_attr_like: [n_edges_like, 2] 相似度边特征 [weight, -1]
            edge_index_cc: [2, n_edges_cc] celltype-celltype边
            edge_attr_cc: [n_edges_cc, 2] cell-cell边特征 [lr_score, lr_id]
            return_attention: 是否返回cell-cell边的注意力得分
            edge_mask_ratio: 边mask比例 (0.1-0.2)

        Returns:
            spot_repr: [k+1, output_dim] spot表示
            cell_repr: [n_cells, output_dim] cell表示
            combined: [k+1+n_celltypes, output_dim] 组合表示
            spot_proj: [k+1, output_dim] spot投影（用于对比学习）
            cc_attention: [n_edges_cc, num_heads] cell-cell注意力得分（如果return_attention=True）
            predicted_masked_edges: 被mask边的预测值（用于训练）
            edge_mask: 边mask信息
        """
        # VAE编码（使用预训练VAE编码器）
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

        # ========== 边Mask机制 ==========
        # 对通讯边进行mask，用于训练边预测任务
        edge_mask = None
        masked_edge_attr_cc = edge_attr_cc
        predicted_masked_edges = None

        if self.training and edge_index_cc.size(1) > 0:
            # 随机选择一部分边进行mask
            n_edges_cc = edge_index_cc.size(1)
            n_mask = int(n_edges_cc * edge_mask_ratio)

            if n_mask > 0:
                # 随机选择要mask的边
                mask_indices = torch.randperm(n_edges_cc, device=edge_attr_cc.device)[:n_mask]
                edge_mask = torch.zeros(n_edges_cc, dtype=torch.bool, device=edge_attr_cc.device)
                edge_mask[mask_indices] = True

                # 创建mask后的边特征（被mask的边特征设为0）
                masked_edge_attr_cc = edge_attr_cc.clone()
                masked_edge_attr_cc[mask_indices] = 0

        # ========== 应用可学习的边权重调节 ==========
        # 对相似度边应用可学习的权重调节因子（只调节权重部分）
        if edge_index_like.size(1) > 0:
            # 区分spot-spot和spot-cell边
            src_nodes = edge_index_like[0]  # [n_edges_like]
            dst_nodes = edge_index_like[1]  # [n_edges_like]

            # spot-spot边：src < n_spots and dst < n_spots
            ss_mask = (src_nodes < n_spots) & (dst_nodes < n_spots)
            # spot-cell边：src < n_spots and dst >= n_spots
            sc_mask = (src_nodes < n_spots) & (dst_nodes >= n_spots)

            # 应用softplus确保权重为正，并调节原始权重（只调节第一个维度）
            edge_attr_like_regulated = edge_attr_like.clone()
            edge_attr_like_regulated[ss_mask, 0] = edge_attr_like[ss_mask, 0] * torch.nn.functional.softplus(self.alpha_ss)
            edge_attr_like_regulated[sc_mask, 0] = edge_attr_like[sc_mask, 0] * torch.nn.functional.softplus(self.alpha_sc)
            # 第二个维度（ID）保持不变（已经是-1）
        else:
            edge_attr_like_regulated = edge_attr_like

        # ========== Edge Attention处理 ==========
        # ✅ 空间相似度边 - 使用Edge Attention
        if edge_index_like.size(1) > 0:
            spatial_repr, _ = self.edge_attn_spatial(
                edge_attr_like_regulated,  # [n_edges_like, 1]
                edge_index_like, all_feat, return_attention=False
            )
        else:
            spatial_repr = self.fallback_proj(all_feat)

        # ✅ 通讯边 - 使用Edge Attention
        if edge_index_cc.size(1) > 0:
            comm_repr, cc_attention = self.edge_attn_comm(
                masked_edge_attr_cc, edge_index_cc, all_feat, return_attention=return_attention
            )

            # 如果有mask，预测被mask的边
            if edge_mask is not None and edge_mask.any():
                # 使用注意力权重和节点表示来预测被mask的边
                src_idx, dst_idx = edge_index_cc
                masked_src_idx = src_idx[edge_mask]
                masked_dst_idx = dst_idx[edge_mask]

                # 获取节点表示
                masked_src_repr = comm_repr[masked_src_idx]  # [n_masked, hidden_dim]
                masked_dst_repr = comm_repr[masked_dst_idx]  # [n_masked, hidden_dim]

                # 获取对应的边强度logits（已经是1维的）
                masked_attention = cc_attention[edge_mask]  # [n_masked] - edge_strength_logits

                # 组合特征进行预测
                masked_edge_features = torch.cat([
                    masked_attention.unsqueeze(-1),  # [n_masked, 1]
                    masked_src_repr,                 # [n_masked, hidden_dim]
                    masked_dst_repr                   # [n_masked, hidden_dim]
                ], dim=-1)  # [n_masked, 1 + hidden_dim * 2]

                predicted_masked_edges = self.comm_predictor(masked_edge_features).squeeze(-1)  # [n_masked]
        else:
            comm_repr = self.fallback_proj(all_feat)
            cc_attention = None

        # ✅ 融合空间表示和通讯表示
        combined_feat = torch.cat([spatial_repr, comm_repr], dim=-1)  # [n_nodes, hidden_dim*2]
        combined = self.fusion_layer(combined_feat)  # [n_nodes, hidden_dim]

        # 输出投影
        repr_out = self.output_proj(combined)  # [n_spots+n_cells_total, output_dim]

        # 分离spot和cell的表示
        spot_repr_out = repr_out[:n_spots]  # [n_spots, output_dim]
        cell_repr_out = repr_out[n_spots:]  # [n_cells_total, output_dim]

        # 对比学习投影
        spot_proj = self.projection_head(spot_repr_out)  # [n_spots, output_dim]

        # 返回结果
        if return_attention:
            return spot_repr_out, cell_repr_out, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask
        else:
            return spot_repr_out, cell_repr_out, combined, spot_proj, None, predicted_masked_edges, edge_mask
