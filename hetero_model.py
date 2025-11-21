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

        # 计算每条边跨头的平均注意力 (attn_avg)
        attn_avg = attention_weights.mean(dim=-1)  # [n_edges]

        # 返回边强度logits（用于监督学习和评估）以及实际用于聚合的归一化注意力
        if return_attention:
            # 返回 (attn_avg, edge_strength_logits) 作为元组，便于上游使用真实attention用于报告和sigmoid/logits用于监督
            return out, (attn_avg, edge_strength_logits)
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
                # 最后一层返回注意力得分 (可能是元组)
                x, attn_scores = layer(edge_attr, edge_index, x, return_attention=True)
            else:
                x, _ = layer(edge_attr, edge_index, x, return_attention=False)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x, attn_scores if return_attention else None
class MLPEncoder(nn.Module):
    """MLP编码器 - 替代VAE的简单多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128], latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 构建MLP层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] 输入特征
            
        Returns:
            latent: [batch_size, latent_dim] 编码后的特征
        """
        return self.encoder(x)


class HeteroSTModel(nn.Module):
    """异构ST通讯模型 - 使用MLP编码器"""
    
    def __init__(self, n_genes: int = None, mlp_latent_dim: int = 64, mlp_hidden_dims: list = [256, 128],
                 image_dim: int = None, fusion_dim: int = 256, 
                 gat_layers: int = 3, gat_hidden_dims: list = None,
                 gat_heads: int = 4, gat_dropout: float = 0.1,
                 output_dim: int = 64, n_celltypes: int = None,
                 temperature: float = 1.0):
        super().__init__()
        
        if gat_hidden_dims is None:
            gat_hidden_dims = [256, 256, 128]
        
        self.n_genes = n_genes
        self.mlp_latent_dim = mlp_latent_dim
        self.output_dim = output_dim
        self.gat_hidden_dims = gat_hidden_dims
        
        # MLP编码器替代VAE
        self.mlp_encoder = MLPEncoder(
            input_dim=n_genes,
            hidden_dims=mlp_hidden_dims,
            latent_dim=mlp_latent_dim,
            dropout=gat_dropout
        )
        
        # ✅ Edge Attention网络 - 基于边特征的注意力
        # 注意：统一使用edge_dim=2，所有边都包含[score, id]
        # 对于相似度边：[weight, -1]（-1表示非通讯边）
        # 对于通讯边：[lr_score, lr_id]
        self.edge_attn_spatial = EdgeAttentionNetwork(edge_dim=2, node_dim=mlp_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout, temperature=temperature)  # 空间相似度图
        self.edge_attn_comm = EdgeAttentionNetwork(edge_dim=2, node_dim=mlp_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout, temperature=temperature)     # 通讯图
        
        # ✅ 备用投影层：当没有边时，将 MLP latent 投影到 GAT 输出维度
        self.fallback_proj = nn.Linear(mlp_latent_dim, gat_hidden_dims[-1])
        
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
        
        # ✅ 去掉双头架构：边存在性判别器 + 边强度回归器
        # 边存在性判别器（用于识别假阳性边）
        # self.edge_exist_head = nn.Sequential(
        #     nn.Linear(gat_hidden_dims[-1] * 2, gat_hidden_dims[-1]),
        #     nn.ReLU(),
        #     nn.Dropout(gat_dropout),
        #     nn.Linear(gat_hidden_dims[-1], 1)
        # )
        
        # 边强度回归器（预测真实的通讯强度）
        # self.edge_rate_head = nn.Sequential(
        #     nn.Linear(gat_hidden_dims[-1] * 2, gat_hidden_dims[-1]),
        #     nn.ReLU(),
        #     nn.Dropout(gat_dropout),
        #     nn.Linear(gat_hidden_dims[-1], 1)
        # )
        
        # 可学习的相似度边权重调节因子
        # α_ss: spot-spot边权重调节因子
        # α_sc: spot-cell边权重调节因子
        self.alpha_ss = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
        self.alpha_sc = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0
        # DGI discriminator weight（双线性判别器）
        self.dgi_discriminator_weight = nn.Parameter(torch.randn(gat_hidden_dims[-1], gat_hidden_dims[-1]))
        nn.init.xavier_uniform_(self.dgi_discriminator_weight)
        # DGI readout mode
        self.dgi_readout_mode = 'mean'
    
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
            cc_attention: [n_edges_cc] cell-cell注意力得分/边强度logits（如果return_attention=True）
            predicted_masked_edges: 被mask边的预测值（用于训练）
            edge_mask: 边mask信息
            exist_logits: [n_edges_cc] 边存在性logits（用于BCE损失）
            rate_pred: [n_edges_cc] 边强度预测值（回归目标）
        """
        # MLP编码（使用log1p预处理）
        # 对基因表达进行log1p预处理
        expr_raw_log = torch.log1p(expr_raw)  # [k+1, n_genes]
        cell_expr_raw_log = torch.log1p(cell_expr_raw)  # [(k+1)*n_cells, n_genes]
        
        spot_latent = self.mlp_encoder(expr_raw_log)  # [k+1, mlp_latent_dim]
        cell_latent = self.mlp_encoder(cell_expr_raw_log)  # [(k+1)*n_cells, mlp_latent_dim]

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
            comm_repr, cc_attention_tuple = self.edge_attn_comm(
                masked_edge_attr_cc, edge_index_cc, all_feat, return_attention=return_attention
            )

            # cc_attention_tuple: (attn_avg, edge_strength_logits) or None
            if cc_attention_tuple is not None:
                attn_avg, edge_strength_logits = cc_attention_tuple
                # cc_attention (external) => normalized per-edge attention used by GAT
                cc_attention = attn_avg
            else:
                attn_avg, edge_strength_logits = None, None
                cc_attention = None

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
                # masked_attention用于mask预测器，应使用edge_strength_logits（raw logits）
                masked_attention = edge_strength_logits[edge_mask]  # [n_masked] - edge_strength_logits

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

        # ✅ 去掉双头预测：边存在性判别 + 边强度回归
        # exist_logits = None
        # rate_pred = None
        # if edge_index_cc.size(1) > 0:
        #     # 获取边表示：源节点和目标节点的表示
        #     src_repr = comm_repr[edge_index_cc[0]]  # [n_edges, hidden_dim]
        #     dst_repr = comm_repr[edge_index_cc[1]]  # [n_edges, hidden_dim]
        #     edge_repr = torch.cat([src_repr, dst_repr], dim=-1)  # [n_edges, hidden_dim*2]
        #     
        #     # 边存在性判别器（输出logits，BCE with logits会自动处理sigmoid）
        #     exist_logits = self.edge_exist_head(edge_repr).squeeze(-1)  # [n_edges]
        #     
        #     # 边强度回归器（使用softplus确保非负）
        #     rate_pred = torch.nn.functional.softplus(
        #         self.edge_rate_head(edge_repr).squeeze(-1)
        #     )  # [n_edges]

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

        # ✅ 返回结果（去掉exist_logits和rate_pred）
        if return_attention:
            return spot_repr_out, cell_repr_out, combined, spot_proj, cc_attention, predicted_masked_edges, edge_mask, None, None
        else:
            return spot_repr_out, cell_repr_out, combined, spot_proj, None, predicted_masked_edges, edge_mask, None, None


    # ==================== DGI相关方法 ====================
    def _readout(self, node_embeddings: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
        if mode == 'mean':
            return node_embeddings.mean(dim=0)
        elif mode == 'sum':
            return node_embeddings.sum(dim=0)
        elif mode == 'gated':
            gate = torch.sigmoid(nn.Linear(node_embeddings.size(1), node_embeddings.size(1)).to(node_embeddings.device)(node_embeddings))
            return (gate * node_embeddings).mean(dim=0)
        else:
            return node_embeddings.mean(dim=0)

    def corrupt_features(self, features: torch.Tensor, mode: str = 'feature_mask', mask_ratio: float = 0.3, noise_std: float = 0.1) -> torch.Tensor:
        corrupted = features.clone()
        if mode == 'feature_mask':
            mask = (torch.rand_like(features) > mask_ratio).float()
            corrupted = corrupted * mask
        elif mode == 'gaussian_noise':
            noise = torch.randn_like(features) * noise_std
            corrupted = corrupted + noise
        elif mode == 'shuffle':
            perm = torch.randperm(features.size(0), device=features.device)
            corrupted = corrupted[perm]
        else:
            mask = (torch.rand_like(features) > mask_ratio).float()
            corrupted = corrupted * mask
        return corrupted

    def corrupt_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, mode: str = 'edge_drop_random', mask_ratio: float = 0.2, epoch: int = None, total_epochs: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        n_edges = edge_index.size(1)
        device = edge_index.device
        if edge_attr is None or edge_attr.numel() == 0:
            edge_scores = torch.zeros(n_edges, device=device)
        else:
            edge_scores = edge_attr[:, 0].view(-1)
        if n_edges == 0:
            return edge_index, edge_attr
        min_s, max_s = float(edge_scores.min()), float(edge_scores.max())
        norm_scores = (edge_scores - min_s) / (max(1e-8, max_s - min_s))

        if mode == 'edge_drop_random':
            keep_mask = (torch.rand(n_edges, device=device) > mask_ratio)
        elif mode == 'edge_drop_low':
            thr = torch.quantile(edge_scores, mask_ratio)
            keep_mask = edge_scores > thr
        elif mode == 'edge_drop_high':
            thr = torch.quantile(edge_scores, 1.0 - mask_ratio)
            keep_mask = edge_scores < thr
        elif mode == 'edge_drop_weighted':
            prob = norm_scores * mask_ratio
            keep_mask = (torch.rand(n_edges, device=device) > prob)
        elif mode == 'edge_drop_anneal':
            if epoch is None or total_epochs is None:
                keep_mask = (torch.rand(n_edges, device=device) > mask_ratio)
            else:
                alpha = min(1.0, epoch / max(1, total_epochs))
                rand_mask = (torch.rand(n_edges, device=device) > mask_ratio)
                prob = norm_scores * mask_ratio
                weighted_mask = (torch.rand(n_edges, device=device) > prob)
                # mix
                keep_mask = torch.where(torch.rand(n_edges, device=device) < alpha, weighted_mask, rand_mask)
        else:
            keep_mask = (torch.rand(n_edges, device=device) > mask_ratio)

        if keep_mask.sum() == 0:
            idx = torch.randint(0, n_edges, (1,), device=device)
            keep_mask[idx] = True
        return edge_index[:, keep_mask], edge_attr[keep_mask]

    def compute_dgi_loss(self, expr_raw: torch.Tensor, cell_expr_raw: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                         corruption_mode: str = 'feature_mask', mask_ratio: float = 0.3,
                         edge_drop_mode: str = 'edge_drop_random', edge_drop_rate: float = 0.2,
                         epoch: int = None, total_epochs: int = None) -> torch.Tensor:
        feat_spot = self.mlp_encoder(torch.log1p(expr_raw))
        feat_cell = self.mlp_encoder(torch.log1p(cell_expr_raw))
        all_features = torch.cat([feat_spot, feat_cell], dim=0)

        # positive embeddings
        pos_embeddings, _ = self.edge_attn_spatial(edge_attr, edge_index, all_features, return_attention=False)
        summary = self._readout(pos_embeddings, mode=self.dgi_readout_mode)

        # corruption
        corrupted_features = self.corrupt_features(all_features, mode=corruption_mode, mask_ratio=mask_ratio)
        corrupted_edge_index, corrupted_edge_attr = self.corrupt_edges(edge_index, edge_attr, mode=edge_drop_mode, mask_ratio=edge_drop_rate, epoch=epoch, total_epochs=total_epochs)
        neg_embeddings, _ = self.edge_attn_spatial(corrupted_edge_attr, corrupted_edge_index, corrupted_features, return_attention=False)

        pos_scores = torch.matmul(pos_embeddings, torch.matmul(self.dgi_discriminator_weight, summary))
        neg_scores = torch.matmul(neg_embeddings, torch.matmul(self.dgi_discriminator_weight, summary))

        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        return pos_loss + neg_loss
