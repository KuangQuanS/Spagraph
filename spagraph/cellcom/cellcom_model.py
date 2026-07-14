import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


def sample_relation_negatives(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_batch: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build direction-reversal and within-graph receiver-swap negatives.

    Returns corrupted edges, inherited edge features, and the positive-edge
    index paired with each corruption. Existing directed edges and self loops
    are removed so that observed relations are not mislabeled as negatives.
    """
    n_edges = int(edge_index.size(1))
    if n_edges == 0:
        return (
            edge_index.new_empty((2, 0)),
            edge_attr.new_empty((0, edge_attr.size(-1))),
            edge_index.new_empty((0,)),
        )

    if edge_batch is None:
        edge_batch = edge_index.new_zeros(n_edges)
    else:
        edge_batch = edge_batch.to(device=edge_index.device, dtype=torch.long)
    if edge_batch.numel() != n_edges:
        raise ValueError("edge_batch must have one entry per communication edge")

    src, dst = edge_index
    candidate_edges = [torch.stack([dst, src], dim=0)]
    candidate_attrs = [edge_attr]
    positive_indices = [torch.arange(n_edges, device=edge_index.device)]

    swapped_parts = []
    swapped_attr_parts = []
    swapped_positive_parts = []
    for graph_id in torch.unique(edge_batch, sorted=True):
        indices = torch.nonzero(edge_batch == graph_id, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        permutation = torch.randperm(
            indices.numel(), device=edge_index.device, generator=generator
        )
        swapped_parts.append(
            torch.stack([src[indices], dst[indices[permutation]]], dim=0)
        )
        swapped_attr_parts.append(edge_attr[indices])
        swapped_positive_parts.append(indices)

    if swapped_parts:
        candidate_edges.append(torch.cat(swapped_parts, dim=1))
        candidate_attrs.append(torch.cat(swapped_attr_parts, dim=0))
        positive_indices.append(torch.cat(swapped_positive_parts, dim=0))

    negative_edges = torch.cat(candidate_edges, dim=1)
    negative_attrs = torch.cat(candidate_attrs, dim=0)
    paired_positive = torch.cat(positive_indices, dim=0)

    n_nodes = max(int(edge_index.max().item()) + 1, 1)
    positive_hash = src * n_nodes + dst
    negative_hash = negative_edges[0] * n_nodes + negative_edges[1]
    valid = negative_edges[0].ne(negative_edges[1])
    valid &= ~torch.isin(negative_hash, positive_hash)
    return negative_edges[:, valid], negative_attrs[valid], paired_positive[valid]


def pairwise_relation_ranking_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Smooth pairwise loss requiring observed relations to outrank controls."""
    if positive_logits.numel() == 0:
        return positive_logits.sum() * 0.0
    return torch.nn.functional.softplus(
        float(margin) - (positive_logits - negative_logits)
    ).mean()

class EdgeAttentionLayer(nn.Module):
    """基于边特征的注意力层 - 简化版Edge Attention"""

    def __init__(self, edge_dim: int, hidden_dim: int, node_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 注意力和消息计算需要结合边特征和节点特征
        attention_input_dim = edge_dim + 2 * node_dim  # 边特征 + 源节点 + 目标节点
        message_input_dim = edge_dim + node_dim  # 边特征 + 源节点

        # 边特征编码器 - 结合边特征和源节点特征生成消息
        self.edge_encoder = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重预测器 - 结合边特征、源节点和目标节点特征（动态注意力）
        self.attention_predictor = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )
        
        # ✅ 边强度预测器 - 结合边特征和节点特征评估边的重要性（输出raw logits）
        self.edge_strength_predictor = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
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

        # Handle empty edges case
        if n_edges == 0:
            # No edges, just return node features projected to hidden_dim
            out = self.node_proj(node_feat)
            out = self.output_projection(out)
            out = self.output_dropout(out)
            if return_attention:
                return out, (torch.empty(0, self.num_heads, device=edge_attr.device), torch.empty(0, device=edge_attr.device))
            else:
                return out, None

        # 获取源节点和目标节点特征
        src_feat = node_feat[edge_index[0]]  # [n_edges, node_dim]
        dst_feat = node_feat[edge_index[1]]  # [n_edges, node_dim]

        # 1. 预测边强度（结合边特征和节点特征）
        strength_input = torch.cat([edge_attr, src_feat, dst_feat], dim=-1)  # [n_edges, edge_dim + 2*node_dim]
        edge_strength_logits = self.edge_strength_predictor(strength_input).squeeze(-1)  # [n_edges]

        # 2. 计算注意力权重用于节点聚合（结合边特征和节点特征实现动态注意力）
        attention_logits = self.score_attention_logits(
            edge_attr, edge_index, node_feat
        )  # [n_edges, num_heads]
        # 使用edge-wise softmax进行归一化（对每个目标节点的所有入边）
        attention_weights = self._edge_softmax(attention_logits, edge_index[1])  # [n_edges, num_heads]

        # 3. 生成消息（结合边特征和源节点特征）
        message_input = torch.cat([edge_attr, src_feat], dim=-1)  # [n_edges, edge_dim + node_dim]
        edge_updates = self.edge_encoder(message_input)  # [n_edges, hidden_dim]

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
    
    def score_attention_logits(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Score arbitrary directed relations with the shared attention head."""
        if edge_index.size(1) == 0:
            return edge_attr.new_empty((0, self.num_heads))
        src_feat = node_feat[edge_index[0]]
        dst_feat = node_feat[edge_index[1]]
        attention_input = torch.cat([edge_attr, src_feat, dst_feat], dim=-1)
        return self.attention_predictor(attention_input)

    def _edge_softmax(self, logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        对每个目标节点的所有入边做softmax归一化 (优化版本)
        """
        n_edges = logits.size(0)
        if n_edges == 0:
            return logits
        
        num_heads = logits.size(1)
        max_node = edge_index.max().item() + 1
        
        # 数值稳定性：减去每组最大值
        max_logits = logits.new_full((max_node, num_heads), float('-inf'))
        max_logits.scatter_reduce_(0, edge_index.unsqueeze(-1).expand(-1, num_heads), 
                                   logits, reduce='amax', include_self=False)
        max_per_edge = max_logits[edge_index]  # [n_edges, num_heads]
        
        # exp(logits - max)
        exp_logits = torch.exp(logits - max_per_edge)
        
        # 求和
        sum_exp = logits.new_zeros((max_node, num_heads))
        sum_exp.scatter_add_(0, edge_index.unsqueeze(-1).expand(-1, num_heads), exp_logits)
        sum_per_edge = sum_exp[edge_index] + 1e-8  # [n_edges, num_heads]
        
        return exp_logits / sum_per_edge
class EdgeAttentionNetwork(nn.Module):
    """多层边注意力网络"""

    def __init__(self, edge_dim: int, node_dim: int, hidden_dims: list = [256, 256, 128],
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            EdgeAttentionLayer(edge_dim, hidden_dims[0], node_dim, num_heads, dropout)
        ])
        
        # 后续层都使用相同的edge_dim（因为边特征不变）和递增的hidden_dim
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                EdgeAttentionLayer(edge_dim, hidden_dims[i+1], hidden_dims[i], num_heads, dropout)
            )

        self.activation = nn.ReLU()

    def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor,
                node_feat: torch.Tensor, return_attention: bool = False,
                relation_negative_edge_index: Optional[torch.Tensor] = None,
                relation_negative_edge_attr: Optional[torch.Tensor] = None,
                return_relation_scores: bool = False):
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
        attn_scores = None
        relation_scores = None

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                if return_relation_scores:
                    positive_logits = layer.score_attention_logits(
                        edge_attr, edge_index, x
                    ).mean(dim=-1)
                    if relation_negative_edge_index is not None:
                        negative_logits = layer.score_attention_logits(
                            relation_negative_edge_attr,
                            relation_negative_edge_index,
                            x,
                        ).mean(dim=-1)
                    else:
                        negative_logits = positive_logits.new_empty((0,))
                    relation_scores = (positive_logits, negative_logits)
                # 最后一层返回注意力得分 (可能是元组)
                x, attn_scores = layer(
                    edge_attr, edge_index, x, return_attention=return_attention
                )
            else:
                x, _ = layer(edge_attr, edge_index, x, return_attention=False)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        result = (x, attn_scores if return_attention else None)
        if return_relation_scores:
            return result + (relation_scores,)
        return result
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
                 gat_hidden_dims: list = None,
                 gat_heads: int = 4, gat_dropout: float = 0.1,
                 output_dim: int = 64, n_celltypes: int = None):
        super().__init__()
        
        if gat_hidden_dims is None:
            gat_hidden_dims = [256, 256, 128]
        
        self.n_genes = n_genes
        self.mlp_latent_dim = mlp_latent_dim
        self.output_dim = output_dim
        self.gat_hidden_dims = gat_hidden_dims
        self.node_recon_head = nn.Linear(output_dim, n_genes)
        
        # ✅ Mask Token (既然要Mask，就用个Learnable Token，显得高级)
        self.mask_token = nn.Parameter(torch.zeros(1, n_genes))
        nn.init.normal_(self.mask_token, std=0.02)

        
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
        self.edge_attn_spatial = EdgeAttentionNetwork(edge_dim=2, node_dim=mlp_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout)  # 空间相似度图
        # Communication message passing uses aggregate edge strength only.
        # Candidate LR identities are handled after the GNN, because several
        # pairs can support the same aggregate edge.
        comm_edge_dim = 1
        self.edge_attn_comm = EdgeAttentionNetwork(edge_dim=comm_edge_dim, node_dim=mlp_latent_dim, hidden_dims=gat_hidden_dims, num_heads=gat_heads, dropout=gat_dropout)     # 通讯图
        
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
        # DGI相关组件已移除
    
    def forward(self, expr_raw: torch.Tensor,
                cell_expr_raw: torch.Tensor,
                edge_index_like: torch.Tensor, edge_attr_like: torch.Tensor,
                edge_index_cc: torch.Tensor, edge_attr_cc: torch.Tensor,
                return_attention: bool = False, edge_mask_ratio: float = 0.15, node_mask_ratio: float = 0.0,
                mask_generator: torch.Generator = None,
                return_relation_loss: bool = False,
                relation_edge_batch: Optional[torch.Tensor] = None,
                relation_rank_margin: float = 0.1,
                relation_generator: Optional[torch.Generator] = None) -> Tuple:
        """
        Args:
            expr_raw: [k+1, n_genes] 原始Spot基因表达量
            cell_expr_raw: [(k+1)*n_cells, n_marker_genes] 原始Cell基因表达量
            edge_index_like: [2, n_edges_like] 相似度边 (spot-spot + spot-celltype)
            edge_attr_like: [n_edges_like, 2] 相似度边特征 [weight, -1]
            edge_index_cc: [2, n_edges_cc] celltype-celltype边
            edge_attr_cc: [n_edges_cc, >=1] communication-edge features. Only
                the aggregate LR score in column 0 enters the GNN; candidate
                LR identities remain evaluation metadata.
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
            exist_logits: None（保留占位，不再使用）
            rate_pred: None（保留占位，不再使用）
        """
        n_spots = expr_raw.size(0)
        n_cells_total = cell_expr_raw.size(0)

        # ========== Step 1: Input Masking (True Masked Graph Reconstruction) ==========
        # 既然要做 Masking，就在输入进 Encoder 之前做！
        
        # 1. 准备原始特征
        expr_raw_input = expr_raw
        cell_expr_raw_input = cell_expr_raw
        
        node_mask = None
        
        # 2. 生成 Mask 并替换为 Token (只在训练时或强制Mask时)
        if node_mask_ratio > 0 and self.training:
             # 合并以统一计算 mask
            base_feat = torch.cat([expr_raw, cell_expr_raw], dim=0)
            n_total = base_feat.size(0)
            
            # 生成遮挡掩码
            rand_mask = torch.rand(n_total, device=base_feat.device, generator=mask_generator)
            node_mask = (rand_mask < node_mask_ratio)
            
            # 如果有被 mask 的节点
            if node_mask.any():
                masked_feat = base_feat.clone()
                # 用可学习的 Token 替换被 Mask 的节点特征
                masked_feat[node_mask] = self.mask_token.expand(node_mask.sum(), -1)
                
                # 拆分回 spot 和 cell
                expr_raw_input = masked_feat[:n_spots]
                cell_expr_raw_input = masked_feat[n_spots:]
            else:
                node_mask = torch.zeros(n_total, dtype=torch.bool, device=base_feat.device)
        else:
            n_total = expr_raw.size(0) + cell_expr_raw.size(0)
            node_mask = torch.zeros(n_total, dtype=torch.bool, device=expr_raw.device)

        # 3. MLP编码（使用可能是 Masked 的特征！）
        spot_latent = self.mlp_encoder(expr_raw_input)  # [k+1, mlp_latent_dim]
        cell_latent = self.mlp_encoder(cell_expr_raw_input)  # [(k+1)*n_cells, mlp_latent_dim]

        # Spot节点特征
        spot_feat = spot_latent  # [n_spots, vae_latent_dim]

        # Cell节点特征
        cell_feat = cell_latent  # [n_cells_total, vae_latent_dim]

        # 拼接所有节点特征
        all_feat = torch.cat([spot_feat, cell_feat], dim=0)  # [n_spots+n_cells_total, vae_latent_dim]

        # ========== 边Mask机制（训练时遮一部分score，验证默认不遮） ==========
        edge_mask = None
        masked_edge_attr_cc = edge_attr_cc
        predicted_masked_edges = None

        if edge_mask_ratio > 0 and edge_index_cc.size(1) > 0:
            # 随机选择一部分边进行mask（只mask score，保留id用于重构）
            n_edges_cc = edge_index_cc.size(1)
            n_mask = int(n_edges_cc * edge_mask_ratio)

            if n_mask > 0:
                mask_indices = torch.randperm(n_edges_cc, device=edge_attr_cc.device)[:n_mask]
                if mask_generator is not None:
                    mask_indices = torch.randperm(n_edges_cc, device=edge_attr_cc.device, generator=mask_generator)[:n_mask]
                edge_mask = torch.zeros(n_edges_cc, dtype=torch.bool, device=edge_attr_cc.device)
                edge_mask[mask_indices] = True

                masked_edge_attr_cc = edge_attr_cc.clone()
                masked_edge_attr_cc[mask_indices, 0] = 0

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
            # Masked edges use score=0. LR identity is deliberately excluded
            # from aggregate message passing to avoid first-pair leakage.
            comm_edge_feat = masked_edge_attr_cc[:, 0:1]

            relation_negative_edges = None
            relation_negative_attrs = None
            relation_positive_indices = None
            if return_relation_loss:
                (
                    relation_negative_edges,
                    relation_negative_attrs,
                    relation_positive_indices,
                ) = sample_relation_negatives(
                    edge_index_cc,
                    edge_attr_cc,
                    edge_batch=relation_edge_batch,
                    generator=relation_generator,
                )

            comm_outputs = self.edge_attn_comm(
                comm_edge_feat,
                edge_index_cc,
                all_feat,
                return_attention=return_attention,
                relation_negative_edge_index=relation_negative_edges,
                relation_negative_edge_attr=relation_negative_attrs,
                return_relation_scores=return_relation_loss,
            )
            if return_relation_loss:
                comm_repr, cc_attention_tuple, relation_scores = comm_outputs
            else:
                comm_repr, cc_attention_tuple = comm_outputs
                relation_scores = None

            # cc_attention_tuple: (attn_avg, edge_strength_logits) or None
            if cc_attention_tuple is not None:
                attn_avg, edge_strength_logits = cc_attention_tuple
                cc_attention = attn_avg
                # 只对被mask的边进行重构预测
                if edge_mask is not None and edge_mask.any():
                    src_idx, dst_idx = edge_index_cc
                    masked_src_idx = src_idx[edge_mask]
                    masked_dst_idx = dst_idx[edge_mask]
                    masked_src_repr = comm_repr[masked_src_idx]
                    masked_dst_repr = comm_repr[masked_dst_idx]
                    masked_attention = edge_strength_logits[edge_mask]
                    masked_edge_features = torch.cat([
                        masked_attention.unsqueeze(-1),
                        masked_src_repr,
                        masked_dst_repr
                    ], dim=-1)
                    predicted_masked_edges = self.comm_predictor(masked_edge_features).squeeze(-1)
            else:
                attn_avg, edge_strength_logits = None, None
                cc_attention = None

            if relation_scores is not None and relation_positive_indices.numel() > 0:
                positive_logits, negative_logits = relation_scores
                relation_rank_loss = pairwise_relation_ranking_loss(
                    positive_logits[relation_positive_indices],
                    negative_logits,
                    margin=relation_rank_margin,
                )
            else:
                relation_rank_loss = all_feat.sum() * 0.0
        else:
            comm_repr = self.fallback_proj(all_feat)
            cc_attention = None
            relation_rank_loss = all_feat.sum() * 0.0

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

        # 节点特征重构（计算 Loss 用的 Mask 已经在前面生成了）
        node_recon_pred = self.node_recon_head(repr_out)  # [n_nodes, n_genes]
        # node_mask 已经在 Step 1 生成


        # 对比学习投影
        spot_proj = self.projection_head(spot_repr_out)  # [n_spots, output_dim]

        # ✅ 返回结果（去掉exist_logits和rate_pred）
        output_attention = cc_attention if return_attention else None
        outputs = (
            spot_repr_out, cell_repr_out, combined, spot_proj, output_attention,
            predicted_masked_edges, edge_mask, node_recon_pred, node_mask,
        )
        if return_relation_loss:
            return outputs + (relation_rank_loss,)
        return outputs


# ==================== 辅助方法 ====================
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
