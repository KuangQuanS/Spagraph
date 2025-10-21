import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional

# ================================
# Stage 1: VAE Models
# ================================

class VAEEncoder(nn.Module):
    """VAE编码器"""
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 均值和方差分支
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

class VAEDecoder(nn.Module):
    """VAE解码器"""
    def __init__(self, latent_dim, hidden_dims=[256, 512], output_dim=None, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # 解码器
        decoder_hidden = hidden_dims[::-1]  # 反向
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim,
            dropout=dropout
        )
        
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        # 编码
        mu, log_var = self.encoder(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, log_var)
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var, z
    
    def encode(self, x):
        """仅编码，返回潜在表示"""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

# Loss Functions
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """VAE损失函数：重建损失 + KL散度"""
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div

# ================================
# Stage 2: GAT Models
# ================================

class HeterogeneousGATDeconvolution(nn.Module):
    """异构图注意力网络解卷积模型"""
    def __init__(self, embedding_dim=128,n_cell_types=9,gat_hidden_dim=64,gat_layers=3,gat_heads=4,
                 dropout=0.1,k_spatial=6,similarity_threshold=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_cell_types = n_cell_types
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        self.k_spatial = k_spatial
        self.similarity_threshold = similarity_threshold
        
        # 1. 节点嵌入层
        # Spot节点：从VAE embedding到GAT输入
        self.spot_projection = nn.Sequential(
            nn.Linear(embedding_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CellType节点：可学习的嵌入
        self.celltype_embeddings = nn.Parameter(
            torch.randn(n_cell_types, gat_hidden_dim)
        )
        
        # 2. GAT层序列
        self.gat_layers_list = nn.ModuleList()
        for i in range(gat_layers):
            if i == 0:
                # 第一层：处理异构输入
                gat_layer = GATConv(
                    in_channels=gat_hidden_dim,
                    out_channels=gat_hidden_dim // gat_heads,
                    heads=gat_heads,
                    dropout=dropout,
                    concat=True
                )
            elif i == gat_layers - 1:
                # 最后一层：输出层
                gat_layer = GATConv(
                    in_channels=gat_hidden_dim,
                    out_channels=gat_hidden_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False
                )
            else:
                # 中间层
                gat_layer = GATConv(
                    in_channels=gat_hidden_dim,
                    out_channels=gat_hidden_dim // gat_heads,
                    heads=gat_heads,
                    dropout=dropout,
                    concat=True
                )
            
            self.gat_layers_list.append(gat_layer)
        
        # 3. 简化的注意力权重计算（向量化，高效）
        self.attention_mlp = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim, 1)

        )
        
    def build_heterogeneous_graph(self, 
                                spot_embeddings: torch.Tensor,
                                spatial_coords: torch.Tensor,
                                celltype_prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建异构图：Spot节点 + CellType节点"""
        n_spots = spot_embeddings.shape[0]
        n_cell_types = celltype_prototypes.shape[0]
        device = spot_embeddings.device
        
        # 1. 处理节点特征
        # Spot节点特征
        spot_features = self.spot_projection(spot_embeddings)  # [n_spots, gat_hidden_dim]
        
        # CellType节点特征（可学习参数）
        celltype_features = self.celltype_embeddings  # [n_cell_types, gat_hidden_dim]
        
        # 合并节点特征 [spots; celltypes]
        node_features = torch.cat([spot_features, celltype_features], dim=0)
        
        # 2. 构建边
        edge_indices = []
        edge_attrs = []
        
        # 2.1 Spot-Spot边（基于空间距离的KNN）
        if len(spatial_coords) > 1:
            # 计算KNN
            coords_np = spatial_coords.detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=min(self.k_spatial+1, len(coords_np))).fit(coords_np)
            distances, indices = nbrs.kneighbors(coords_np)
            
            for i in range(len(indices)):
                for j in range(1, len(indices[i])):  # 跳过自己（第0个）
                    neighbor_idx = indices[i][j]
                    distance = distances[i][j]
                    
                    # 转换为相似度权重
                    weight = np.exp(-distance / np.std(distances))
                    
                    # 双向边
                    edge_indices.append([i, neighbor_idx])
                    edge_attrs.append([weight])
                    edge_indices.append([neighbor_idx, i])
                    edge_attrs.append([weight])
        
        # 2.2 Spot-CellType边（基于余弦相似度）
        # 计算spot与celltype原型的余弦相似度
        spot_emb_np = spot_embeddings.detach().cpu().numpy()
        celltype_emb_np = celltype_prototypes.detach().cpu().numpy()
        
        similarity_matrix = cosine_similarity(spot_emb_np, celltype_emb_np)  # [n_spots, n_cell_types]
        
        for spot_idx in range(n_spots):
            for celltype_idx in range(n_cell_types):
                similarity = similarity_matrix[spot_idx, celltype_idx]
                
                # 只连接相似度超过阈值的边
                if similarity > self.similarity_threshold:
                    # Spot -> CellType
                    edge_indices.append([spot_idx, n_spots + celltype_idx])
                    edge_attrs.append([similarity])
                    
                    # CellType -> Spot
                    edge_indices.append([n_spots + celltype_idx, spot_idx])
                    edge_attrs.append([similarity])
        
        # 转换为tensor
        if len(edge_indices) > 0:
            edge_index = torch.LongTensor(edge_indices).t().contiguous().to(device)
            edge_attr = torch.FloatTensor(edge_attrs).to(device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=device)
        
        return edge_index, edge_attr, node_features
    
    def forward(self, 
               spot_embeddings: torch.Tensor,
               spatial_coords: torch.Tensor,
               celltype_prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播 """
        n_spots = spot_embeddings.shape[0]
        
        # 1. 构建异构图
        edge_index, edge_attr, node_features = self.build_heterogeneous_graph(
            spot_embeddings, spatial_coords, celltype_prototypes
        )
        
        # 2. GAT处理
        x = node_features
        for i, gat_layer in enumerate(self.gat_layers_list):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers_list) - 1:  # 除了最后一层都用ReLU
                x = F.relu(x)
        
        # 3. 分离spot和celltype节点特征
        spot_features = x[:n_spots]  # [n_spots, gat_hidden_dim]
        celltype_features = x[n_spots:]  # [n_cell_types, gat_hidden_dim]
        
        # 4. 向量化计算注意力权重（高效实现）
        # 扩展维度进行批量计算: [n_spots, n_cell_types, gat_hidden_dim]
        spot_expanded = spot_features.unsqueeze(1).expand(-1, self.n_cell_types, -1)  # [n_spots, n_cell_types, gat_hidden_dim]
        cell_expanded = celltype_features.unsqueeze(0).expand(n_spots, -1, -1)        # [n_spots, n_cell_types, gat_hidden_dim]
        
        # 拼接特征: [n_spots, n_cell_types, 2*gat_hidden_dim]
        combined = torch.cat([spot_expanded, cell_expanded], dim=-1)
        
        # 批量计算注意力分数
        attention_scores = self.attention_mlp(combined).squeeze(-1)  # [n_spots, n_cell_types]
        
        # softmax归一化，确保每个spot的权重和为1
        deconv_weights = F.softmax(attention_scores, dim=1)  # [n_spots, n_cell_types]
        
        return {
            'spot_features': spot_features,
            'celltype_features': celltype_features,
            'attention_scores': attention_scores,    # 原始分数（用于正则化）
            'deconv_weights': deconv_weights,        # 解卷积权重（归一化后，用于重建）
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }


# ================================
# Stage 2: Loss Functions
# ================================

class SpatialDeconvolutionLoss(nn.Module):
    """空间解卷积损失函数
    
    L_total = λ_pearson·L_pearson + λ_cosine·L_cosine + λ_align·L_align + λ_reg·L_reg + λ_sparse·L_sparse
    
    其中：
    - L_pearson: Pearson相关系数损失（基因表达相似性）
    - L_cosine: Cosine相似度损失（基因表达相似性）
    - L_align: Cosine Alignment模态对齐损失（embedding空间）
    - L_reg: 权重正则化
    - L_sparse: 稀疏性正则化
    """
    
    def __init__(self, lambda_pearson=1.0, lambda_cosine=1.0, lambda_align=1.0, 
                 lambda_reg=0.5, lambda_sparse=0.01):
        super().__init__()
        self.lambda_pearson = lambda_pearson  # Pearson损失权重
        self.lambda_cosine = lambda_cosine    # Cosine损失权重
        self.lambda_align = lambda_align      # 模态对齐权重
        self.lambda_reg = lambda_reg          # 权重正则化权重
        self.lambda_sparse = lambda_sparse    # 稀疏性正则化权重
        
    def forward(self, 
               attention_weights: torch.Tensor,
               celltype_expression: torch.Tensor,
               true_spot_expression: torch.Tensor,
               spot_embedding: torch.Tensor = None,
               celltype_embedding: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            attention_weights: 注意力权重 [n_spots, n_cell_types]（已softmax）
            celltype_expression: 细胞类型表达 [n_cell_types, n_genes]
            true_spot_expression: 真实spot表达 [n_spots, n_genes]
            spot_embedding: spot embedding [n_spots, embedding_dim]（可选）
            celltype_embedding: 细胞类型embedding [n_cell_types, embedding_dim]（可选）
            
        Returns:
            损失字典
        """
        n_spots = attention_weights.shape[0]
        
        # 计算重建的spot表达
        reconstructed_spot = torch.matmul(attention_weights, celltype_expression)  # [n_spots, n_genes]
        
        # ============ 1. Pearson Correlation Loss ============
        # Pearson相关系数（衡量线性相关性）
        pred_centered = reconstructed_spot - reconstructed_spot.mean(dim=1, keepdim=True)
        target_centered = true_spot_expression - true_spot_expression.mean(dim=1, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1) + 1e-8)
        pearson_corr = numerator / denominator
        L_pearson = 1.0 - pearson_corr.mean()
        
        # ============ 2. Cosine Similarity Loss ============
        # Cosine相似度（越接近1越好）
        cos_sim_rec = F.cosine_similarity(reconstructed_spot, true_spot_expression, dim=-1)
        L_cosine = 1.0 - cos_sim_rec.mean()
        
        # ============ 3. Cosine Alignment Loss（可选） ============
        L_align = torch.tensor(0.0, device=attention_weights.device)
        
        if spot_embedding is not None and celltype_embedding is not None:
            # 计算重建的spot embedding
            recon_embedding = torch.matmul(attention_weights, celltype_embedding)  # [n_spots, embedding_dim]
            
            # Cosine相似度
            cos_sim_align = F.cosine_similarity(spot_embedding, recon_embedding, dim=-1)
            L_align = 1.0 - cos_sim_align.mean()
        
        # ============ 4. Weight Regularization Loss ============
        # 确保权重和为1（softmax已保证，但作为额外约束）
        weight_sum_loss = F.mse_loss(attention_weights.sum(dim=1), 
                                     torch.ones(n_spots, device=attention_weights.device))
        
        # ============ 5. Sparsity Regularization Loss ============
        # 鼓励稀疏的注意力分布
        sparsity_loss = -torch.mean(attention_weights * torch.log(attention_weights + 1e-8))
        
        # ============ 总损失 ============
        total_loss = (self.lambda_pearson * L_pearson +
                     self.lambda_cosine * L_cosine +
                     self.lambda_align * L_align +
                     self.lambda_reg * weight_sum_loss +
                     self.lambda_sparse * sparsity_loss)
        
        return {
            'total_loss': total_loss,
            'pearson_loss': L_pearson,
            'cosine_loss': L_cosine,
            'alignment_loss': L_align,
            'weight_reg': weight_sum_loss,
            'sparsity_loss': sparsity_loss,
            'pearson_corr': pearson_corr.mean(),
            'cos_sim_rec': cos_sim_rec.mean(),
        }