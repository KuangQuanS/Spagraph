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
    """VAE编码器（多模态友好，使用LayerNorm）"""
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm for multi-modal data
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
    """VAE解码器（多模态友好，使用LayerNorm）"""
    def __init__(self, latent_dim, hidden_dims=[256, 512], output_dim=None, dropout=0.2, output_type='mse'):
        super().__init__()
        
        self.output_type = output_type  # 'mse' or 'zinb'
        
        layers = []
        prev_dim = latent_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm for multi-modal data
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_decoder = nn.Sequential(*layers)
        
        if output_type == 'zinb':
            # ZINB需要三个输出：mean, dispersion, dropout probability
            self.mean_decoder = nn.Linear(prev_dim, output_dim)
            self.disp_decoder = nn.Linear(prev_dim, output_dim)
            self.pi_decoder = nn.Linear(prev_dim, output_dim)
        else:
            # MSE只需要一个输出
            self.output_decoder = nn.Linear(prev_dim, output_dim)
        
    def forward(self, z):
        h = self.shared_decoder(z)
        
        if self.output_type == 'zinb':
            mean = self.mean_decoder(h)
            disp = self.disp_decoder(h)
            pi = self.pi_decoder(h)
            return mean, disp, pi
        else:
            return self.output_decoder(h)

class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.2, output_type='mse'):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_type = output_type
        
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
            dropout=dropout,
            output_type=output_type
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
        decoder_output = self.decoder(z)
        
        if self.output_type == 'zinb':
            # ZINB模式：返回 mean, disp, pi
            mean, disp, pi = decoder_output
            return mean, disp, pi, mu, log_var, z
        else:
            # MSE模式：返回 x_recon
            return decoder_output, mu, log_var, z
    
    def encode(self, x):
        """仅编码，返回潜在表示"""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

# Loss Functions
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """VAE损失函数：重建损失 (MSE) + KL散度"""
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div

def zinb_loss_function(mean, disp, pi, x, mu, log_var, beta=1.0, eps=1e-8, ridge_lambda=0.0):
    """
    VAE损失函数：ZINB重建损失 + KL散度
    
    Args:
        mean: 预测的均值 (batch_size, n_genes)
        disp: 预测的离散度 (batch_size, n_genes) 
        pi: 预测的dropout概率 (batch_size, n_genes)
        x: 真实count数据 (batch_size, n_genes)
        mu: VAE编码器的均值 (batch_size, latent_dim)
        log_var: VAE编码器的log方差 (batch_size, latent_dim)
        beta: KL散度权重
        eps: 数值稳定性常数
        ridge_lambda: L2正则化参数
    
    Returns:
        total_loss: 总损失
        recon_loss: ZINB重建损失
        kl_div: KL散度
    """
    # ZINB负对数似然
    # 参考: https://github.com/gokceneraslan/neuralnet_countmodels
    
    # softplus确保参数为正
    mean = F.softplus(mean) + eps
    disp = F.softplus(disp) + eps
    pi = torch.sigmoid(pi)
    
    # Negative Binomial部分
    # NB(x; mu, theta) where theta is dispersion
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_log_likelihood = t1 + t2
    
    # Zero-inflation部分
    # P(x=0) = pi + (1-pi)*NB(0)
    nb_zero = -disp * torch.log(1.0 + mean / disp + eps)
    zero_nb = torch.log(pi + (1.0 - pi) * torch.exp(nb_zero) + eps)
    non_zero_nb = torch.log(1.0 - pi + eps) - nb_log_likelihood
    
    # 根据是否为0选择相应的似然
    zinb_log_likelihood = torch.where(
        x < 1e-8,
        zero_nb,
        non_zero_nb
    )
    
    # 重建损失 (负对数似然)
    recon_loss = -torch.sum(zinb_log_likelihood)
    
    # KL散度 (与标准正态分布)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Ridge正则化 (可选)
    ridge_loss = ridge_lambda * torch.sum(mean)
    
    # 总损失
    total_loss = recon_loss + beta * kl_div + ridge_loss
    
    return total_loss, recon_loss, kl_div

def compute_mmd(x, y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    
    Args:
        x: Tensor of shape (n_samples_x, n_features) - SC embeddings
        y: Tensor of shape (n_samples_y, n_features) - ST embeddings
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth (if None, use median heuristic)
    
    Returns:
        mmd_loss: Scalar MMD loss value
    """
    # Ensure tensors are on the same device
    device = x.device
    y = y.to(device)
    
    n_x = x.size(0)
    n_y = y.size(0)
    
    if kernel == 'rbf':
        # Compute pairwise distances
        def compute_kernel(a, b, gamma):
            """RBF kernel K(a,b) = exp(-gamma * ||a-b||^2)"""
            # a: (n, d), b: (m, d)
            # output: (n, m)
            a_norm = (a ** 2).sum(1).view(-1, 1)  # (n, 1)
            b_norm = (b ** 2).sum(1).view(1, -1)  # (1, m)
            dist = a_norm + b_norm - 2.0 * torch.mm(a, b.transpose(0, 1))  # (n, m)
            return torch.exp(-gamma * dist)
        
        # Use median heuristic for gamma if not specified
        if gamma is None:
            # Sample a subset for efficiency
            n_samples = min(1000, n_x, n_y)
            x_sample = x[:n_samples]
            y_sample = y[:n_samples]
            
            # Compute pairwise distances
            xy = torch.cat([x_sample, y_sample], dim=0)
            dists = torch.cdist(xy, xy, p=2)
            
            # Median heuristic: gamma = 1 / (2 * median^2)
            median_dist = torch.median(dists[dists > 0])
            gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)
        
        # Compute kernels
        K_xx = compute_kernel(x, x, gamma)
        K_yy = compute_kernel(y, y, gamma)
        K_xy = compute_kernel(x, y, gamma)
        
        # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
        mmd_loss = K_xx.sum() / (n_x * n_x) + K_yy.sum() / (n_y * n_y) - 2 * K_xy.sum() / (n_x * n_y)
        
    elif kernel == 'linear':
        # Linear kernel: K(a,b) = <a,b>
        mean_x = x.mean(0)
        mean_y = y.mean(0)
        mmd_loss = torch.sum((mean_x - mean_y) ** 2)
    
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    return mmd_loss

# ================================
# Stage 2: GAT Models
# ================================

class HeterogeneousGATDeconvolution(nn.Module):
    """异构图注意力网络解卷积模型"""
    def __init__(self, embedding_dim=128,n_cell_types=9,gat_hidden_dim=64,gat_layers=3,gat_heads=4,
                 dropout=0.1,k_spatial=6,k_celltype=10, celltype_prototypes=None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_cell_types = n_cell_types
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        self.k_spatial = k_spatial
        self.k_celltype = k_celltype  # 每个spot连接最近的k个celltype
        
        # 1. 节点嵌入层
        # Spot节点：从VAE embedding到GAT输入
        self.spot_projection = nn.Sequential(
            nn.Linear(embedding_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CellType节点：使用第一阶段VAE的celltype prototypes初始化
        if celltype_prototypes is not None:
            # 从VAE embedding投影到GAT hidden dim
            # 确保投影层和prototypes在同一设备上
            device = celltype_prototypes.device
            celltype_proj = nn.Linear(embedding_dim, gat_hidden_dim, bias=False).to(device)
            with torch.no_grad():
                projected_prototypes = celltype_proj(celltype_prototypes)
            self.celltype_embeddings = nn.Parameter(projected_prototypes)
        else:
            # 如果没有提供prototypes，则随机初始化（用于向后兼容）
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
        """构建异构图：Spot节点 + CellType节点
        
        注意：celltype_prototypes参数用于计算Spot-CellType边的相似度（在原始VAE embedding空间）
              而self.celltype_embeddings用于GAT的节点特征（已投影到gat_hidden_dim空间）
        """
        n_spots = spot_embeddings.shape[0]
        n_cell_types = celltype_prototypes.shape[0]
        device = spot_embeddings.device
        
        # 1. 处理节点特征
        # Spot节点特征：投影到GAT hidden dim
        spot_features = self.spot_projection(spot_embeddings)  # [n_spots, gat_hidden_dim]
        
        # CellType节点特征：使用已初始化的可学习embedding
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
        
        # 2.2 Spot-CellType边（基于KNN，每个spot连接k个最近的celltype）
        # 在原始VAE embedding空间计算相似度（更准确地反映语义相似性）
        spot_emb_np = spot_embeddings.detach().cpu().numpy()
        celltype_emb_np = celltype_prototypes.detach().cpu().numpy()
        
        similarity_matrix = cosine_similarity(spot_emb_np, celltype_emb_np)  # [n_spots, n_cell_types]
        
        # 对于每个spot，找到最相似的k个celltype
        k_neighbors = min(self.k_celltype, n_cell_types)  # 防止k超过celltype数量
        
        for spot_idx in range(n_spots):
            # 获取该spot与所有celltype的相似度
            similarities = similarity_matrix[spot_idx]
            
            # 找到最大的k个相似度的索引
            top_k_indices = np.argsort(-similarities)[:k_neighbors]
            
            for celltype_idx in top_k_indices:
                similarity = similarities[celltype_idx]
                
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
        """Forward pass"""
        n_spots = spot_embeddings.shape[0]
        n_cell_types = celltype_prototypes.shape[0]
        device = spot_embeddings.device
        
        # 1. Build heterogeneous graph
        edge_index, edge_attr, node_features = self.build_heterogeneous_graph(
            spot_embeddings, spatial_coords, celltype_prototypes
        )
        
        # 2. Create sparse mask: which spot-celltype pairs are connected in the graph
        # This is used to enforce k_celltype sparsity
        sparse_mask = torch.zeros(n_spots, n_cell_types, dtype=torch.bool, device=device)
        
        # Extract spot-celltype edges from edge_index
        n_nodes = n_spots + n_cell_types
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Spot -> CellType edge
            if src < n_spots and dst >= n_spots:
                celltype_idx = dst - n_spots
                sparse_mask[src, celltype_idx] = True
        
        # 3. GAT processing
        x = node_features
        for i, gat_layer in enumerate(self.gat_layers_list):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers_list) - 1:
                x = F.relu(x)
        
        # 4. Extract spot and celltype node features
        spot_features = x[:n_spots]  # [n_spots, gat_hidden_dim]
        celltype_features = x[n_spots:]  # [n_cell_types, gat_hidden_dim]
        
        # 5. Vectorized attention weight computation
        # Expand dimensions for batch computation
        spot_expanded = spot_features.unsqueeze(1).expand(-1, n_cell_types, -1)
        cell_expanded = celltype_features.unsqueeze(0).expand(n_spots, -1, -1)
        
        # Concatenate features
        combined = torch.cat([spot_expanded, cell_expanded], dim=-1)
        
        # Compute attention scores
        attention_scores = self.attention_mlp(combined).squeeze(-1)  # [n_spots, n_cell_types]
        
        # Apply sparse mask: set non-connected celltypes to -inf
        # After softmax, -inf becomes 0
        attention_scores_masked = attention_scores.clone()
        attention_scores_masked[~sparse_mask] = float('-inf')
        
        # Softmax normalization (only over connected celltypes)
        deconv_weights = F.softmax(attention_scores_masked, dim=1)
        
        # Handle any NaN from all -inf rows (shouldn't happen with k_celltype)
        deconv_weights = torch.nan_to_num(deconv_weights, 0.0)
        
        return {
            'spot_features': spot_features,
            'celltype_features': celltype_features,
            'attention_scores': attention_scores,    # Raw scores
            'attention_scores_masked': attention_scores_masked,  # Masked scores
            'deconv_weights': deconv_weights,        # Normalized weights (sparse)
            'sparse_mask': sparse_mask,              # For debugging
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }


# ================================
# Stage 2: Loss Functions
# ================================

class SpatialDeconvolutionLoss(nn.Module):
    """空间解卷积损失函数
    
    L_total = λ_pearson·L_pearson + λ_mse·L_mse + λ_cosine·L_cosine + λ_reg·L_reg + λ_sparse·L_sparse + λ_diversity·L_diversity + λ_hetero·L_hetero + λ_proportion·L_proportion
    
    其中：
    - L_pearson: Pearson相关系数损失(基因表达相关性)
    - L_mse: 均方误差损失(基因表达重建误差)
    - L_cosine: Cosine相似度损失(基因表达相似性)
    - L_reg: 权重正则化
    - L_sparse: 稀疏性正则化
    - L_diversity: 多样性损失(防止所有spot都使用相同的细胞类型)
    - L_hetero: 空间异质性损失(鼓励相邻spot的细胞组成有差异)
    - L_proportion: 全局细胞类型比例一致性损失(与单细胞数据的cluster比例一致)
    """
    
    def __init__(self, lambda_pearson=1.0, lambda_mse=1.0, lambda_cosine=1.0, 
                 lambda_reg=0.5, lambda_sparse=0.01, lambda_diversity=0.1, lambda_hetero=0.05,
                 lambda_proportion=1.0, sc_celltype_proportions=None):
        super().__init__()
        self.lambda_pearson = lambda_pearson      # Pearson损失权重
        self.lambda_mse = lambda_mse              # MSE损失权重
        self.lambda_cosine = lambda_cosine        # Cosine损失权重
        self.lambda_reg = lambda_reg              # 权重正则化权重
        self.lambda_sparse = lambda_sparse        # 稀疏性正则化权重
        self.lambda_diversity = lambda_diversity  # 多样性损失权重
        self.lambda_hetero = lambda_hetero        # 空间异质性损失权重
        self.lambda_proportion = lambda_proportion  # 细胞类型比例一致性权重
        
        # 单细胞数据中各cluster的比例 [n_cell_types]
        # 例如: [0.10, 0.15, 0.20, ...] 表示cluster0占10%, cluster1占15%等
        if sc_celltype_proportions is not None:
            self.register_buffer('sc_celltype_proportions', 
                               torch.FloatTensor(sc_celltype_proportions))
        else:
            self.sc_celltype_proportions = None
        
    def forward(self, 
               attention_weights: torch.Tensor,
               celltype_expression: torch.Tensor,
               true_spot_expression: torch.Tensor,
               spot_embedding: torch.Tensor = None,
               celltype_embedding: torch.Tensor = None,
               edge_index: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            attention_weights: 注意力权重 [n_spots, n_cell_types]（已softmax）
            celltype_expression: 细胞类型表达 [n_cell_types, n_genes]
            true_spot_expression: 真实spot表达 [n_spots, n_genes]
            spot_embedding: spot embedding [n_spots, embedding_dim]（可选）
            celltype_embedding: 细胞类型embedding [n_cell_types, embedding_dim]（可选）
            edge_index: 图的边索引 [2, num_edges]（可选，用于计算空间异质性）
            
        Returns:
            损失字典
        """
        n_spots = attention_weights.shape[0]
        
        # 计算重建的spot表达
        reconstructed_spot = torch.matmul(attention_weights, celltype_expression)  # [n_spots, n_genes]
        
        # ============ 1. Pearson Correlation Loss ============
        # 计算Pearson相关系数(基于基因表达的相关性)
        def pearson_correlation(x, y):
            """计算Pearson相关系数"""
            x_centered = x - x.mean(dim=-1, keepdim=True)
            y_centered = y - y.mean(dim=-1, keepdim=True)
            
            numerator = (x_centered * y_centered).sum(dim=-1)
            denominator = torch.sqrt((x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1))
            
            corr = numerator / (denominator + 1e-8)
            return corr
        
        pearson_corr = pearson_correlation(reconstructed_spot, true_spot_expression)
        L_pearson = 1.0 - pearson_corr.mean()  # 1 - 相关系数,越小越好
        
        # ============ 2. MSE Loss (重建误差) ============
        L_mse = F.mse_loss(reconstructed_spot, true_spot_expression)
        
        # ============ 3. Cosine Similarity Loss ============
        # Cosine相似度(越接近1越好)
        cos_sim_rec = F.cosine_similarity(reconstructed_spot, true_spot_expression, dim=-1)
        L_cosine = 1.0 - cos_sim_rec.mean()
        
        # ============ 4. Weight Regularization Loss ============
        # 确保权重和为1(softmax已保证,但作为额外约束)
        weight_sum_loss = F.mse_loss(attention_weights.sum(dim=1), 
                                     torch.ones(n_spots, device=attention_weights.device))
        
        # ============ 5. Sparsity Regularization Loss ============
        # 鼓励稀疏的注意力分布（每个spot只使用少数细胞类型）
        sparsity_loss = -torch.mean(attention_weights * torch.log(attention_weights + 1e-8))
        
        # ============ 6. Diversity Loss ============
        # 防止所有spot都使用相同的细胞类型组合
        # 计算全局细胞类型使用频率（所有spot的平均权重）
        global_celltype_usage = attention_weights.mean(dim=0)  # [n_cell_types]
        
        # 目标：每个细胞类型的全局使用率应该尽可能均匀
        # 使用KL散度，鼓励全局使用分布接近均匀分布
        n_cell_types = attention_weights.shape[1]
        uniform_dist = torch.ones_like(global_celltype_usage) / n_cell_types
        
        # KL(global_usage || uniform) = sum(p * log(p/q))
        diversity_loss = torch.sum(
            global_celltype_usage * torch.log(
                (global_celltype_usage + 1e-8) / (uniform_dist + 1e-8)
            )
        )
        
        # ============ 7. Spatial Heterogeneity Loss ============
        # 鼓励相邻spot的细胞组成有差异，保持空间异质性
        hetero_loss = torch.tensor(0.0, device=attention_weights.device)
        
        if edge_index is not None and edge_index.shape[1] > 0:
            # 只考虑spot-spot边（节点索引 < n_spots）
            spot_spot_mask = (edge_index[0] < n_spots) & (edge_index[1] < n_spots)
            spot_edges = edge_index[:, spot_spot_mask]
            
            if spot_edges.shape[1] > 0:
                # 获取边的源节点和目标节点
                src_nodes = spot_edges[0]  # [num_edges]
                dst_nodes = spot_edges[1]  # [num_edges]
                
                # 获取相邻spot的权重
                src_weights = attention_weights[src_nodes]  # [num_edges, n_cell_types]
                dst_weights = attention_weights[dst_nodes]  # [num_edges, n_cell_types]
                
                # 计算相邻spot权重的相似度（使用cosine相似度）
                # 我们希望这个相似度不要太高（保持差异）
                similarity = F.cosine_similarity(src_weights, dst_weights, dim=-1)  # [num_edges]
                
                # 异质性损失：惩罚过高的相似度
                # 使用 ReLU 确保只惩罚相似度 > threshold 的情况
                similarity_threshold = 0.7  # 相邻spot的相似度不应超过0.7
                hetero_loss = F.relu(similarity - similarity_threshold).mean()
        
        # ============ 8. Global Cell Type Proportion Consistency Loss ============
        # 确保解卷积结果的**全局**细胞类型比例与单细胞数据一致
        # 
        # 重要理解：
        #   - 这是对**整个组织切片**的全局约束，不是对单个spot的约束
        #   - 单个spot可以有不同的细胞组成（保持空间异质性）
        #   - 但所有spots加起来的总体分布应该与单细胞数据匹配
        # 
        # 例子：
        #   - 单细胞数据: T细胞30%, B细胞20%, 上皮细胞50%
        #   - Spot A (肿瘤核心): T细胞5%, B细胞5%, 上皮细胞90%  ✓ 允许
        #   - Spot B (免疫区域): T细胞60%, B细胞30%, 上皮细胞10% ✓ 允许
        #   - 全局平均: T细胞30%, B细胞20%, 上皮细胞50%         ✓ 匹配单细胞
        proportion_loss = torch.tensor(0.0, device=attention_weights.device)
        
        if self.sc_celltype_proportions is not None:
            # 计算ST数据中预测的全局细胞类型比例
            # attention_weights: [n_spots, n_cell_types]
            # 全局比例 = 所有spot的平均权重（这是关键：对整个batch求平均）
            st_predicted_proportions = attention_weights.mean(dim=0)  # [n_cell_types]
            
            # 单细胞数据的真实比例（从聚类统计得到）
            sc_proportions = self.sc_celltype_proportions.to(attention_weights.device)  # [n_cell_types]
            
            # 方法1: KL散度 - 衡量两个分布的差异
            # KL(ST || SC) = sum(p_st * log(p_st / p_sc))
            kl_loss = torch.sum(
                st_predicted_proportions * torch.log(
                    (st_predicted_proportions + 1e-8) / (sc_proportions + 1e-8)
                )
            )
            
            # 方法2: L2距离 - 直接衡量比例差异
            l2_loss = torch.sum((st_predicted_proportions - sc_proportions) ** 2)
            
            # 方法3: L1距离 - Total Variation Distance
            l1_loss = torch.sum(torch.abs(st_predicted_proportions - sc_proportions))
            
            # 使用KL散度作为主要损失（也可以组合使用）
            proportion_loss = kl_loss
            # 或者使用组合: proportion_loss = 0.5 * kl_loss + 0.5 * l2_loss
        
        # ============ 总损失 ============
        total_loss = (self.lambda_pearson * L_pearson +
                     self.lambda_mse * L_mse +
                     self.lambda_cosine * L_cosine +
                     self.lambda_reg * weight_sum_loss +
                     self.lambda_sparse * sparsity_loss +
                     self.lambda_diversity * diversity_loss +
                     self.lambda_hetero * hetero_loss +
                     self.lambda_proportion * proportion_loss)
        
        return {
            'total_loss': total_loss,
            'pearson_loss': L_pearson,
            'mse_loss': L_mse,
            'cosine_loss': L_cosine,
            'weight_reg': weight_sum_loss,
            'sparsity_loss': sparsity_loss,
            'diversity_loss': diversity_loss,
            'hetero_loss': hetero_loss,
            'proportion_loss': proportion_loss
        }