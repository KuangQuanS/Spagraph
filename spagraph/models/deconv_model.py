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


class DualDecoderVAE(nn.Module):
    """
    双解码器VAE - 用于SC和ST模态对齐
    
    架构:
        - 共享Encoder: 将SC和ST数据编码到同一个latent space
        - SC Decoder: 专门重建SC数据
        - ST Decoder: 专门重建ST数据
        - MMD Loss: 在latent space上对齐两个模态的分布
    
    前向传播:
        SC: x_sc → Encoder → z_sc → Decoder_SC → recon_sc
        ST: x_st → Encoder → z_st → Decoder_ST → recon_st
        Alignment: MMD(z_sc, z_st) → 强制两个模态的latent分布对齐
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128, dropout=0.2, output_type='mse'):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_type = output_type
        
        # 共享编码器 (Shared Encoder)
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # SC专用解码器 (SC-specific Decoder)
        decoder_hidden = hidden_dims[::-1]
        self.decoder_sc = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim,
            dropout=dropout,
            output_type=output_type
        )
        
        # ST专用解码器 (ST-specific Decoder)
        self.decoder_st = VAEDecoder(
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
    
    def forward(self, x, modality):
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            modality: 模态标签 [batch_size] (0=SC, 1=ST)
        
        Returns:
            根据output_type返回不同格式:
            - MSE模式: recon_x, mu, log_var, z
            - ZINB模式: mean, disp, pi, mu, log_var, z
        """
        # 共享编码 (所有数据都通过同一个encoder)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # 根据modality选择不同的decoder
        # modality: 0=SC, 1=ST
        sc_mask = (modality == 0)
        st_mask = (modality == 1)
        
        if self.output_type == 'zinb':
            # 初始化输出张量
            batch_size = x.shape[0]
            input_dim = x.shape[1]
            device = x.device
            
            mean = torch.zeros(batch_size, input_dim, device=device)
            disp = torch.zeros(batch_size, input_dim, device=device)
            pi = torch.zeros(batch_size, input_dim, device=device)
            
            # SC数据通过decoder_sc
            if sc_mask.sum() > 0:
                mean_sc, disp_sc, pi_sc = self.decoder_sc(z[sc_mask])
                mean[sc_mask] = mean_sc
                disp[sc_mask] = disp_sc
                pi[sc_mask] = pi_sc
            
            # ST数据通过decoder_st
            if st_mask.sum() > 0:
                mean_st, disp_st, pi_st = self.decoder_st(z[st_mask])
                mean[st_mask] = mean_st
                disp[st_mask] = disp_st
                pi[st_mask] = pi_st
            
            return mean, disp, pi, mu, log_var, z
        else:
            # MSE模式
            batch_size = x.shape[0]
            input_dim = x.shape[1]
            device = x.device
            
            recon_x = torch.zeros(batch_size, input_dim, device=device)
            
            # SC数据通过decoder_sc
            if sc_mask.sum() > 0:
                recon_x[sc_mask] = self.decoder_sc(z[sc_mask])
            
            # ST数据通过decoder_st
            if st_mask.sum() > 0:
                recon_x[st_mask] = self.decoder_st(z[st_mask])
            
            return recon_x, mu, log_var, z
    
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
    """异构图注意力网络解卷积模型
    
    支持两种cluster表示模式：
    1. 静态模式（默认）: 使用cluster平均表达
    2. 动态模式: 每个spot使用该spot最近的k个细胞的加权表达
    """
    def __init__(self, embedding_dim=128,n_cell_types=9,gat_hidden_dim=64,gat_layers=3,gat_heads=4,
                 dropout=0.1,k_spatial=6,k_celltype=10, celltype_prototypes=None,
                 use_dynamic_cluster_repr=False, k_cells_per_cluster=10,
                 sc_cell_expressions=None,
                 normalize_attention=True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_cell_types = n_cell_types
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        self.k_spatial = k_spatial
        self.k_celltype = k_celltype  # 每个spot连接最近的k个celltype
        
        # 动态cluster表示参数
        self.use_dynamic_cluster_repr = use_dynamic_cluster_repr
        self.k_cells_per_cluster = k_cells_per_cluster
        self.normalize_attention = normalize_attention
        
        # 如果启用动态模式，保存单细胞全基因表达（原始count）
        # k-nearest cells索引应该在第一阶段预计算好
        if use_dynamic_cluster_repr:
            if sc_cell_expressions is None:
                raise ValueError(
                    "Dynamic cluster representation requires sc_cell_expressions (raw counts of all genes).\n"
                    "k-nearest cell indices should be pre-computed in stage1."
                )
            # sc_cell_expressions: [n_cells, n_all_genes] 全基因原始count
            self.register_buffer('sc_cell_expressions', torch.FloatTensor(sc_cell_expressions))
        
        # 1. 节点特征投影层
        # Spot节点：从VAE embedding到GAT输入
        self.spot_projection = nn.Sequential(
            nn.Linear(embedding_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CellType节点：使用第一阶段VAE的celltype prototypes初始化
        if celltype_prototypes is not None:
            # 从VAE embedding投影到GAT hidden dim
            device = celltype_prototypes.device
            celltype_proj = nn.Linear(embedding_dim, gat_hidden_dim, bias=False).to(device)
            with torch.no_grad():
                projected_prototypes = celltype_proj(celltype_prototypes)
            # 初始化为prototypes，但允许训练（通过残差连接保持联系）
            self.celltype_embeddings = nn.Parameter(projected_prototypes)
        else:
            # 如果没有提供prototypes，则随机初始化
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
                                celltype_prototypes: torch.Tensor,
                                use_embedding_knn: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        # 2.1 Spot-Spot边（空间KNN或嵌入KNN）
        if n_spots > 1:
            if not use_embedding_knn and spatial_coords is not None and len(spatial_coords) > 1:
                coords_np = spatial_coords.detach().cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=min(self.k_spatial+1, len(coords_np))).fit(coords_np)
                distances, indices = nbrs.kneighbors(coords_np)
                
                for i in range(len(indices)):
                    for j in range(1, len(indices[i])):  # 跳过自己（第0个）
                        neighbor_idx = indices[i][j]
                        distance = distances[i][j]
                        
                        # 转换为相似度权重
                        weight = np.exp(-distance / (np.std(distances) + 1e-8))
                        
                        # 双向边
                        edge_indices.append([i, neighbor_idx])
                        edge_attrs.append([weight])
                        edge_indices.append([neighbor_idx, i])
                        edge_attrs.append([weight])
            else:
                # 基于spot embedding的KNN（使用余弦相似度）
                spot_np = spot_embeddings.detach().cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=min(self.k_spatial+1, len(spot_np)), metric='cosine').fit(spot_np)
                distances, indices = nbrs.kneighbors(spot_np)
                # cosine metric gives distance = 1 - cosine_sim
                for i in range(len(indices)):
                    for j in range(1, len(indices[i])):
                        neighbor_idx = indices[i][j]
                        distance = distances[i][j]
                        weight = np.exp(-(distance) / (np.std(distances) + 1e-8))
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
               celltype_prototypes: torch.Tensor,
               use_embedding_knn: bool = False,
               normalize_attention: bool = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        n_spots = spot_embeddings.shape[0]
        n_cell_types = celltype_prototypes.shape[0]
        device = spot_embeddings.device
        
        # Handle normalize_attention parameter
        if normalize_attention is None:
            normalize_attention = self.normalize_attention
        
        # 1. Build heterogeneous graph
        edge_index, edge_attr, node_features = self.build_heterogeneous_graph(
            spot_embeddings, spatial_coords, celltype_prototypes, use_embedding_knn=use_embedding_knn
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
        
        # 3. GAT processing with residual connections
        # 保存原始节点特征用于残差连接
        original_features = node_features.clone()
        
        x = node_features
        for i, gat_layer in enumerate(self.gat_layers_list):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers_list) - 1:
                x = F.relu(x)
        
        # 添加残差连接：保持与原始VAE embeddings的联系
        # 这样既能利用GAT的空间聚合，又不会完全偏离原始语义
        alpha = 0.5  # 残差权重，可以调整
        x = alpha * x + (1 - alpha) * original_features
        
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
        
        # Attention weight normalization (optional)
        if normalize_attention:
            # Softmax normalization (only over connected celltypes)
            deconv_weights = F.softmax(attention_scores_masked, dim=1)
        else:
            # No normalization: use raw scores, set non-connected to 0
            deconv_weights = attention_scores_masked.clone()
            deconv_weights[~sparse_mask] = 0.0
        
        # Handle any NaN from all -inf rows (shouldn't happen with k_celltype)
        deconv_weights = torch.nan_to_num(deconv_weights, 0.0)
        
        # ========== 动态Cluster表示（可选） ==========
        # k-nearest cell indices已在第一阶段预计算，这里不再计算
        # 直接返回None，实际的动态权重计算在Loss中完成
        dynamic_cluster_weights = None
        dynamic_cluster_indices = None
        
        return {
            'spot_features': spot_features,
            'celltype_features': celltype_features,
            'attention_scores': attention_scores,    # Raw scores
            'attention_scores_masked': attention_scores_masked,  # Masked scores
            'deconv_weights': deconv_weights,        # Normalized weights (sparse)
            'sparse_mask': sparse_mask,              # For debugging
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'dynamic_cluster_weights': dynamic_cluster_weights,  # [n_spots, n_cell_types, k] or None
            'dynamic_cluster_indices': dynamic_cluster_indices   # [n_spots, n_cell_types, k] or None
        }
    
    def compute_cell_weights_from_knn(self, 
                                     spot_embeddings: torch.Tensor,
                                     knn_cell_indices: torch.Tensor,
                                     sc_cell_embeddings: torch.Tensor) -> torch.Tensor:
        """根据预计算的k-nearest cells索引，计算cell权重（向量化优化版本）
        
        两种模式：
        1. 可学习模式（use_learnable_weights=True）：用MLP计算context-dependent权重
        2. 均匀平均模式（use_learnable_weights=False）：每个cell权重=1/k（简单平均）
        
        关键逻辑：
        1. 使用第一阶段预计算好的k-nearest cell索引
        2. 获取这些细胞的embeddings（仅MLP模式需要）
        3. 用MLP计算权重 或 直接返回均匀权重
        4. Softmax归一化，确保每个cluster的k个cell权重和为1（百分比）
        
        优化：一次性gather所有cell embeddings，batch计算MLP
        
        Args:
            spot_embeddings: [batch_size, embedding_dim] 当前batch的spot embeddings
            knn_cell_indices: [batch_size, n_cell_types, k] 预计算的k-nearest cell索引
            sc_cell_embeddings: [n_cells, embedding_dim] 所有单细胞的embeddings
        
        Returns:
            cell_weights: [batch_size, n_cell_types, k] 归一化的cell权重（百分比矩阵）
        
        Note:
            - knn_cell_indices中-1表示padding（cluster细胞数<k），对应权重会被置0
            - 每个cluster的k个权重softmax归一化，和为1
        """
        batch_size, n_cell_types, k = knn_cell_indices.shape
        embedding_dim = spot_embeddings.shape[1]
        device = spot_embeddings.device
        
        # ✅ 向量化优化：创建mask并处理padding
        valid_mask = (knn_cell_indices >= 0)  # [batch_size, n_cell_types, k]
        
        # ✅ 均匀权重模式：直接返回1/k（基于有效cell数量）
        # 计算每个cluster的有效cell数量
        num_valid_cells = valid_mask.sum(dim=-1, keepdim=True).float()  # [batch_size, n_cell_types, 1]
        num_valid_cells = torch.clamp(num_valid_cells, min=1.0)  # 避免除零
        
        # 均匀权重：有效cell权重=1/num_valid，padding cell权重=0
        uniform_weights = valid_mask.float() / num_valid_cells  # [batch_size, n_cell_types, k]
        return uniform_weights


# ================================
# Stage 2: Loss Functions
# ================================

class SpatialDeconvolutionLoss(nn.Module):
    """空间解卷积损失函数
    
    L_total = λ_pearson·L_pearson + λ_mse·L_mse + λ_cosine·L_cosine + λ_reg·L_reg + λ_sparse·L_sparse + λ_proportion·L_proportion + λ_gene_pearson·L_gene_pearson + λ_gene_cosine·L_gene_cosine
    
    其中：
    - L_pearson: Pearson相关系数损失(基因表达相关性)
    - L_mse: 均方误差损失(基因表达重建误差)
    - L_cosine: Cosine相似度损失(基因表达相似性)
    - L_reg: 权重正则化
    - L_sparse: 稀疏性正则化
    - L_proportion: 全局细胞类型比例一致性损失(与单细胞数据的cluster比例一致)
    
    注：已移除 L_diversity 和 L_hetero
    """
    
    def __init__(self, lambda_pearson=1.0, lambda_mse=1.0, lambda_cosine=1.0, 
                 lambda_gene_pearson=0.0, lambda_gene_cosine=0.0,
                 lambda_reg=0.5, lambda_sparse=0.01,
                 lambda_proportion=1.0, sc_celltype_proportions=None, spot_total_counts=None,
                 celltype_expressions_full=None, marker_gene_indices=None,
                 hvg_gene_indices=None, scale_basis: str = "hvg"):
        super().__init__()
        self.lambda_pearson = lambda_pearson      # Pearson损失权重
        self.lambda_mse = lambda_mse              # MSE损失权重
        self.lambda_cosine = lambda_cosine        # Cosine损失权重
        self.lambda_gene_pearson = lambda_gene_pearson  # 基因维度Pearson权重
        self.lambda_gene_cosine = lambda_gene_cosine    # 基因维度Cosine权重
        self.lambda_reg = lambda_reg              # 权重正则化权重
        self.lambda_sparse = lambda_sparse        # 稀疏性正则化权重
        self.lambda_proportion = lambda_proportion  # 细胞类型比例一致性权重
        self.scale_basis = scale_basis            # 用于缩放的基因集合: marker / hvg / all / none / fixed_10
        
        # 当 scale_basis='none' 或 'fixed_10' 时允许为 None（不需要spot_total_counts）
        if spot_total_counts is None and scale_basis not in ['none', 'fixed_10']:
            raise ValueError("spot_total_counts is required for reconstruction when scale_basis not in ['none', 'fixed_10']!")
        if spot_total_counts is not None:
            self.register_buffer('spot_total_counts', torch.FloatTensor(spot_total_counts))
        else:
            self.spot_total_counts = None
        
        # 用于重建完整的 spot 表达谱，然后只在 marker 基因上计算 loss
        if celltype_expressions_full is None:
            raise ValueError("celltype_expressions_full is required for full gene reconstruction!")
        self.register_buffer('celltype_expressions_full', torch.FloatTensor(celltype_expressions_full))
        
        # 用于从重建的全部基因表达中提取 marker 基因
        if marker_gene_indices is None:
            raise ValueError("marker_gene_indices is required to extract marker genes!")
        self.register_buffer('marker_gene_indices', torch.LongTensor(marker_gene_indices))
        
        if hvg_gene_indices is not None and len(hvg_gene_indices) > 0:
            self.register_buffer('hvg_gene_indices', torch.LongTensor(hvg_gene_indices))
        else:
            self.hvg_gene_indices = None
        
        # 单细胞数据中各cluster的比例 [n_cell_types]
        # 例如: [0.10, 0.15, 0.20, ...] 表示cluster0占10%, cluster1占15%等
        if sc_celltype_proportions is not None:
            self.register_buffer('sc_celltype_proportions', 
                               torch.FloatTensor(sc_celltype_proportions))
        else:
            self.sc_celltype_proportions = None
    
    def compute_dynamic_mixed_expression(self, 
                                        attention_weights: torch.Tensor,
                                        dynamic_cluster_weights: torch.Tensor,
                                        dynamic_cluster_indices: torch.Tensor,
                                        sc_cell_expressions: torch.Tensor) -> torch.Tensor:
        """使用动态cluster权重计算混合表达（向量化优化版本）
        
        关键逻辑：
        1. 每个cluster的表达 = Σ(cell百分比 × cell原始count表达) 
           - cell百分比来自 dynamic_cluster_weights（已归一化为1）
           - cell表达是原始count（raw counts，不做normalize/log1p）
        2. Spot表达 = Σ(spot对cluster权重 × cluster动态表达)
        
        优化：使用gather和batch矩阵乘法，避免双重循环
        
        Args:
            attention_weights: [n_spots, n_cell_types] spot对cluster的权重
            dynamic_cluster_weights: [n_spots, n_cell_types, k] 每个cluster的k个cell百分比（和为1）
            dynamic_cluster_indices: [n_spots, n_cell_types, k] 每个cluster的k个cell全局索引
            sc_cell_expressions: [n_cells, n_all_genes] 单细胞所有基因原始count
        
        Returns:
            mixed_expr_full: [n_spots, n_all_genes] 混合表达（原始count scale）
        """
        n_spots, n_cell_types, k = dynamic_cluster_weights.shape
        n_genes = sc_cell_expressions.shape[1]
        device = attention_weights.device
        
        # ✅ 内存优化：逐cluster处理，避免一次性分配巨大tensor
        # 原来：[n_spots, n_cell_types, k, n_genes] 太大！
        # 现在：逐个cluster处理，内存占用 = [n_spots, k, n_genes]
        
        cluster_dynamic_expr = torch.zeros(n_spots, n_cell_types, n_genes, device=device)
        
        for cluster_id in range(n_cell_types):
            # 当前cluster的索引和权重
            cluster_indices = dynamic_cluster_indices[:, cluster_id, :]  # [n_spots, k]
            cluster_weights = dynamic_cluster_weights[:, cluster_id, :]  # [n_spots, k]
            
            # 处理padding：将-1替换为0
            safe_indices = torch.clamp(cluster_indices, min=0).long()  # [n_spots, k]
            
            # Gather该cluster的cell表达: [n_spots, k, n_genes]
            cell_expr = sc_cell_expressions[safe_indices.reshape(-1)].reshape(n_spots, k, n_genes)
            
            # 计算加权和: [n_spots, k] × [n_spots, k, n_genes] → [n_spots, n_genes]
            cluster_dynamic_expr[:, cluster_id, :] = torch.einsum('sk,skg->sg', 
                                                                   cluster_weights, 
                                                                   cell_expr)
        
        # ✅ 最终混合：用attention_weights加权
        # attention_weights: [n_spots, n_cell_types]
        # cluster_dynamic_expr: [n_spots, n_cell_types, n_genes]
        # → mixed_expr_full: [n_spots, n_genes]
        mixed_expr_full = torch.einsum('sc,scg->sg', 
                                       attention_weights, 
                                       cluster_dynamic_expr)
        
        return mixed_expr_full
        
    def forward(self, 
               attention_weights: torch.Tensor,
               celltype_expression: torch.Tensor,
               true_spot_expression: torch.Tensor,
               spot_embedding: torch.Tensor = None,
               celltype_embedding: torch.Tensor = None,
               edge_index: torch.Tensor = None,
               batch_spot_total_counts: torch.Tensor = None,
               knn_cell_indices: torch.Tensor = None,
               sc_cell_embeddings: torch.Tensor = None,
               sc_cell_expressions: torch.Tensor = None,
               gat_model: nn.Module = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            attention_weights: 注意力权重 [batch_size, n_cell_types]（已softmax）
            celltype_expression: 细胞类型表达 [n_cell_types, n_marker_genes] (未使用，已废弃)
            true_spot_expression: 真实spot表达 [batch_size, n_marker_genes] (marker genes only)
            spot_embedding: spot embedding [batch_size, embedding_dim]（可选）
            celltype_embedding: 细胞类型embedding [n_cell_types, embedding_dim]（可选）
            edge_index: 图的边索引 [2, num_edges]（可选，用于计算空间异质性）
            batch_spot_total_counts: 当前batch的spot总counts [batch_size]（优先使用）
            
            # 动态cluster模式参数（第一阶段预计算）
            knn_cell_indices: [batch_size, n_cell_types, k] 预计算的k-nearest cell索引（可选）
            sc_cell_embeddings: [n_cells, embedding_dim] 单细胞embeddings（动态模式需要，用于MLP）
            sc_cell_expressions: [n_cells, n_all_genes] 单细胞所有基因原始count（后续按基因名截取）
            gat_model: GAT模型实例（动态模式需要，用于调用cell_weight_mlp）
            
        Returns:
            损失字典
        """
        batch_size = attention_weights.shape[0]

        # ========== 计算混合表达（支持静态和动态模式） ==========
        if knn_cell_indices is not None:
            # ✅ 动态模式：使用预计算的k-nearest cells + MLP学习权重
            if sc_cell_embeddings is None or sc_cell_expressions is None or gat_model is None:
                raise ValueError(
                    "Dynamic cluster mode requires: knn_cell_indices, sc_cell_embeddings, "
                    "sc_cell_expressions, and gat_model"
                )
            
            # 1) 用MLP计算cell权重（归一化为百分比）
            # gat_model.compute_cell_weights_from_knn() 会调用 cell_weight_mlp
            dynamic_cluster_weights = gat_model.compute_cell_weights_from_knn(
                spot_embeddings=spot_embedding,
                knn_cell_indices=knn_cell_indices,
                sc_cell_embeddings=sc_cell_embeddings
            )  # [batch_size, n_cell_types, k]
            
            # 2) 用动态权重计算混合表达
            # sc_cell_expressions 必须是原始count（raw counts，不做normalize/log1p）
            mixed_expr_full = self.compute_dynamic_mixed_expression(
                attention_weights=attention_weights,
                dynamic_cluster_weights=dynamic_cluster_weights, 
                dynamic_cluster_indices=knn_cell_indices,
                sc_cell_expressions=sc_cell_expressions
            )
        else:
            # ✅ 静态模式：使用cluster平均表达
            # celltype_expressions_full 也应该是原始count（raw counts）
            mixed_expr_full = torch.matmul(
                attention_weights,               # [batch_size, n_cell_types]
                self.celltype_expressions_full   # [n_cell_types, n_all_genes] raw counts
            )  # [batch_size, n_all_genes]

        # 3) 根据 scale_basis 决定是否缩放
        if self.scale_basis == "none":
            # 不使用缩放，直接使用混合表达
            reconstructed_spot_full = mixed_expr_full
        elif self.scale_basis == "fixed_10":
            # 固定缩放因子10：比例×10 = 细胞数量
            # 例如：cluster1占比0.3 → 0.3×10=3个细胞
            reconstructed_spot_full = mixed_expr_full * 10.0
        else:
            # 使用 spot_total_counts 进行缩放
            # ✅ 修复：使用传入的 batch_spot_total_counts（对应当前 batch 的实际 spots）
            if batch_spot_total_counts is None:
                raise ValueError("batch_spot_total_counts is required when scale_basis != 'none' or 'fixed_10'")
            spot_counts = batch_spot_total_counts.unsqueeze(-1)  # [batch_size, 1]

            if self.scale_basis == "all":
                # 使用全部基因总量
                mixed_basis_totals = mixed_expr_full.sum(dim=1, keepdim=True)
            elif self.scale_basis == "hvg" and self.hvg_gene_indices is not None:
                # 使用 HVG 交集
                mixed_basis_totals = mixed_expr_full[:, self.hvg_gene_indices].sum(dim=1, keepdim=True)
            else:
                # 默认：使用 marker 子集
                mixed_basis_totals = mixed_expr_full[:, self.marker_gene_indices].sum(dim=1, keepdim=True)

            scale = spot_counts / (mixed_basis_totals + 1e-8)
            reconstructed_spot_full = mixed_expr_full * scale  # [batch_size, n_all_genes]
        
        # 提取 marker 基因: 只在 marker 基因上计算 loss
        reconstructed_spot_marker = reconstructed_spot_full[:, self.marker_gene_indices]  # [batch_size, n_marker_genes]
        
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
        
        # # Log-transform (避免高表达基因主导)
        # reconstructed_log = torch.log1p(reconstructed_spot_marker)
        # true_log = torch.log1p(true_spot_expression)


        # normalize_total 到 1e4 并 log1p
        def normalize_total(mat, target_sum=1e4):
            scale = target_sum / (mat.sum(dim=1, keepdim=True) + 1e-8)
            return mat * scale

        reconstructed_log = torch.log1p(normalize_total(reconstructed_spot_marker, 1e4))
        true_log = torch.log1p(normalize_total(true_spot_expression, 1e4))

        pearson_corr = pearson_correlation(reconstructed_log, true_log)
        L_pearson = 1.0 - pearson_corr.mean()  # 1 - 相关系数,越小越好
        
        # ============ 2. MSE Loss (重建误差) ============
        # 改为在 log1p 空间计算，聚焦表达模式差异，减弱高表达基因主导
        L_mse = F.mse_loss(reconstructed_log, true_log)
        
        # ============ 3. Cosine Similarity Loss ============
        # Cosine相似度在 log-normalized 空间计算（避免高表达基因主导）
        cos_sim_rec = F.cosine_similarity(reconstructed_log, true_log, dim=-1)
        L_cosine = 1.0 - cos_sim_rec.mean()
        
        # ============ 4. Weight Regularization Loss ============
        # 确保权重和为1(softmax已保证,但作为额外约束)
        weight_sum_loss = F.mse_loss(attention_weights.sum(dim=1), 
                                     torch.ones(batch_size, device=attention_weights.device))
        
        # ============ 5. Sparsity Regularization Loss ============
        # 鼓励稀疏的注意力分布（每个spot只使用少数细胞类型）
        sparsity_loss = -torch.mean(attention_weights * torch.log(attention_weights + 1e-8))

        # ============ 6. 基因维度 Pearson/Cosine（跨 spot） ============
        gene_pearson_loss = torch.tensor(0.0, device=attention_weights.device)
        gene_cosine_loss = torch.tensor(0.0, device=attention_weights.device)
        if reconstructed_log.shape[0] > 1:  # 至少两个spot才有意义
            rec_T = reconstructed_log.transpose(0, 1)  # [n_genes, batch]
            true_T = true_log.transpose(0, 1)
            rec_centered = rec_T - rec_T.mean(dim=1, keepdim=True)
            true_centered = true_T - true_T.mean(dim=1, keepdim=True)
            num = (rec_centered * true_centered).sum(dim=1)
            denom = torch.sqrt((rec_centered.pow(2).sum(dim=1)) * (true_centered.pow(2).sum(dim=1)) + 1e-8)
            gene_corr = num / (denom + 1e-8)
            gene_pearson_loss = 1.0 - gene_corr.mean()
            gene_cos = F.cosine_similarity(rec_T, true_T, dim=1)
            gene_cosine_loss = 1.0 - gene_cos.mean()

        proportion_loss = torch.tensor(0.0, device=attention_weights.device)
        
        if self.sc_celltype_proportions is not None:
            # 计算ST数据中预测的全局细胞类型比例
            # attention_weights: [n_spots, n_cell_types]
            # 全局比例 = 所有spot的平均权重（这是关键：对整个batch求平均）
            st_predicted_proportions = attention_weights.mean(dim=0)  # [n_cell_types]
            
            # 单细胞数据的真实比例（从聚类统计得到）
            sc_proportions = self.sc_celltype_proportions.to(attention_weights.device)  # [n_cell_types]

            # KL(ST || SC) = sum(p_st * log(p_st / p_sc))
            proportion_loss = torch.sum(
                st_predicted_proportions * torch.log(
                    (st_predicted_proportions + 1e-8) / (sc_proportions + 1e-8)
                )
            )

        # ============ 总损失 ============
        total_loss = (self.lambda_pearson * L_pearson +
                     self.lambda_mse * L_mse +
                     self.lambda_cosine * L_cosine +
                     self.lambda_gene_pearson * gene_pearson_loss +
                     self.lambda_gene_cosine * gene_cosine_loss +
                     self.lambda_reg * weight_sum_loss +
                     self.lambda_sparse * sparsity_loss +
                     self.lambda_proportion * proportion_loss)

        return {
            'total_loss': total_loss,
            'pearson_loss': L_pearson,
            'mse_loss': L_mse,
            'cosine_loss': L_cosine,
            'gene_pearson_loss': gene_pearson_loss,
            'gene_cosine_loss': gene_cosine_loss,
            'weight_reg': weight_sum_loss,
            'sparsity_loss': sparsity_loss,
            'proportion_loss': proportion_loss
        }


# ================================
# Stage 1: VAE Training & Utilities
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import umap
import anndata as ad


class SimpleDataset:
    """Simple dataset for VAE training"""
    def __init__(self, X, modality):
        self.X = torch.FloatTensor(X)
        self.modality = torch.LongTensor(modality)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.modality[idx]


def train_vae_epoch(vae, train_loader, optimizer, device, loss_type='mse', beta=1.0, lambda_mmd=0.0):
    """Train VAE for one epoch
    
    Args:
        vae: VAE model (can be VAE or DualDecoderVAE)
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        loss_type: 'mse' or 'zinb'
        beta: KL divergence weight
        lambda_mmd: MMD loss weight
    
    Returns:
        avg_loss, avg_recon, avg_kl, avg_mmd
    """
    vae.train()
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    epoch_mmd = 0.0
    
    # 检查是否是双解码器架构
    is_dual_decoder = hasattr(vae, 'decoder_sc') and hasattr(vae, 'decoder_st')
    
    for batch_data, batch_modality in train_loader:
        batch_data = batch_data.to(device)
        batch_modality = batch_modality.to(device)
        
        optimizer.zero_grad()
        
        # VAE forward pass
        if is_dual_decoder:
            # 双解码器: 需要传入modality参数
            if loss_type == 'zinb':
                mean, disp, pi, mu, log_var, z = vae(batch_data, batch_modality)
                total_loss, recon_loss, kl_div = zinb_loss_function(
                    mean, disp, pi, batch_data, mu, log_var, beta=beta
                )
            else:
                recon_data, mu, log_var, z = vae(batch_data, batch_modality)
                total_loss, recon_loss, kl_div = vae_loss_function(
                    recon_data, batch_data, mu, log_var, beta=beta
                )

        # Compute MMD loss for modality alignment
        mmd_loss = torch.tensor(0.0, device=device)
        if lambda_mmd > 0:
            # Separate SC and ST embeddings in this batch
            sc_mask = batch_modality == 0
            st_mask = batch_modality == 1
            
            # Only compute MMD if both modalities present in batch
            if sc_mask.sum() > 0 and st_mask.sum() > 0:
                sc_embeddings = z[sc_mask]
                st_embeddings = z[st_mask]
                mmd_loss = compute_mmd(sc_embeddings, st_embeddings, kernel='rbf')
        
        # Total loss with MMD
        total_loss = total_loss + lambda_mmd * mmd_loss
        
        # Normalize loss
        total_loss = total_loss / len(batch_data)
        recon_loss = recon_loss / len(batch_data)
        kl_div = kl_div / len(batch_data)
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_div.item()
        epoch_mmd += mmd_loss.item() if lambda_mmd > 0 else 0.0
    
    avg_loss = epoch_loss / len(train_loader)
    avg_recon = epoch_recon / len(train_loader)
    avg_kl = epoch_kl / len(train_loader)
    avg_mmd = epoch_mmd / len(train_loader)
    
    return avg_loss, avg_recon, avg_kl, avg_mmd


def evaluate_vae(vae, test_loader, device, loss_type='mse', beta=1.0):
    """Evaluate VAE
    
    Args:
        vae: VAE model (can be VAE or DualDecoderVAE)
        test_loader: Test data loader
        device: Device (cuda/cpu)
        loss_type: 'mse' or 'zinb'
        beta: KL divergence weight
    
    Returns:
        test_loss
    """
    vae.eval()
    total_loss = 0.0
    
    # 检查是否是双解码器架构
    is_dual_decoder = hasattr(vae, 'decoder_sc') and hasattr(vae, 'decoder_st')
    
    with torch.no_grad():
        for batch_data, batch_modality in test_loader:
            batch_data = batch_data.to(device)
            batch_modality = batch_modality.to(device)
            
            if is_dual_decoder:
                # 双解码器: 需要传入modality参数
                if loss_type == 'zinb':
                    mean, disp, pi, mu, log_var, z = vae(batch_data, batch_modality)
                    loss, _, _ = zinb_loss_function(mean, disp, pi, batch_data, mu, log_var, beta)
                else:
                    recon_data, mu, log_var, z = vae(batch_data, batch_modality)
                    loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
            else:
                # 单解码器: 不需要modality参数
                if loss_type == 'zinb':
                    mean, disp, pi, mu, log_var, z = vae(batch_data)
                    loss, _, _ = zinb_loss_function(mean, disp, pi, batch_data, mu, log_var, beta)
                else:
                    recon_data, mu, log_var, z = vae(batch_data)
                    loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
                
            total_loss += loss.item() / len(batch_data)
    
    return total_loss / len(test_loader)


def train_vae(vae, train_X, test_X, train_modality, test_modality, device,
              batch_size=256, n_epochs=100, lr=1e-3, beta=1.0, loss_type='mse', 
              lambda_mmd=1.0, output_dir="./stage1_results", print_every=50,
              patience=20, min_delta=1.0):
    """Train VAE with optional MMD loss for modality alignment
    
    Args:
        vae: VAE model
        train_X: Training data
        test_X: Test data (can be None to use all data for training)
        train_modality: Training modality labels
        test_modality: Test modality labels (can be None if test_X is None)
        device: Device (cuda/cpu)
        batch_size: Batch size
        n_epochs: Number of epochs
        lr: Learning rate
        beta: KL divergence weight
        loss_type: 'mse' or 'zinb'
        lambda_mmd: MMD loss weight
        output_dir: Output directory for saving plots
        print_every: Print loss every N epochs (default: 50)
    
    Returns:
        best_loss
    """
    # Data loader
    from torch.utils.data import DataLoader
    train_dataset = SimpleDataset(train_X, train_modality)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create test loader only if test data exists
    if test_X is not None:
        test_dataset = SimpleDataset(test_X, test_modality)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None
    
    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Training history
    train_losses = []
    test_losses = []
    recon_losses = []
    kl_losses = []
    mmd_losses = []
    
    best_loss = float('inf')
    patience_counter = 0
    # 使用传入的 patience 参数（默认 20）
    
    for epoch in range(n_epochs):
        # Training
        avg_loss, avg_recon, avg_kl, avg_mmd = train_vae_epoch(
            vae, train_loader, optimizer, device, loss_type, beta, lambda_mmd
        )
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        mmd_losses.append(avg_mmd)
        
        if test_loader is not None:
            test_loss = evaluate_vae(vae, test_loader, device, loss_type, beta)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            # Print every N epochs (total loss only)
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | train_loss={avg_loss:.4f} | test_loss={test_loss:.4f}")
            
            # Save best model based on test loss
            # 绝对改进阈值：best_loss - test_loss > min_delta
            improvement = best_loss - test_loss
            if improvement > min_delta:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                if best_loss != float('inf'):
                    print(f"Early stopping at epoch {epoch+1}/{n_epochs}, best_loss={best_loss:.4f}")
                else:
                    print(f"Early stopping at epoch {epoch+1}/{n_epochs}")
                break
        else:
            # No test set - use training loss for scheduling and early stopping
            scheduler.step(avg_loss)
            
            # Print every N epochs (total loss only)
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | train_loss={avg_loss:.4f}")
            
            # Save best model based on training loss
            # 绝对改进阈值：best_loss - avg_loss > min_delta
            improvement = best_loss - avg_loss
            if improvement > min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                if best_loss != float('inf'):
                    print(f"Early stopping at epoch {epoch+1}/{n_epochs}, best_loss={best_loss:.4f}")
                else:
                    print(f"Early stopping at epoch {epoch+1}/{n_epochs}")
                break
    
    # Plot training curves (only if output_dir is provided)
    if output_dir is not None:
        plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses, mmd_losses, output_dir)
    
    return best_loss


def plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses, mmd_losses=None, output_dir=None):
    """Plot VAE training curves
    
    Args:
        output_dir: If None, skip saving the plot
    """
    if output_dir is None:
        return
    
    # Determine if we need to plot MMD
    has_mmd = mmd_losses is not None and len(mmd_losses) > 0 and max(mmd_losses) > 0
    
    if has_mmd:
        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    ax1.plot(train_losses, label='Train')
    if len(test_losses) > 0:
        # ✅ Test losses are now recorded every epoch (same length as train_losses)
        ax1.plot(test_losses, label='Test', linestyle='--')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Reconstruction loss
    ax2.plot(recon_losses, 'g-')
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # KL divergence
    ax3.plot(kl_losses, 'r-')
    ax3.set_title('KL Divergence')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('KL Div')
    ax3.grid(True)
    
    # Loss components comparison
    ax4.plot(recon_losses, label='Reconstruction', color='green')
    ax4.plot(kl_losses, label='KL Divergence', color='red')
    if has_mmd:
        ax4.plot(mmd_losses, label='MMD', color='purple')
    ax4.set_title('Loss Components')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    if has_mmd:
        # MMD loss
        ax5.plot(mmd_losses, 'purple')
        ax5.set_title('MMD Loss (Modality Alignment)')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('MMD')
        ax5.grid(True)
        
        # All components normalized
        ax6.plot(np.array(recon_losses) / (max(recon_losses) + 1e-8), label='Recon (norm)', color='green')
        ax6.plot(np.array(kl_losses) / (max(kl_losses) + 1e-8), label='KL (norm)', color='red')
        ax6.plot(np.array(mmd_losses) / (max(mmd_losses) + 1e-8), label='MMD (norm)', color='purple')
        ax6.set_title('Normalized Loss Components')
        ax6.set_xlabel('Epochs')
        ax6.set_ylabel('Normalized Loss')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vae_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_vae_checkpoint(vae, label_encoder, marker_genes, genes, all_genes,
                       sc_clusters, resolution, filepath):
    """Save VAE weights and basic metadata (cluster data saved separately)."""
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'label_encoder': label_encoder,
        'marker_genes': marker_genes,
        'genes': genes,
        'input_dim': len(genes),
        'latent_dim': vae.latent_dim,
        'output_type': vae.output_type,
        'sc_clusters': sc_clusters,
        'resolution': resolution,
        'all_genes': all_genes
    }, filepath)
    
def load_vae_for_inference(filepath, device):
    """Load VAE model for inference (basic loading)
    
    Args:
        filepath: Path to checkpoint
        device: Device (cuda/cpu)
    
    Returns:
        vae, label_encoder, marker_genes, genes
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    output_type = checkpoint.get('output_type', 'mse')
    
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(device)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    
    label_encoder = checkpoint['label_encoder']
    marker_genes = checkpoint['marker_genes']
    genes = checkpoint['genes']
    
    return vae, label_encoder, marker_genes, genes


def load_vae_pretrained(filepath, device):
    """Load pretrained VAE weights for continued training
    
    Args:
        filepath: Path to checkpoint
        device: Device (cuda/cpu)
    
    Returns:
        Tuple of (vae, components_dict, output_type, latent_dim)
        where components_dict contains all other checkpoint components
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pretrained model not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model architecture info
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    output_type = checkpoint.get('output_type', 'mse')
    
    
    # Build VAE model with same architecture
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(device)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    
    # Extract other components from .pth
    components = {
        'label_encoder': checkpoint.get('label_encoder', None),
        'marker_genes': checkpoint.get('marker_genes', None),
        'genes': checkpoint.get('genes', None),
        'all_genes': checkpoint.get('all_genes', None),
        'sc_clusters': checkpoint.get('sc_clusters', None),
        'resolution': checkpoint.get('resolution', 0.5)
    }
    
    # Try to load cluster data from .npz file
    npz_filepath = filepath.replace('.pth', '_cluster_data.npz')
    if os.path.exists(npz_filepath):
        cluster_data = np.load(npz_filepath, allow_pickle=True)
        
        cluster_ids = cluster_data['cluster_ids']
        prototypes_array = cluster_data['cluster_prototypes']
        expressions_array = cluster_data['cluster_expressions']
        expressions_full_array = cluster_data['cluster_expressions_full']
        
        # Convert back to dict format
        cluster_prototypes = {int(cid): prototypes_array[i] for i, cid in enumerate(cluster_ids)}
        cluster_expressions = {int(cid): expressions_array[i] for i, cid in enumerate(cluster_ids)}
        cluster_expressions_full = {int(cid): expressions_full_array[i] for i, cid in enumerate(cluster_ids)}
        
        # Load cell weights if available
        cluster_cell_weights = {}
        for cid in cluster_ids:
            weight_key = f'cluster_{cid}_weights'
            if weight_key in cluster_data:
                cluster_cell_weights[int(cid)] = cluster_data[weight_key]
        
        # Load celltype mapping if available
        cluster_to_celltype = None
        if 'cluster_to_celltype' in cluster_data:
            celltype_mapping_array = cluster_data['cluster_to_celltype']
            cluster_to_celltype = {str(row['cluster_id']): str(row['celltype']) 
                                  for row in celltype_mapping_array}
        
        components['cluster_prototypes'] = cluster_prototypes
        components['cluster_expressions'] = cluster_expressions
        components['cluster_expressions_full'] = cluster_expressions_full
        components['cluster_expressions_full_count'] = cluster_expressions_full
        components['cluster_cell_weights'] = cluster_cell_weights if cluster_cell_weights else None
        components['cluster_to_celltype'] = cluster_to_celltype

    else:
        # Try to load from old format (in .pth)
        components['cluster_prototypes'] = checkpoint.get('cluster_prototypes', None)
        components['cluster_expressions'] = checkpoint.get('cluster_expressions', None)
        components['cluster_expressions_full'] = checkpoint.get('cluster_expressions_full', None)
        components['cluster_expressions_full_count'] = checkpoint.get('cluster_expressions_full_count', None)
        components['cluster_cell_weights'] = None
    
    return vae, components, output_type, latent_dim


def plot_modality_alignment_umap(vae, train_X, train_modality, device, y_train=None, output_dir="./stage1_results"):
    """
    Plot UMAP visualization of SC and ST modality alignment
    
    Args:
        vae: Trained VAE model
        train_X: Training data (combined SC + ST)
        train_modality: Modality labels (0=SC, 1=ST)
        device: Device (cuda/cpu)
        y_train: Optional cluster labels for SC samples
        output_dir: Output directory
    """
    # Get embeddings from trained VAE
    vae.eval()
    with torch.no_grad():
        batch_size = 1000
        all_embeddings = []
        
        for i in range(0, len(train_X), batch_size):
            batch_data = train_X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_data).to(device)
            mu, log_var = vae.encoder(batch_tensor)
            all_embeddings.append(mu.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by modality (SC vs ST)
    ax1 = axes[0]
    sc_mask = train_modality == 0
    st_mask = train_modality == 1
    
    ax1.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
               c='#1f77b4', s=20, alpha=0.6, label=f'SC (n={sum(sc_mask)})', edgecolors='none')
    ax1.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
               c='#ff7f0e', s=20, alpha=0.6, label=f'ST (n={sum(st_mask)})', edgecolors='none')
    
    ax1.set_title('UMAP: SC vs ST Modality Alignment', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.legend(fontsize=11, markerscale=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by cluster (SC only) + ST
    ax2 = axes[1]
    
    if y_train is not None:
        # Get SC data with cluster labels
        sc_clusters = y_train
        n_clusters = len(np.unique(sc_clusters))
        
        # Use a colormap for clusters
        cmap = plt.cm.get_cmap('tab20', n_clusters)
        
        # Plot each cluster
        for cluster_id in np.unique(sc_clusters):
            cluster_mask_in_sc = sc_clusters == cluster_id
            # Convert to global index (all train_X)
            sc_indices = np.where(sc_mask)[0]
            cluster_global_mask = np.zeros(len(train_X), dtype=bool)
            cluster_global_mask[sc_indices[cluster_mask_in_sc]] = True
            
            ax2.scatter(umap_coords[cluster_global_mask, 0], 
                       umap_coords[cluster_global_mask, 1],
                       c=[cmap(cluster_id)], s=20, alpha=0.6, 
                       label=f'Cluster {cluster_id}', edgecolors='none')
        
        # Plot ST in gray
        ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
                   c='lightgray', s=20, alpha=0.4, label=f'ST (n={sum(st_mask)})', edgecolors='none')
        
        ax2.set_title(f'UMAP: SC Clusters (n={n_clusters}) + ST', fontsize=14, fontweight='bold')
    else:
        # If no cluster labels, just plot SC and ST
        ax2.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
                   c='#1f77b4', s=20, alpha=0.6, label=f'SC', edgecolors='none')
        ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
                   c='#ff7f0e', s=20, alpha=0.6, label=f'ST', edgecolors='none')
        ax2.set_title('UMAP: SC + ST', fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('UMAP 1', fontsize=12)
    ax2.set_ylabel('UMAP 2', fontsize=12)
    ax2.legend(fontsize=9, markerscale=2, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/modality_alignment_umap.png", dpi=300, bbox_inches='tight')
    plt.close()
