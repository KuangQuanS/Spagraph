import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, Optional, List


class VAEEncoder(nn.Module):
    """VAE编码器 - 用于基因表达编码推理（冻结权重）"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def encode(self, x):
        """推理时的编码方法"""
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class VAEExpressionProcessor:
    """基因表达编码处理器 - 使用预训练的VAE编码器（权重冻结）"""
    
    def __init__(self, stage1_dir: str, stage2_dir: str, sample_id: str,
                 vae_weight_path: str, vae_latent_dim: int = 64, 
                 vae_hidden_dim: int = 256, device: str = 'cpu'):
        """
        初始化表达编码器
        
        Args:
            stage1_dir: Stage 1结果目录
            stage2_dir: Stage 2结果目录  
            sample_id: 样本ID
            vae_weight_path: 预训练VAE权重路径（必需）
            vae_latent_dim: VAE隐空间维度
            vae_hidden_dim: VAE隐层维度
            device: 设备
        """
        self.stage1_dir = Path(stage1_dir)
        self.stage2_dir = Path(stage2_dir)
        self.sample_id = sample_id
        self.latent_dim = vae_latent_dim
        self.device = device
        
        # 加载cluster表达量
        self.cluster_expr = self.load_cluster_expressions()
        self.n_genes = self.cluster_expr.shape[1]
        
        # 初始化并加载预训练encoder
        self.encoder = VAEEncoder(
            input_dim=self.n_genes,
            hidden_dim=vae_hidden_dim,
            latent_dim=vae_latent_dim
        ).to(device)
        self.load_encoder_weights(vae_weight_path)
        self.encoder.eval()
        
        # 冻结encoder权重
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 加载composition矩阵和计算celltype表达量
        self.composition_matrix = self.load_composition_matrix()
        self.celltype_expr = self.compute_celltype_expressions()
    
    def load_encoder_weights(self, vae_weight_path: str):
        """加载预训练的VAE encoder权重"""
        if not Path(vae_weight_path).exists():
            raise FileNotFoundError(f"找不到VAE权重文件: {vae_weight_path}")
        
        state = torch.load(vae_weight_path, map_location=self.device)
        
        # 如果权重是完整VAE模型，只提取encoder部分
        if 'encoder.fc1.weight' in state:
            encoder_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_state)
        else:
            self.encoder.load_state_dict(state)
        
        print(f"[VAEEncoder] 已加载权重: {vae_weight_path}")
    
    def load_cluster_expressions(self) -> pd.DataFrame:
        """从Stage 1加载cluster表达量"""
        cluster_expr_path = self.stage1_dir / f"{self.sample_id}_cluster_expressions.csv"
        if cluster_expr_path.exists():
            return pd.read_csv(cluster_expr_path, index_col=0)
        else:
            raise FileNotFoundError(f"找不到cluster表达量文件: {cluster_expr_path}")
    
    def load_composition_matrix(self) -> pd.DataFrame:
        """从Stage 2加载celltype composition"""
        composition_path = self.stage2_dir / f"{self.sample_id}_cell_composition.csv"
        if composition_path.exists():
            return pd.read_csv(composition_path, index_col=0)
        else:
            # 返回默认的composition矩阵
            n_spots = 100
            n_clusters = self.cluster_expr.shape[0]
            comp = pd.DataFrame(
                np.random.dirichlet(np.ones(n_clusters), n_spots),
                columns=self.cluster_expr.index
            )
            return comp
    
    def compute_celltype_expressions(self) -> pd.DataFrame:
        """计算celltype表达量"""
        celltype_expr = self.cluster_expr.copy()
        return celltype_expr
    
    def encode_expression(self, expr_vec: np.ndarray) -> torch.Tensor:
        """
        使用encoder编码表达量向量
        
        Args:
            expr_vec: [n_genes] 基因表达向量
        
        Returns:
            z: [latent_dim] 编码向量
        """
        expr_tensor = torch.tensor(expr_vec, dtype=torch.float32).to(self.device)
        if expr_tensor.dim() == 1:
            expr_tensor = expr_tensor.unsqueeze(0)
        
        with torch.no_grad():
            z = self.encoder.encode(expr_tensor)
        
        return z.squeeze(0)


class STHeteroDataset:
    """空间转录组异构图数据集"""
    
    def __init__(self, st_h5ad_path: str, cluster_expr: pd.DataFrame, 
                 celltype_expr: pd.DataFrame, vae_encoder, graph_data: Dict, 
                 celltype_full_expr: Optional[pd.DataFrame] = None,
                 device: str = 'cpu', transform=None):
        """
        初始化数据集
        
        Args:
            st_h5ad_path: ST数据路径
            cluster_expr: Cluster marker基因表达量DataFrame [n_clusters, n_marker_genes]
            celltype_expr: Celltype marker基因表达量DataFrame [n_celltypes, n_marker_genes]
            vae_encoder: 预训练的VAE encoder（冻结）
            graph_data: 图数据
            celltype_full_expr: Celltype全基因表达量DataFrame [n_celltypes, n_all_genes]（可选）
            device: 设备
        """
        self.cluster_expr = cluster_expr
        self.celltype_expr = celltype_expr
        self.celltype_full_expr = celltype_full_expr if celltype_full_expr is not None else celltype_expr
        self.vae_encoder = vae_encoder
        self.graph_data = graph_data
        self.device = device
        self.transform = transform
        
        # 获取marker基因表达（用于embedding计算）
        self.celltype_marker_expr = graph_data.get('celltype_marker_expr', celltype_expr)
        
        # 加载ST数据
        self.adata = sc.read_h5ad(st_h5ad_path)
        self.n_spots = self.adata.n_obs
        
        # 获取ST中与cluster表达量匹配的基因
        self.genes = cluster_expr.columns.tolist()
        self.st_X = self.adata[:, self.genes].X
        if hasattr(self.st_X, 'toarray'):
            self.st_X = self.st_X.toarray()
    
    def encode_expression(self, expr_vec: np.ndarray) -> torch.Tensor:
        """使用VAE encoder对基因表达进行编码"""
        expr_tensor = torch.tensor(expr_vec, dtype=torch.float32).to(self.device)
        if expr_tensor.dim() == 1:
            expr_tensor = expr_tensor.unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = self.vae_encoder(expr_tensor)
        
        return mu.squeeze(0)
    
    def __len__(self):
        return self.n_spots
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取spot图像（如果有）
        try:
            image = self.adata.obsm['spatial_image'][idx]
            image = torch.tensor(image, dtype=torch.float32)
        except:
            # 如果没有图像，使用随机初始化
            image = torch.randn(256)  # 假设256维的特征向量
        
        # 获取表达量的encoder编码（使用marker基因）
        expr_vec = self.st_X[idx].astype(np.float32)
        expr_latent = self.encode_expression(expr_vec)
        
        # 获取所有celltype的编码表达量（使用marker基因进行embedding计算）
        celltype_expr_latent_list = []
        for i in range(len(self.celltype_marker_expr)):
            ct_expr = self.celltype_marker_expr.iloc[i].values.astype(np.float32)
            ct_latent = self.encode_expression(ct_expr)
            celltype_expr_latent_list.append(ct_latent)
        celltype_expr_latent = torch.stack(celltype_expr_latent_list, dim=0)
        
        # 获取图的边信息
        batch = {
            'images': image,
            'expr_latent': expr_latent,
            'celltype_expr_latent': celltype_expr_latent,
            'edge_index_ss': torch.tensor(self.graph_data['edge_index_ss'], dtype=torch.long),
            'edge_attr_ss': torch.tensor(self.graph_data['edge_attr_ss'], dtype=torch.float32),
            'edge_index_sc': torch.tensor(self.graph_data['edge_index_sc'], dtype=torch.long),
            'edge_attr_sc': torch.tensor(self.graph_data['edge_attr_sc'], dtype=torch.float32),
            'edge_index_cc': torch.tensor(self.graph_data['edge_index_cc'], dtype=torch.long),
            'edge_attr_cc': torch.tensor(self.graph_data['edge_attr_cc'], dtype=torch.float32),
        }
        
        return batch
