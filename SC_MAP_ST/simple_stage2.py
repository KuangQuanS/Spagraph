#!/usr/bin/env python3
"""
简化版SC_MAP_ST - 第二阶段：图注意力网络解卷积
使用VAE encoder + GAT进行空间转录组解卷积
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入第一阶段的VAE组件
from simple_stage1 import VAE, VAEEncoder, SimpleVAEMapper

class SpatialDataset(Dataset):
    """空间转录组数据集"""
    def __init__(self, st_data, spatial_coords, spot_ids):
        self.st_data = torch.FloatTensor(st_data)
        self.spatial_coords = torch.FloatTensor(spatial_coords)
        self.spot_ids = spot_ids
        
    def __len__(self):
        return len(self.st_data)
    
    def __getitem__(self, idx):
        return {
            'expression': self.st_data[idx],
            'coords': self.spatial_coords[idx],
            'spot_id': self.spot_ids[idx]
        }

class HeterogeneousGATDeconvolution(nn.Module):
    """异构图注意力网络解卷积模型"""
    
    def __init__(self, 
                 embedding_dim=128,
                 n_cell_types=9,
                 gat_hidden_dim=64,
                 gat_layers=3,
                 gat_heads=4,
                 dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_cell_types = n_cell_types
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        
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
        
        # 3. 注意力权重计算（spot-celltype相似度）
        self.attention_mlp = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 4. 解卷积输出层
        self.deconv_head = nn.Sequential(
            nn.Linear(gat_hidden_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim, n_cell_types),
            nn.Softmax(dim=-1)  # 细胞类型比例
        )
        
    def build_heterogeneous_graph(self, 
                                spot_embeddings: torch.Tensor,
                                spatial_coords: torch.Tensor,
                                celltype_prototypes: torch.Tensor,
                                k_spatial: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建异构图：Spot节点 + CellType节点
        
        Args:
            spot_embeddings: Spot嵌入 [n_spots, embedding_dim]
            spatial_coords: 空间坐标 [n_spots, 2]
            celltype_prototypes: 细胞类型原型 [n_cell_types, embedding_dim]
            k_spatial: 空间邻居数
            
        Returns:
            edge_index: 边索引 [2, n_edges]
            edge_attr: 边属性 [n_edges, 1]
            node_features: 节点特征 [n_total_nodes, gat_hidden_dim]
        """
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
            nbrs = NearestNeighbors(n_neighbors=min(k_spatial+1, len(coords_np))).fit(coords_np)
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
                if similarity > 0.1:  # 相似度阈值
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
        """
        前向传播
        
        Args:
            spot_embeddings: Spot嵌入 [n_spots, embedding_dim]
            spatial_coords: 空间坐标 [n_spots, 2]
            celltype_prototypes: 细胞类型原型 [n_cell_types, embedding_dim]
            
        Returns:
            结果字典
        """
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
        
        # 4. 计算注意力权重（增强版相似度）
        attention_scores = []
        for i in range(n_spots):
            spot_feat = spot_features[i:i+1].expand(self.n_cell_types, -1)  # [n_cell_types, gat_hidden_dim]
            combined = torch.cat([spot_feat, celltype_features], dim=1)  # [n_cell_types, 2*gat_hidden_dim]
            scores = self.attention_mlp(combined).squeeze()  # [n_cell_types]
            attention_scores.append(scores)
        
        attention_matrix = torch.stack(attention_scores)  # [n_spots, n_cell_types]
        
        # 5. 解卷积预测细胞类型比例
        cell_proportions = self.deconv_head(spot_features)  # [n_spots, n_cell_types]
        
        return {
            'spot_features': spot_features,
            'celltype_features': celltype_features,
            'attention_scores': attention_matrix,
            'cell_proportions': cell_proportions,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

class SpatialDeconvolutionLoss(nn.Module):
    """空间解卷积损失函数"""
    
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.beta = beta   # 正则化权重
        
    def forward(self, 
               predicted_proportions: torch.Tensor,
               attention_scores: torch.Tensor,
               celltype_expressions: torch.Tensor,
               target_expressions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算解卷积损失
        
        Args:
            predicted_proportions: 预测的细胞类型比例 [n_spots, n_cell_types]
            attention_scores: 注意力分数 [n_spots, n_cell_types]
            celltype_expressions: 细胞类型表达谱 [n_cell_types, n_genes]
            target_expressions: 目标spot表达 [n_spots, n_genes]
            
        Returns:
            损失字典
        """
        # 1. 组合相似度（预测比例 + 注意力分数）
        combined_weights = predicted_proportions + self.beta * attention_scores
        combined_weights = F.softmax(combined_weights, dim=1)  # 重新归一化
        
        # 2. 重建表达谱
        reconstructed_expressions = torch.mm(combined_weights, celltype_expressions)  # [n_spots, n_genes]
        
        # 3. 重建损失
        recon_loss = F.mse_loss(reconstructed_expressions, target_expressions)
        
        # 4. 比例正则化（确保比例和为1）
        proportion_reg = F.mse_loss(predicted_proportions.sum(dim=1), torch.ones(predicted_proportions.shape[0], device=predicted_proportions.device))
        
        # 5. 稀疏性正则化（鼓励稀疏的细胞类型分布）
        sparsity_reg = torch.mean(predicted_proportions * torch.log(predicted_proportions + 1e-8))
        
        # 总损失
        total_loss = (self.alpha * recon_loss + 
                     0.1 * proportion_reg + 
                     0.01 * (-sparsity_reg))  # 负号因为我们想要最大化稀疏性
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'proportion_reg': proportion_reg,
            'sparsity_reg': sparsity_reg,
            'reconstructed_expressions': reconstructed_expressions,
            'combined_weights': combined_weights
        }

class Stage2GATDeconvolution:
    """第二阶段GAT解卷积训练器"""
    
    def __init__(self, 
                 stage1_model_path: str,
                 output_dir: str = "./simple_sc_results/stage2",
                 device: str = None):
        """
        初始化第二阶段训练器
        
        Args:
            stage1_model_path: 第一阶段VAE模型路径
            output_dir: 输出目录
            device: 计算设备
        """
        self.stage1_model_path = stage1_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 第二阶段GAT解卷积初始化")
        print(f"   第一阶段模型: {stage1_model_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   设备: {self.device}")
        
        # 模型组件
        self.vae_encoder = None
        self.gat_model = None
        self.loss_fn = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_prototypes = None
        
    def load_stage1_components(self):
        """加载第一阶段VAE组件"""
        print("📂 加载第一阶段VAE组件...")
        
        checkpoint = torch.load(self.stage1_model_path, map_location=self.device)
        
        # 重建VAE
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        # 只使用encoder部分
        full_vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        full_vae.load_state_dict(checkpoint['vae_state_dict'])
        
        # 提取encoder
        self.vae_encoder = full_vae.encoder
        self.vae_encoder.eval()  # 冻结encoder
        
        # 其他信息
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        
        print(f"   ✅ VAE Encoder加载成功: {input_dim} -> {latent_dim}")
        print(f"   细胞类型: {list(self.label_encoder.classes_)}")
        print(f"   Marker基因数: {len(self.genes)}")
        
        # 冻结encoder参数
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
    
    def compute_celltype_prototypes(self, sc_data: np.ndarray, sc_labels: np.ndarray) -> torch.Tensor:
        """
        计算每个细胞类型的原型（平均embedding）
        
        Args:
            sc_data: SC表达数据 [n_cells, n_genes]
            sc_labels: 细胞类型标签 [n_cells]
            
        Returns:
            细胞类型原型 [n_cell_types, embedding_dim]
        """
        print("🧬 计算细胞类型原型...")
        
        # 转换为tensor
        sc_tensor = torch.FloatTensor(sc_data).to(self.device)
        
        # 计算所有细胞的embedding
        with torch.no_grad():
            mu, log_var = self.vae_encoder(sc_tensor)
            # 使用均值作为embedding
            sc_embeddings = mu
        
        # 计算每个细胞类型的平均embedding
        prototypes = []
        cell_types = self.label_encoder.classes_
        
        for i, cell_type in enumerate(cell_types):
            # 找到该细胞类型的所有细胞
            cell_mask = (sc_labels == i)
            if cell_mask.sum() > 0:
                cell_embeddings = sc_embeddings[cell_mask]
                prototype = cell_embeddings.mean(dim=0)
                prototypes.append(prototype)
                
                print(f"   {cell_type}: {cell_mask.sum()} 个细胞 -> 原型embedding")
        
        self.celltype_prototypes = torch.stack(prototypes)  # [n_cell_types, embedding_dim]
        
        print(f"   ✅ 细胞类型原型: {self.celltype_prototypes.shape}")
        
        return self.celltype_prototypes
    
    def compute_celltype_expressions(self, sc_data: np.ndarray, sc_labels: np.ndarray) -> torch.Tensor:
        """
        计算每个细胞类型的平均表达谱
        
        Args:
            sc_data: SC表达数据 [n_cells, n_genes]
            sc_labels: 细胞类型标签 [n_cells]
            
        Returns:
            细胞类型表达谱 [n_cell_types, n_genes]
        """
        print("🧬 计算细胞类型表达谱...")
        
        cell_types = self.label_encoder.classes_
        celltype_expressions = []
        
        for i, cell_type in enumerate(cell_types):
            # 找到该细胞类型的所有细胞
            cell_mask = (sc_labels == i)
            if cell_mask.sum() > 0:
                cell_expression = sc_data[cell_mask].mean(axis=0)
                celltype_expressions.append(cell_expression)
                
                print(f"   {cell_type}: {cell_mask.sum()} 个细胞 -> 表达谱")
        
        celltype_expressions = torch.FloatTensor(np.array(celltype_expressions)).to(self.device)
        
        print(f"   ✅ 细胞类型表达谱: {celltype_expressions.shape}")
        
        return celltype_expressions
    
    def build_gat_model(self, n_cell_types: int):
        """构建GAT解卷积模型"""
        print("🏗️ 构建GAT解卷积模型...")
        
        self.gat_model = HeterogeneousGATDeconvolution(
            embedding_dim=128,  # VAE embedding维度
            n_cell_types=n_cell_types,
            gat_hidden_dim=64,
            gat_layers=3,
            gat_heads=4,
            dropout=0.1
        ).to(self.device)
        
        self.loss_fn = SpatialDeconvolutionLoss(alpha=1.0, beta=0.1)
        
        print(f"   GAT模型: 异构图 -> {n_cell_types}类比例")
        gat_params = sum(p.numel() for p in self.gat_model.parameters())
        print(f"   参数量: {gat_params:,}")
    
    def prepare_st_data(self, st_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备空间转录组数据
        
        Args:
            st_file: ST数据文件路径
            
        Returns:
            ST表达数据, 空间坐标, spot IDs
        """
        print(f"📂 加载空间转录组数据: {st_file}")
        
        # 这里需要根据实际数据格式调整
        # 假设是h5ad格式
        st_adata = sc.read_h5ad(st_file)
        
        # 提取marker基因
        available_genes = [g for g in self.genes if g in st_adata.var.index]
        st_subset = st_adata[:, available_genes].copy()
        
        # 使用原始counts
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        # 空间坐标
        if 'spatial' in st_adata.obsm:
            spatial_coords = st_adata.obsm['spatial']
        else:
            # 如果没有空间坐标，生成伪坐标
            print("   ⚠️ 未找到空间坐标，生成伪坐标...")
            n_spots = len(st_adata)
            grid_size = int(np.ceil(np.sqrt(n_spots)))
            x_coords = np.tile(np.arange(grid_size), (grid_size, 1)).flatten()[:n_spots]
            y_coords = np.repeat(np.arange(grid_size), grid_size)[:n_spots]
            spatial_coords = np.column_stack([x_coords, y_coords])
        
        spot_ids = list(st_adata.obs.index)
        
        print(f"   ST数据: {st_X.shape}")
        print(f"   空间坐标: {spatial_coords.shape}")
        print(f"   可用基因: {len(available_genes)}/{len(self.genes)}")
        
        return st_X, spatial_coords, spot_ids
    
    def train_epoch(self, 
                   st_data: torch.Tensor, 
                   spatial_coords: torch.Tensor,
                   celltype_expressions: torch.Tensor,
                   optimizer) -> Dict[str, float]:
        """训练一个epoch"""
        self.gat_model.train()
        
        # 计算spot embeddings
        with torch.no_grad():
            mu, log_var = self.vae_encoder(st_data)
            spot_embeddings = mu
        
        # GAT前向传播
        gat_outputs = self.gat_model(
            spot_embeddings=spot_embeddings,
            spatial_coords=spatial_coords,
            celltype_prototypes=self.celltype_prototypes
        )
        
        # 计算损失
        loss_outputs = self.loss_fn(
            predicted_proportions=gat_outputs['cell_proportions'],
            attention_scores=gat_outputs['attention_scores'],
            celltype_expressions=celltype_expressions,
            target_expressions=st_data
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss_outputs['total_loss'].backward()
        optimizer.step()
        
        return {
            'total_loss': loss_outputs['total_loss'].item(),
            'recon_loss': loss_outputs['recon_loss'].item(),
            'proportion_reg': loss_outputs['proportion_reg'].item(),
            'sparsity_reg': loss_outputs['sparsity_reg'].item()
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'gat_state_dict': self.gat_model.state_dict(),
            'celltype_prototypes': self.celltype_prototypes,
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'stage1_model_path': self.stage1_model_path
        }, filepath)
        
        print(f"💾 第二阶段模型已保存: {filepath}")

def main():
    """主函数 - 第二阶段框架测试"""
    print("🚀 第二阶段GAT解卷积框架")
    print("="*60)
    
    # 初始化训练器
    trainer = Stage2GATDeconvolution(
        stage1_model_path="./simple_sc_results/best_vae.pth",
        output_dir="./simple_sc_results/stage2"
    )
    
    # 加载第一阶段组件
    trainer.load_stage1_components()
    
    # 示例：加载一些SC数据计算原型
    print("\n📊 示例流程:")
    print("   1. ✅ VAE Encoder已加载并冻结")
    print("   2. 🔄 计算细胞类型原型（需要SC数据）")
    print("   3. 🔄 构建GAT解卷积模型")
    print("   4. 🔄 准备ST数据")
    print("   5. 🔄 训练GAT模型")
    
    print("="*60)
    print("🎯 第二阶段框架搭建完成!")
    print("   下一步: 提供SC数据计算原型，然后开始训练")

if __name__ == "__main__":
    main()