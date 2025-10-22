import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
import argparse
import warnings
warnings.filterwarnings('ignore')

# 导入统一的模型定义
from model import VAE, vae_loss_function

def compute_clusters_and_marker_genes(adata, top_n=100, min_fold_change=1.5, resolution=0.5, save_path=None):
    """
    计算聚类并提取每个cluster的top marker基因
    """
    print(f"🔍 进行聚类分析...")
    
    # 备份原始数据
    adata_backup = adata.copy()
    
    # 预处理：标准化和主成分分析
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # 主成分分析
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # 构建邻接图
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Leiden聚类
    sc.tl.leiden(adata, resolution=resolution)
    
    print(f"📊 聚类结果: {len(adata.obs['leiden'].unique())} 个clusters")
    for cluster in sorted(adata.obs['leiden'].unique()):
        count = (adata.obs['leiden'] == cluster).sum()
        print(f"   Cluster {cluster}: {count} 细胞")
    
    # 恢复到原始基因集进行marker分析
    adata_full = adata_backup.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)
    
    # 将聚类结果转移到完整数据
    adata_full.obs['leiden'] = adata.obs['leiden'].copy()
    
    # 计算每个cluster的marker基因
    sc.tl.rank_genes_groups(
        adata_full, 
        'leiden', 
        method='wilcoxon',
        key_added='rank_genes_groups',
        n_genes=top_n * 2
    )
    
    # 提取marker基因
    marker_genes = set()
    result = adata_full.uns['rank_genes_groups']
    
    print(f"🧬 各cluster的marker基因:")
    for cluster in sorted(adata_full.obs['leiden'].unique()):
        if cluster in result['names'].dtype.names:
            genes = result['names'][cluster]
            scores = result['scores'][cluster]
            pvals = result['pvals_adj'][cluster]
            logfoldchanges = result['logfoldchanges'][cluster]
            
            selected_genes = []
            for i in range(len(genes)):
                if (pvals[i] < 0.05 and 
                    scores[i] > 0 and 
                    logfoldchanges[i] >= np.log2(min_fold_change)):
                    selected_genes.append(genes[i])
                    
                if len(selected_genes) >= top_n:
                    break
            
            marker_genes.update(selected_genes)
            print(f"   Cluster {cluster}: {len(selected_genes)} 个基因")
    
    print(f"🎯 总计: {len(marker_genes)} 个marker基因")
    
    # 返回聚类信息和marker基因
    return sorted(list(marker_genes)), adata_full.obs['leiden'].copy()

#--------------------------主模块-----------------------------
class coEncoder:
    def __init__(self, 
                 data_dir="/home/maweicheng/ST_Graduation_Project/database",
                 output_dir="./stage1_results",
                 device=None):

        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f" use device: {self.device}")

        # 模型组件
        self.vae = None
        self.label_encoder = None
        self.marker_genes = None
        
    def load_data(self) -> Tuple[ad.AnnData, ad.AnnData, List[str]]:

        print("加载数据集...")
        
        wu_dir = os.path.join(self.data_dir, "Wu")

        sample_dirs = [d for d in os.listdir(wu_dir) 
                      if os.path.isdir(os.path.join(wu_dir, d))]
        sample_dirs.sort()
        
        print(f"   发现样本: {sample_dirs}")
        
        sc_data_list = []
        st_data_list = []
        valid_samples = []
        
        for sample in sample_dirs:
            sample_dir = os.path.join(wu_dir, sample)
            sc_file = os.path.join(sample_dir, f"{sample}_SC.h5ad")
            st_file = os.path.join(sample_dir, f"{sample}_ST.h5ad")
            
            if os.path.exists(sc_file) and os.path.exists(st_file):
                print(f"   加载 {sample}...")
                
                # 加载SC数据
                sc_adata = sc.read_h5ad(sc_file)
                sc_adata.obs['sample'] = sample
                sc_adata.obs['modality'] = 'SC'
                
                # 加载ST数据
                st_adata = sc.read_h5ad(st_file)
                st_adata.obs['sample'] = sample
                st_adata.obs['modality'] = 'ST'
                
                print(f" SC: {sc_adata.shape}")
                print(f" ST: {st_adata.shape}")
                
                sc_data_list.append(sc_adata)
                st_data_list.append(st_adata)
                valid_samples.append(sample)
            else:
                print(f" 未找到完整数据: {sample}")
        
        # 合并SC数据 - 使用inner join确保只保留所有样本共有的基因
        print(f"   合并 {len(sc_data_list)} 个SC样本...")
        combined_sc = ad.concat(sc_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        
        # 合并ST数据 - 使用inner join确保只保留所有样本共有的基因  
        print(f"   合并 {len(st_data_list)} 个ST样本...")
        combined_st = ad.concat(st_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        
        print(f"   SC总计: {combined_sc.shape}")
        print(f"   ST总计: {combined_st.shape}")
        # print(f"   细胞类型: {combined_sc.obs['cell_type'].unique()}")  # 将使用聚类代替
        
        return combined_sc, combined_st, valid_samples

    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100, resolution: float = 0.5) -> Tuple:
        """基于marker基因准备训练数据"""

        # 1. 计算聚类和marker基因  
        print("📋 计算聚类和marker基因...")
        cluster_save_path = f"{self.output_dir}/marker_genes.txt"
        self.marker_genes, sc_clusters = compute_clusters_and_marker_genes(
            sc_adata.copy(), 
            top_n=top_n_per_type, 
            resolution=resolution,
            save_path=cluster_save_path
        )
        
        # 保存聚类信息和分辨率
        self.sc_clusters = sc_clusters
        self.resolution = resolution
        
        # 2. 处理SC数据 (提取marker基因后标准化)
        print("📋 处理SC数据...")
        sc_subset = sc_adata[:, sc_adata.var.index.isin(self.marker_genes)].copy()
        
        # SC标准化
        sc.pp.normalize_total(sc_subset, target_sum=1e4)
        sc.pp.log1p(sc_subset)
        
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_clusters.values  # 使用聚类标签
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"   SC数据: {sc_X.shape}")
        print(f"   聚类数: {len(self.label_encoder.classes_)} 个")
        
        # 3. 处理ST数据
        print("📋 处理ST数据...")
        available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
        st_subset = st_adata[:, available_genes].copy()
        
        sc.pp.log1p(st_subset)
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        print(f"   ST数据: {st_X.shape}, 可用基因: {len(available_genes)}/{len(self.marker_genes)}")
        
        # 4. 确保SC和ST特征维度一致
        final_genes = [g for g in self.marker_genes 
                      if g in sc_subset.var.index and g in st_subset.var.index]
        
        sc_gene_indices = [list(sc_subset.var.index).index(g) for g in final_genes]
        st_gene_indices = [list(st_subset.var.index).index(g) for g in final_genes]
        
        sc_X_final = sc_X[:, sc_gene_indices]
        st_X_final = st_X[:, st_gene_indices]
        
        print(f"   最终基因数: {len(final_genes)}")
        
        # 5. 分割数据
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.2, stratify=sc_y, random_state=42
        )
        
        st_train, st_test = train_test_split(
            st_X_final, test_size=0.2, random_state=42
        )
        
        # 6. 合并训练集和测试集
        train_X = np.vstack([sc_train, st_train])
        test_X = np.vstack([sc_test, st_test])
        
        train_modality = np.concatenate([
            np.zeros(len(sc_train)), 
            np.ones(len(st_train))
        ])

        test_modality = np.concatenate([
            np.zeros(len(sc_test)), 
            np.ones(len(st_test))
        ])
        
        print(f"   训练集: {train_X.shape} (SC: {len(sc_train)}, ST: {len(st_train)})")
        print(f"   测试集: {test_X.shape} (SC: {len(sc_test)}, ST: {len(st_test)})")
        
        # 保存基因列表
        self.genes = final_genes
        genes_file = f"{self.output_dir}/final_genes.txt"
        with open(genes_file, 'w') as f:
            for gene in self.genes:
                f.write(f"{gene}\n")

        return train_X, test_X, train_modality, test_modality, y_train, y_test
    
    def build_vae(self, input_dim: int, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        """构建VAE模型"""
        print("🏗️ 构建VAE模型...")
        
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        ).to(self.device)
        
        print(f"   VAE: {input_dim} -> {latent_dim}")
        print(f"   隐藏层: {hidden_dims}")
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"   参数量: {vae_params:,}")
    
    def train_vae(self, train_X, test_X, train_modality, test_modality,
                  batch_size=256, n_epochs=100, lr=1e-3, beta=1.0):
        """训练VAE"""

        print("开始VAE训练...")
        print(f"   训练数据: {train_X.shape} (SC: {sum(train_modality==0)}, ST: {sum(train_modality==1)})")
        print(f"   测试数据: {test_X.shape} (SC: {sum(test_modality==0)}, ST: {sum(test_modality==1)})")

        class SimpleDataset(Dataset):
            def __init__(self, X, modality):
                self.X = torch.FloatTensor(X)
                self.modality = torch.LongTensor(modality)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.modality[idx]

        # 数据加载器
        train_dataset = SimpleDataset(train_X, train_modality)
        test_dataset = SimpleDataset(test_X, test_modality)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
        
        # 训练历史
        train_losses = []
        test_losses = []
        recon_losses = []
        kl_losses = []
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(n_epochs):
            # 训练
            self.vae.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            for batch_data, batch_modality in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE前向传播
                recon_data, mu, log_var, z = self.vae(batch_data)
                
                # 计算损失
                total_loss, recon_loss, kl_div = vae_loss_function(
                    recon_data, batch_data, mu, log_var, beta=beta
                )
                
                # 归一化损失
                total_loss = total_loss / len(batch_data)
                recon_loss = recon_loss / len(batch_data)
                kl_div = kl_div / len(batch_data)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_div.item()
            
            avg_loss = epoch_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            
            # 评估
            if (epoch + 1) % 5 == 0:
                test_loss = self.evaluate_vae(test_loader, beta)
                test_losses.append(test_loss)
                
                scheduler.step(test_loss)
                
                print(f"Epoch {epoch+1:3d}: Train Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, "
                      f"KL={avg_kl:.4f}), Test Loss={test_loss:.4f}")
                
                # 保存最佳模型
                if test_loss < best_loss:
                    best_loss = test_loss
                    # 这里不保存best_vae.pth，等到聚类中心计算完后再保存
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # 绘制训练曲线
        self.plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses)
        
        print(f"✅ VAE训练完成! 最佳测试损失: {best_loss:.4f}")
        return best_loss
    
    def evaluate_vae(self, test_loader, beta=1.0):
        """评估VAE"""
        self.vae.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)
                
                recon_data, mu, log_var, z = self.vae(batch_data)
                loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
                total_loss += loss.item() / len(batch_data)
        
        return total_loss / len(test_loader)
    
    def plot_vae_training_curves(self, train_losses, test_losses, recon_losses, kl_losses):
        """绘制VAE训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        ax1.plot(train_losses, label='Train')
        if len(test_losses) > 0:
            test_epochs = range(5, len(train_losses)+1, 5)
            if len(test_epochs) == len(test_losses):
                ax1.plot(test_epochs, test_losses, label='Test')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 重建损失
        ax2.plot(recon_losses, 'g-')
        ax2.set_title('Reconstruction Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # KL散度
        ax3.plot(kl_losses, 'r-')
        ax3.set_title('KL Divergence')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('KL Div')
        ax3.grid(True)
        
        # 损失组件对比
        ax4.plot(recon_losses, label='Reconstruction', color='green')
        ax4.plot(kl_losses, label='KL Divergence', color='red')
        ax4.set_title('Loss Components')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/vae_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_vae(self, filepath):
        """保存VAE模型"""
        # 检查聚类信息是否存在
        cluster_prototypes = getattr(self, 'cluster_prototypes', None)
        cluster_expressions = getattr(self, 'cluster_expressions', None)
        cluster_expressions_full = getattr(self, 'cluster_expressions_full', None)  # 全基因版本
        
        print(f"💾 保存模型到: {filepath}")
        if cluster_prototypes is not None:
            print(f"   ✅ 包含聚类中心: {len(cluster_prototypes)} 个聚类")
        else:
            print(f"   ⚠️  缺少聚类中心")
            
        if cluster_expressions is not None:
            print(f"   ✅ 包含聚类表达谱 (marker基因): {len(cluster_expressions)} 个聚类")
        else:
            print(f"   ⚠️  缺少聚类表达谱")
        
        if cluster_expressions_full is not None:
            print(f"   ✅ 包含聚类表达谱 (全基因): {len(cluster_expressions_full)} 个聚类")
        else:
            print(f"   ⚠️  缺少聚类全基因表达谱")
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'input_dim': len(self.genes),
            'latent_dim': self.vae.latent_dim,
            'sc_clusters': getattr(self, 'sc_clusters', None),  # 保存聚类信息
            'resolution': getattr(self, 'resolution', 0.5),    # 保存分辨率参数
            'cluster_prototypes': cluster_prototypes,  # 保存聚类中心
            'cluster_expressions': cluster_expressions,  # 保存聚类表达谱 (marker基因)
            'cluster_expressions_full': cluster_expressions_full,  # 保存聚类表达谱 (全基因)
            'all_genes': getattr(self, 'all_genes', None)  # 保存所有基因列表
        }, filepath)
    
    def load_vae(self, filepath):
        """加载VAE模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        
        print(f"📂 VAE模型已加载: {filepath}")
    
    def run_stage1_training(self, top_n_per_type=100, resolution=0.5, batch_size=256, n_epochs=100, 
                           lr=1e-3, beta=1.0, hidden_dims=[512, 256], latent_dim=128):
        """运行第一阶段VAE训练"""
        print("开始第一阶段训练: VAE (SC + ST, Marker基因)")
        print("="*60)
        print(f"   参数配置:")
        print(f"   - 每类型marker基因数: {top_n_per_type}")
        print(f"   - 批次大小: {batch_size}")
        print(f"   - 训练轮数: {n_epochs}")
        print(f"   - 学习率: {lr}")
        print(f"   - Beta (KL权重): {beta}")
        print(f"   - 隐藏层维度: {hidden_dims}")
        print(f"   - 潜在维度: {latent_dim}")
        print("="*60)
        
        # 1. 加载数据
        sc_adata, st_adata, samples = self.load_data()
        
        # 2. 基于marker基因准备数据
        train_X, test_X, train_modality, test_modality, y_train, y_test = self.prepare_marker_gene_data(
            sc_adata, st_adata, top_n_per_type=top_n_per_type, resolution=resolution
        )
        
        # 3. 构建VAE
        input_dim = len(self.genes)
        self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        
        # 4. 训练VAE
        best_loss = self.train_vae(train_X, test_X, train_modality, test_modality,
                                  batch_size=batch_size, n_epochs=n_epochs, lr=lr, beta=beta)
        
        # 保存训练数据以便计算聚类中心
        self.train_X = train_X
        self.train_modality = train_modality  
        self.y_train = y_train
        
        # 5. 计算并保存聚类中心
        print("🔄 计算聚类中心...")
        
        # 使用训练数据计算聚类中心（这些数据已经过marker基因筛选和标准化）
        # 获取SC训练数据部分
        sc_train_mask = train_modality == 0  # SC数据的mask
        sc_train_data = train_X[sc_train_mask]  # SC训练数据
        sc_train_labels = y_train  # SC训练标签
        
        print(f"   用于计算聚类中心的SC数据: {sc_train_data.shape}")
        print(f"   聚类数: {len(np.unique(sc_train_labels))}")
        
        # 使用训练好的VAE计算embeddings
        self.vae.eval()
        with torch.no_grad():
            # 分批处理以避免内存问题
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(sc_train_data), batch_size):
                batch_data = sc_train_data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                
                # 获取潜在表示
                mu, log_var = self.vae.encoder(batch_tensor)
                all_embeddings.append(mu.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
        
        # 计算每个聚类的中心和表达谱
        cluster_prototypes = {}
        cluster_expressions = {}
        cluster_expressions_full = {}  # 保存全基因表达
        
        for cluster_id in np.unique(sc_train_labels):
            cluster_mask = sc_train_labels == cluster_id
            cluster_cells = np.sum(cluster_mask)
            
            # 计算聚类中心（潜在空间）
            cluster_center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_prototypes[cluster_id] = cluster_center
            
            # 计算聚类表达谱（marker基因）
            cluster_expression = np.mean(sc_train_data[cluster_mask], axis=0)
            cluster_expressions[cluster_id] = cluster_expression
            
            print(f"     Cluster {cluster_id}: {cluster_cells} cells")
        
        # 计算全基因表达（需要从原始SC数据中提取，并进行相同的预处理）
        print("   📊 计算全基因聚类表达谱...")
        # 获取原始SC数据，进行相同的预处理
        sc_adata_full = self.load_data()[0].copy()
        
        # 保存所有基因列表（用于后续输出）
        self.all_genes = list(sc_adata_full.var.index)
        print(f"      总基因数: {len(self.all_genes)}")
        
        # 进行相同的预处理（归一化和log变换）
        sc.pp.normalize_total(sc_adata_full, target_sum=1e4)
        sc.pp.log1p(sc_adata_full)
        sc_full_X = sc_adata_full.X.toarray() if hasattr(sc_adata_full.X, 'toarray') else sc_adata_full.X
        
        for cluster_id in np.unique(sc_train_labels):
            # 找到该聚类对应的所有单细胞
            cluster_indices = np.where(sc_train_labels == cluster_id)[0]
            
            # 获取这些单细胞对应的完整基因表达（预处理后）
            cluster_cells_full_expr = sc_full_X[cluster_indices]
            
            # 计算平均表达
            cluster_expr_full = np.mean(cluster_cells_full_expr, axis=0)
            cluster_expressions_full[cluster_id] = cluster_expr_full
        
        # 保存聚类中心和表达谱
        self.cluster_prototypes = cluster_prototypes
        self.cluster_expressions = cluster_expressions
        self.cluster_expressions_full = cluster_expressions_full  # 保存全基因版本
        print(f"   ✅ 计算完成: {len(cluster_prototypes)} 个聚类中心和表达谱（包含全基因）")
        
        # 6. 保存最终模型
        self.save_vae(f"{self.output_dir}/final_vae.pth")
        
        return {
            'best_loss': best_loss,
            'n_genes': len(self.genes),
            'n_clusters': len(self.label_encoder.classes_),
            'model_path': f"{self.output_dir}/final_vae.pth",
            'samples': samples,
            'clusters': list(self.label_encoder.classes_)
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 1: VAE Training for SC-ST Integration')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default="/home/maweicheng/ST_Graduation_Project/database",
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="./stage1_results",
                       help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='每个聚类的marker基因数')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden聚类分辨率')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='VAE隐藏层维度')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='VAE潜在空间维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL散度权重 (β-VAE)')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu，None为自动选择)')
    
    args = parser.parse_args()
    

    # 创建VAE映射器
    co_encoder = coEncoder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 运行第一阶段VAE训练
    results = co_encoder.run_stage1_training(
        top_n_per_type=args.top_n_per_type,
        resolution=args.resolution,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        beta=args.beta,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim
    )
    
if __name__ == "__main__":
    main()