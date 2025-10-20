#!/usr/bin/env python3
"""
简化版SC_MAP_ST - 第一阶段：VAE训练
使用每个细胞类型的前100个marker基因训练VAE
损失函数：重建损失 + KL散度
"""

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
import warnings
warnings.filterwarnings('ignore')

class MultiModalDataset(Dataset):
    """多模态数据集 - SC + ST"""
    def __init__(self, sc_data, st_data, sc_labels=None):
        self.sc_data = torch.FloatTensor(sc_data)
        self.st_data = torch.FloatTensor(st_data) if st_data is not None else None
        self.sc_labels = torch.LongTensor(sc_labels) if sc_labels is not None else None
        
        # 合并数据，SC在前，ST在后
        if self.st_data is not None:
            self.data = torch.cat([self.sc_data, self.st_data], dim=0)
            self.modality = torch.cat([
                torch.zeros(len(self.sc_data)), 
                torch.ones(len(self.st_data))
            ]).long()  # 0=SC, 1=ST
        else:
            self.data = self.sc_data
            self.modality = torch.zeros(len(self.sc_data)).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.modality[idx]

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

def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """
    VAE损失函数：重建损失 + KL散度
    
    Args:
        recon_x: 重建输出
        x: 原始输入
        mu: 编码器输出的均值
        log_var: 编码器输出的对数方差
        beta: KL散度权重
    
    Returns:
        总损失, 重建损失, KL散度
    """
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div

def compute_top_marker_genes(adata, top_n=100, min_fold_change=1.5, min_pct=0.25):
    """
    计算每个细胞类型的top marker基因
    
    Args:
        adata: AnnData对象
        top_n: 每个细胞类型选择的基因数
        min_fold_change: 最小fold change
        min_pct: 最小表达细胞比例
        
    Returns:
        marker基因集合
    """
    print(f"🧬 计算每个细胞类型的top {top_n} marker基因...")
    
    # 备份原始数据
    adata_backup = adata.copy()
    
    # 标准化用于差异表达分析
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 计算差异表达基因
    sc.tl.rank_genes_groups(
        adata, 
        'cell_type', 
        method='wilcoxon',
        key_added='rank_genes_groups',
        n_genes=top_n * 2  # 计算更多基因以便筛选
    )
    
    # 提取marker基因
    marker_genes = set()
    result = adata.uns['rank_genes_groups']
    
    print(f"   各细胞类型marker基因:")
    for cell_type in adata.obs['cell_type'].unique():
        if cell_type in result['names'].dtype.names:
            # 获取基因名、分数和p值
            genes = result['names'][cell_type]
            scores = result['scores'][cell_type]
            pvals = result['pvals_adj'][cell_type]
            logfoldchanges = result['logfoldchanges'][cell_type]
            
            # 筛选显著且高表达的基因
            selected_genes = []
            for i in range(len(genes)):
                if (pvals[i] < 0.05 and 
                    scores[i] > 0 and 
                    logfoldchanges[i] >= np.log2(min_fold_change)):
                    selected_genes.append(genes[i])
                    
                if len(selected_genes) >= top_n:
                    break
            
            marker_genes.update(selected_genes)
            print(f"     {cell_type}: {len(selected_genes)} 个基因")
    
    print(f"   总计: {len(marker_genes)} 个唯一marker基因")
    
    # 恢复原始数据
    adata.X = adata_backup.X
    adata.var = adata_backup.var
    
    return sorted(list(marker_genes))
class SimpleVAEMapper:
    """简化版VAE映射器 - 第一阶段"""
    
    def __init__(self, 
                 data_dir="/home/maweicheng/ST_Graduation_Project/database",
                 output_dir="./simple_sc_results",
                 device=None):
        """
        初始化
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            device: 计算设备
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 简化版VAE映射器初始化")
        print(f"   数据目录: {data_dir}")
        print(f"   输出目录: {output_dir}")
        print(f"   设备: {self.device}")
        
        # 模型组件
        self.vae = None
        self.label_encoder = None
        self.marker_genes = None
        
    def load_wu_data(self) -> Tuple[ad.AnnData, List[ad.AnnData], List[str]]:
        """
        加载Wu数据集的SC和ST数据
        
        Returns:
            合并的sc_adata, st_adata_list, 样本列表
        """
        print("📂 加载Wu数据集...")
        
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
                
                print(f"     SC: {sc_adata.shape}, 细胞类型: {len(sc_adata.obs['cell_type'].unique())}")
                print(f"     ST: {st_adata.shape}")
                
                sc_data_list.append(sc_adata)
                st_data_list.append(st_adata)
                valid_samples.append(sample)
            else:
                print(f"   ⚠️ 未找到完整数据: {sample}")
        
        # 合并SC数据
        if len(sc_data_list) > 0:
            print(f"   合并 {len(sc_data_list)} 个SC样本...")
            combined_sc = ad.concat(sc_data_list, axis=0, join='outer', 
                                  keys=valid_samples, index_unique='-')
            
            print(f"   SC总计: {combined_sc.shape}")
            print(f"   细胞类型: {combined_sc.obs['cell_type'].unique()}")
            
            return combined_sc, st_data_list, valid_samples
        else:
            raise ValueError("未找到任何有效数据!")
    
    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata_list: List[ad.AnnData], 
                               top_n_per_type: int = 100) -> Tuple:
        """
        基于marker基因准备训练数据
        
        Args:
            sc_adata: SC数据
            st_adata_list: ST数据列表
            top_n_per_type: 每个细胞类型的top基因数
            
        Returns:
            训练数据
        """
        print("🔧 基于marker基因准备训练数据...")
        
        # 1. 计算marker基因
        self.marker_genes = compute_top_marker_genes(sc_adata.copy(), top_n=top_n_per_type)
        
        # 2. 处理SC数据 (normalized)
        print("   处理SC数据 (normalized)...")
        sc_subset = sc_adata[:, sc_adata.var.index.isin(self.marker_genes)].copy()
        
        # SC标准化
        sc.pp.normalize_total(sc_subset, target_sum=1e4)
        sc.pp.log1p(sc_subset)
        
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_subset.obs['cell_type'].values
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"     SC数据: {sc_X.shape}")
        print(f"     细胞类型: {len(self.label_encoder.classes_)}")
        
        # 3. 处理ST数据 (counts)
        print("   处理ST数据 (counts)...")
        st_X_list = []
        
        for i, st_adata in enumerate(st_adata_list):
            # 提取marker基因
            available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
            st_subset = st_adata[:, available_genes].copy()
            
            # ST使用原始counts
            st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
            st_X_list.append(st_X)
            
            print(f"     ST样本{i}: {st_X.shape}, 可用基因: {len(available_genes)}/{len(self.marker_genes)}")
        
        # 合并所有ST数据
        st_X_combined = np.vstack(st_X_list)
        print(f"   ST合并后: {st_X_combined.shape}")
        
        # 4. 确保SC和ST特征维度一致
        final_genes = [g for g in self.marker_genes if g in sc_subset.var.index]
        sc_final_indices = [i for i, g in enumerate(sc_subset.var.index) if g in final_genes]
        
        # 重新对齐数据
        sc_X_final = sc_X[:, sc_final_indices]
        
        # 对ST数据也进行对齐
        st_X_final_list = []
        for st_adata in st_adata_list:
            available_gene_indices = []
            for gene in final_genes:
                if gene in st_adata.var.index:
                    idx = list(st_adata.var.index).index(gene)
                    available_gene_indices.append(idx)
                else:
                    available_gene_indices.append(-1)  # 缺失基因用-1标记
            
            st_subset = st_adata[:, [g for g in final_genes if g in st_adata.var.index]].copy()
            st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
            
            # 对于缺失基因，用0填充
            if st_X.shape[1] < len(final_genes):
                padding = np.zeros((st_X.shape[0], len(final_genes) - st_X.shape[1]))
                st_X = np.hstack([st_X, padding])
            
            st_X_final_list.append(st_X)
        
        st_X_final = np.vstack(st_X_final_list)
        
        print(f"   最终特征维度: {len(final_genes)}")
        print(f"   SC最终: {sc_X_final.shape}")
        print(f"   ST最终: {st_X_final.shape}")
        
        # 5. 分割数据
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.2, stratify=sc_y, random_state=42
        )
        
        print(f"   训练集: SC {sc_train.shape} + ST {st_X_final.shape}")
        print(f"   测试集: SC {sc_test.shape}")
        
        # 保存基因列表
        self.genes = final_genes
        genes_file = f"{self.output_dir}/marker_genes.txt"
        with open(genes_file, 'w') as f:
            for gene in self.genes:
                f.write(f"{gene}\n")
        print(f"   💾 Marker基因已保存: {genes_file} ({len(self.genes)}个基因)")
        
        return sc_train, sc_test, st_X_final, y_train, y_test
    
    def build_vae(self, input_dim: int):
        """构建VAE模型"""
        print("🏗️ 构建VAE模型...")
        
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=[512, 256],
            latent_dim=128,
            dropout=0.2
        ).to(self.device)
        
        print(f"   VAE: {input_dim} -> 128")
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"   参数量: {vae_params:,}")
    
    def train_vae(self, sc_train, st_data, sc_test, y_test,
                  batch_size=256, n_epochs=100, lr=1e-3, beta=1.0):
        """训练VAE"""
        print("🚀 开始VAE训练...")
        
        # 合并训练数据
        train_data = np.vstack([sc_train, st_data])
        print(f"   训练数据: {train_data.shape} (SC: {sc_train.shape[0]}, ST: {st_data.shape[0]})")
        
        # 数据加载器
        train_dataset = MultiModalDataset(sc_train, st_data)
        test_dataset = MultiModalDataset(sc_test, None)
        
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
                    self.save_vae(f"{self.output_dir}/best_vae.pth")
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
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'input_dim': len(self.genes),
            'latent_dim': self.vae.latent_dim
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
    
    def run_stage1_training(self):
        """运行第一阶段VAE训练"""
        print("🚀 开始第一阶段训练: VAE (SC + ST, Marker基因)")
        print("="*60)
        
        # 1. 加载数据
        sc_adata, st_adata_list, samples = self.load_wu_data()
        
        # 2. 基于marker基因准备数据
        sc_train, sc_test, st_data, y_train, y_test = self.prepare_marker_gene_data(
            sc_adata, st_adata_list, top_n_per_type=100
        )
        
        # 3. 构建VAE
        input_dim = len(self.genes)
        self.build_vae(input_dim)
        
        # 4. 训练VAE
        best_loss = self.train_vae(sc_train, st_data, sc_test, y_test)
        
        # 5. 保存最终模型
        self.save_vae(f"{self.output_dir}/final_vae.pth")
        
        print("="*60)
        print("🎉 第一阶段VAE训练完成!")
        print(f"   🎯 最佳测试损失: {best_loss:.4f}")
        print(f"   🧬 Marker基因数: {len(self.genes)}")
        print(f"   📊 细胞类型数: {len(self.label_encoder.classes_)}")
        print(f"   💾 模型保存至: {self.output_dir}")
        print("="*60)
        
        return {
            'best_loss': best_loss,
            'n_genes': len(self.genes),
            'n_cell_types': len(self.label_encoder.classes_),
            'model_path': f"{self.output_dir}/final_vae.pth",
            'samples': samples,
            'cell_types': list(self.label_encoder.classes_)
        }

def main():
    """主函数"""
    # 创建VAE映射器
    mapper = SimpleVAEMapper(
        data_dir="/home/maweicheng/ST_Graduation_Project/database",
        output_dir="./simple_sc_results"
    )
    
    # 运行第一阶段VAE训练
    results = mapper.run_stage1_training()
    
    print("\n📋 训练结果摘要:")
    print(f"   样本数量: {len(results['samples'])}")
    print(f"   🎯 最佳测试损失: {results['best_loss']:.4f}")
    print(f"   🧬 Marker基因数: {results['n_genes']}")
    print(f"   � 细胞类型数: {results['n_cell_types']}")
    print(f"   🧬 细胞类型: {results['cell_types']}")

if __name__ == "__main__":
    main()