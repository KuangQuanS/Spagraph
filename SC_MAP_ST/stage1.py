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

def compute_top_marker_genes(adata, top_n=100, min_fold_change=1.5, min_pct=0.25):
    """
    计算每个细胞类型的top marker基因
    """
    print(f"计算每个细胞类型的top {top_n} marker基因...")
    
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
    
    print(f"各细胞类型marker基因:")
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
            print(f"{cell_type}: {len(selected_genes)} 个基因")
    
    print(f"总计: {len(marker_genes)} 个marker基因")
    
    # 恢复原始数据
    adata.X = adata_backup.X
    adata.var = adata_backup.var
    
    return sorted(list(marker_genes))

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
                
                print(f" SC: {sc_adata.shape}, 细胞类型: {len(sc_adata.obs['cell_type'].unique())}")
                print(f" ST: {st_adata.shape}")
                
                sc_data_list.append(sc_adata)
                st_data_list.append(st_adata)
                valid_samples.append(sample)
            else:
                print(f" 未找到完整数据: {sample}")
        
        # 合并SC数据
        print(f"   合并 {len(sc_data_list)} 个SC样本...")
        combined_sc = ad.concat(sc_data_list, axis=0, join='outer', 
                                keys=valid_samples, index_unique='-')
        
        # 合并ST数据
        print(f"   合并 {len(st_data_list)} 个ST样本...")
        combined_st = ad.concat(st_data_list, axis=0, join='outer', 
                                keys=valid_samples, index_unique='-')
        
        print(f"   SC总计: {combined_sc.shape}")
        print(f"   ST总计: {combined_st.shape}")
        print(f"   细胞类型: {combined_sc.obs['cell_type'].unique()}")
        
        return combined_sc, combined_st, valid_samples

    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100) -> Tuple:
        """基于marker基因准备训练数据"""

        # 1. 计算marker基因
        self.marker_genes = compute_top_marker_genes(sc_adata.copy(), top_n=top_n_per_type)
        
        # 2. 处理SC数据 (normalized)
        print("处理SC数据 (normalized)...")
        sc_subset = sc_adata[:, sc_adata.var.index.isin(self.marker_genes)].copy()
        
        # SC标准化
        sc.pp.normalize_total(sc_subset, target_sum=1e4)
        sc.pp.log1p(sc_subset)
        
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_subset.obs['cell_type'].values
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"SC数据: {sc_X.shape}")
        print(f"细胞类型: {len(self.label_encoder.classes_)}")
        
        # 3. 处理ST数据 (counts)
        print("   处理ST数据 (counts)...")
        # 提取marker基因
        available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
        st_subset = st_adata[:, available_genes].copy()
        
        # ST使用原始counts
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        print(f"ST数据: {st_X.shape}, 可用基因: {len(available_genes)}/{len(self.marker_genes)}")
        
        # 4. 确保SC和ST特征维度一致
        # 取SC和ST共有的marker基因
        final_genes = [g for g in self.marker_genes 
                      if g in sc_subset.var.index and g in st_subset.var.index]
        
        # 获取基因索引并对齐数据（注意要基于已筛选的数据）
        sc_gene_indices = [list(sc_subset.var.index).index(g) for g in final_genes]
        st_gene_indices = [list(st_subset.var.index).index(g) for g in final_genes]
        
        # 对齐数据到共有基因
        sc_X_final = sc_X[:, sc_gene_indices]
        st_X_final = st_X[:, st_gene_indices]
        
        print(f"   最终特征维度: {len(final_genes)}")
        print(f"   SC最终: {sc_X_final.shape}")
        print(f"   ST最终: {st_X_final.shape}")
        
        # 5. 合并SC和ST数据进行训练
        # 分割SC数据
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.2, stratify=sc_y, random_state=42
        )
        
        # 分割ST数据 (ST数据没有标签，所以不需要stratify)
        st_train, st_test = train_test_split(
            st_X_final, test_size=0.2, random_state=42
        )
        
        # 合并训练集和测试集
        train_X = np.vstack([sc_train, st_train])
        test_X = np.vstack([sc_test, st_test])
        
        # 创建模态标签 (0=SC, 1=ST)
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
        genes_file = f"{self.output_dir}/marker_genes.txt"
        with open(genes_file, 'w') as f:
            for gene in self.genes:
                f.write(f"{gene}\n")
        print(f"Marker基因已保存: {genes_file} ({len(self.genes)}个基因)")

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
    
    def run_stage1_training(self, top_n_per_type=100, batch_size=256, n_epochs=100, 
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
            sc_adata, st_adata, top_n_per_type=top_n_per_type
        )
        
        # 3. 构建VAE
        input_dim = len(self.genes)
        self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        
        # 4. 训练VAE
        best_loss = self.train_vae(train_X, test_X, train_modality, test_modality,
                                  batch_size=batch_size, n_epochs=n_epochs, lr=lr, beta=beta)
        
        # 5. 保存最终模型
        self.save_vae(f"{self.output_dir}/final_vae.pth")
        
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
    parser = argparse.ArgumentParser(description='Stage 1: VAE Training for SC-ST Integration')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default="/home/maweicheng/ST_Graduation_Project/database",
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="./stage1_results",
                       help='输出目录路径')
    
    # 模型参数
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='每个细胞类型的marker基因数')
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
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        beta=args.beta,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim
    )
    
if __name__ == "__main__":
    main()