"""
Stage 1 Multi-Modal Training: SC-ST Integration with Image

三阶段训练策略：
1. Phase 1: 训练 SC VAE（单细胞基因表达）
2. Phase 2: 训练 ST Multi-Modal VAE（空转基因表达 + 图像 patch）
3. Phase 3: SC-ST 对齐（通过 KL 散度或对比学习）
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import unified model definitions
from model import VAE, MultiModalVAE, vae_loss_function, multimodal_vae_loss, alignment_loss

# 重用 stage1.py 的聚类函数
from stage1 import compute_clusters_and_marker_genes


class STImageDataset(Dataset):
    """ST 数据集 - 包含基因表达和图像 patch"""
    def __init__(self, gene_expression, image_patches, transform=None):
        """
        Args:
            gene_expression: [n_spots, n_genes] 基因表达矩阵
            image_patches: [n_spots, H, W, 3] 图像 patch 数组
            transform: 图像变换
        """
        self.gene_expression = torch.FloatTensor(gene_expression)
        self.image_patches = image_patches
        self.transform = transform
        
    def __len__(self):
        return len(self.gene_expression)
    
    def __getitem__(self, idx):
        gene = self.gene_expression[idx]
        
        # 处理图像
        image = self.image_patches[idx]  # [H, W, 3]
        if self.transform:
            # 转换为 PIL Image
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image_pil)
        else:
            # 默认转换
            image = torch.FloatTensor(image).permute(2, 0, 1)  # [3, H, W]
        
        return gene, image


class SCSTMultiModalTrainer:
    """三阶段多模态训练器"""
    
    def __init__(self, data_dir, output_dir, device=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # 模型组件
        self.sc_vae = None
        self.st_vae = None
        self.label_encoder = None
        self.marker_genes = None
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet 归一化
        ])
    
    def load_data(self):
        """加载数据（重用 stage1 的逻辑）"""
        print("="*60)
        print("Loading datasets...")
        
        wu_dir = os.path.join(self.data_dir, "Wu")
        sample_dirs = [d for d in os.listdir(wu_dir) 
                      if os.path.isdir(os.path.join(wu_dir, d))]
        sample_dirs.sort()
        
        print(f"   Found samples: {sample_dirs}")
        
        sc_data_list = []
        st_data_list = []
        valid_samples = []
        
        for sample in sample_dirs:
            sample_dir = os.path.join(wu_dir, sample)
            
            # 尝试小写和大写文件名
            sc_file = os.path.join(sample_dir, f"{sample}_sc.h5ad")
            st_file = os.path.join(sample_dir, f"{sample}_st.h5ad")
            
            if not os.path.exists(sc_file):
                sc_file = os.path.join(sample_dir, f"{sample}_SC.h5ad")
            if not os.path.exists(st_file):
                st_file = os.path.join(sample_dir, f"{sample}_ST.h5ad")
            
            if os.path.exists(sc_file) and os.path.exists(st_file):
                try:
                    sc_data = sc.read_h5ad(sc_file)
                    st_data = sc.read_h5ad(st_file)
                    
                    sc_data_list.append(sc_data)
                    st_data_list.append(st_data)
                    valid_samples.append(sample)
                    
                    print(f"   ✓ Loaded {sample}: SC {sc_data.shape}, ST {st_data.shape}")
                except Exception as e:
                    print(f"   ✗ Failed to load {sample}: {e}")
            else:
                print(f"   ✗ {sample}: Files not found (SC: {os.path.exists(sc_file)}, ST: {os.path.exists(st_file)})")
        
        if len(sc_data_list) == 0 or len(st_data_list) == 0:
            raise ValueError(f"No valid SC-ST data pairs found in {wu_dir}. Please check your data directory.")
        
        # Merge data
        combined_sc = ad.concat(sc_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        combined_st = ad.concat(st_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        
        print(f"   SC total: {combined_sc.shape}")
        print(f"   ST total: {combined_st.shape}")
        
        return combined_sc, combined_st, valid_samples
    
    def extract_image_patches(self, st_adata, patch_size=112):
        """从 ST 数据提取图像 patch
        
        Args:
            st_adata: ST AnnData 对象
            patch_size: patch 大小（像素）
        
        Returns:
            image_patches: [n_spots, patch_size, patch_size, 3]
        """
        print("="*60)
        print(f"Extracting image patches (size={patch_size})...")
        
        # 检查是否有图像数据
        if 'spatial' not in st_adata.uns:
            print("Warning: No spatial information found in ST data!")
            print("Returning dummy patches (will be replaced when you have images)")
            # 返回随机 patch 作为占位符
            n_spots = st_adata.shape[0]
            dummy_patches = np.random.rand(n_spots, patch_size, patch_size, 3)
            return dummy_patches
        
        # 获取第一个样本的图像（这里简化处理，实际应该处理多个样本）
        sample_ids = list(st_adata.uns['spatial'].keys())
        if len(sample_ids) == 0:
            print("Warning: No samples found in spatial data!")
            n_spots = st_adata.shape[0]
            dummy_patches = np.random.rand(n_spots, patch_size, patch_size, 3)
            return dummy_patches
        
        patches = []
        
        for sample_id in sample_ids:
            spatial_info = st_adata.uns['spatial'][sample_id]
            
            # 检查是否有 hires 图像
            if 'images' not in spatial_info or 'hires' not in spatial_info['images']:
                print(f"Warning: No hires image found for sample {sample_id}")
                # 获取该样本的 spot 数量
                sample_mask = st_adata.obs.index.str.startswith(f"{sample_id}-")
                n_sample_spots = sample_mask.sum()
                dummy_patches_sample = np.random.rand(n_sample_spots, patch_size, patch_size, 3)
                patches.append(dummy_patches_sample)
                continue
            
            # 获取图像和缩放因子
            image = spatial_info['images']['hires']  # [H, W, 3]
            scale_factor = spatial_info['scalefactors']['tissue_hires_scalef']
            
            # 获取该样本的空间坐标
            sample_mask = st_adata.obs.index.str.startswith(f"{sample_id}-")
            spatial_coords = st_adata.obsm['spatial'][sample_mask]  # [n_spots, 2]
            
            # 提取 patch
            sample_patches = []
            for coord in spatial_coords:
                # 坐标缩放到 hires 图像
                x, y = coord * scale_factor
                x, y = int(x), int(y)
                
                # 计算 patch 边界
                half_size = patch_size // 2
                x_min = max(0, x - half_size)
                x_max = min(image.shape[1], x + half_size)
                y_min = max(0, y - half_size)
                y_max = min(image.shape[0], y + half_size)
                
                # 提取 patch
                patch = image[y_min:y_max, x_min:x_max]
                
                # 填充到固定大小
                padded_patch = np.zeros((patch_size, patch_size, 3))
                actual_h = y_max - y_min
                actual_w = x_max - x_min
                padded_patch[:actual_h, :actual_w] = patch
                
                sample_patches.append(padded_patch)
            
            patches.append(np.array(sample_patches))
            print(f"   Extracted {len(sample_patches)} patches from {sample_id}")
        
        # 合并所有样本的 patch
        all_patches = np.vstack(patches)
        print(f"   Total patches: {all_patches.shape}")
        
        return all_patches
    
    def phase1_train_sc_vae(self, sc_adata, marker_genes, 
                            batch_size=256, n_epochs=50, lr=1e-3, beta=1.0,
                            hidden_dims=[512, 256], latent_dim=128):
        """Phase 1: 训练 SC VAE"""
        print("\n" + "="*60)
        print("PHASE 1: Training SC VAE")
        print("="*60)
        
        # 准备 SC 数据
        sc_adata_count = sc_adata.copy()
        sc.pp.normalize_total(sc_adata_count, target_sum=1e4)
        sc.pp.log1p(sc_adata_count)
        # 提取 marker genes
        sc_subset = sc_adata_count[:, sc_adata_count.var.index.isin(marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        
        print(f"SC data shape: {sc_X.shape}")
        print(f"SC data (count) min: {np.min(sc_X):.4f}, max: {np.max(sc_X):.4f}")
        
        # 划分训练/测试集
        sc_train, sc_test = train_test_split(sc_X, test_size=0.1, random_state=42)
        
        # 创建 DataLoader
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sc_train))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sc_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 构建 SC VAE
        input_dim = sc_X.shape[1]
        self.sc_vae = VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=0.2
        ).to(self.device)
        
        optimizer = optim.Adam(self.sc_vae.parameters(), lr=lr)
        
        # 训练
        train_losses = []
        test_losses = []
        
        for epoch in range(n_epochs):
            # Train
            self.sc_vae.train()
            train_loss_epoch = 0
            train_recon_epoch = 0
            train_kl_epoch = 0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                x_recon, mu, log_var, z = self.sc_vae(x)
                loss, recon, kl = vae_loss_function(x_recon, x, mu, log_var, beta)
                loss.backward()
                optimizer.step()
                
                train_loss_epoch += loss.item()
                train_recon_epoch += recon.item()
                train_kl_epoch += kl.item()
            
            train_loss_epoch /= len(train_loader.dataset)
            train_recon_epoch /= len(train_loader.dataset)
            train_kl_epoch /= len(train_loader.dataset)
            train_losses.append(train_loss_epoch)
            
            # Test
            self.sc_vae.eval()
            test_loss_epoch = 0
            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(self.device)
                    x_recon, mu, log_var, z = self.sc_vae(x)
                    loss, _, _ = vae_loss_function(x_recon, x, mu, log_var, beta)
                    test_loss_epoch += loss.item()
            
            test_loss_epoch /= len(test_loader.dataset)
            test_losses.append(test_loss_epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train: {train_loss_epoch:.4f} (Recon: {train_recon_epoch:.4f}, KL: {train_kl_epoch:.4f}) | "
                      f"Test: {test_loss_epoch:.4f}")
        
        # 保存模型
        torch.save(self.sc_vae.state_dict(), 
                   os.path.join(self.output_dir, 'sc_vae_phase1.pth'))
        print(f"\n✓ Phase 1 complete! SC VAE saved.")
        
        return train_losses, test_losses
    
    def phase2_train_st_multimodal_vae(self, st_adata, marker_genes, image_patches,
                                       batch_size=128, n_epochs=50, lr=1e-3, beta=1.0,
                                       hidden_dims=[512, 256], latent_dim=128,
                                       fusion_method='concat'):
        """Phase 2: 训练 ST 多模态 VAE"""
        print("\n" + "="*60)
        print("PHASE 2: Training ST Multi-Modal VAE")
        print("="*60)
        
        # 准备 ST 数据
        st_adata_count = st_adata.copy()
        sc.pp.normalize_total(st_adata_count, target_sum=1e4)
        sc.pp.log1p(st_adata_count)
        # 提取 marker genes
        available_genes = [g for g in marker_genes if g in st_adata_count.var.index]
        st_subset = st_adata_count[:, available_genes].copy()
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        print(f"ST data shape: {st_X.shape}")
        print(f"ST data (count) min: {np.min(st_X):.4f}, max: {np.max(st_X):.4f}")
        print(f"Image patches shape: {image_patches.shape}")
        
        # 划分训练/测试集
        indices = np.arange(len(st_X))
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        
        st_train = st_X[train_idx]
        st_test = st_X[test_idx]
        img_train = image_patches[train_idx]
        img_test = image_patches[test_idx]
        
        # 创建 DataLoader
        train_dataset = STImageDataset(st_train, img_train, self.image_transform)
        test_dataset = STImageDataset(st_test, img_test, self.image_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 构建 ST 多模态 VAE
        gene_input_dim = st_X.shape[1]
        self.st_vae = MultiModalVAE(
            gene_input_dim=gene_input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=0.2,
            fusion_method=fusion_method,
            pretrained_resnet=True
        ).to(self.device)
        
        optimizer = optim.Adam(self.st_vae.parameters(), lr=lr)
        
        # 训练
        train_losses = []
        test_losses = []
        
        for epoch in range(n_epochs):
            # Train
            self.st_vae.train()
            train_loss_epoch = 0
            train_recon_epoch = 0
            train_kl_epoch = 0
            
            for gene_batch, img_batch in train_loader:
                gene_batch = gene_batch.to(self.device)
                img_batch = img_batch.to(self.device)
                
                optimizer.zero_grad()
                gene_recon, fused_mu, fused_log_var, z, gene_mu, img_mu = self.st_vae(gene_batch, img_batch)
                loss, recon, kl = multimodal_vae_loss(gene_recon, gene_batch, fused_mu, fused_log_var, beta)
                loss.backward()
                optimizer.step()
                
                train_loss_epoch += loss.item()
                train_recon_epoch += recon.item()
                train_kl_epoch += kl.item()
            
            train_loss_epoch /= len(train_loader.dataset)
            train_recon_epoch /= len(train_loader.dataset)
            train_kl_epoch /= len(train_loader.dataset)
            train_losses.append(train_loss_epoch)
            
            # Test
            self.st_vae.eval()
            test_loss_epoch = 0
            with torch.no_grad():
                for gene_batch, img_batch in test_loader:
                    gene_batch = gene_batch.to(self.device)
                    img_batch = img_batch.to(self.device)
                    gene_recon, fused_mu, fused_log_var, z, _, _ = self.st_vae(gene_batch, img_batch)
                    loss, _, _ = multimodal_vae_loss(gene_recon, gene_batch, fused_mu, fused_log_var, beta)
                    test_loss_epoch += loss.item()
            
            test_loss_epoch /= len(test_loader.dataset)
            test_losses.append(test_loss_epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train: {train_loss_epoch:.4f} (Recon: {train_recon_epoch:.4f}, KL: {train_kl_epoch:.4f}) | "
                      f"Test: {test_loss_epoch:.4f}")
        
        # 保存模型
        torch.save(self.st_vae.state_dict(), 
                   os.path.join(self.output_dir, 'st_multimodal_vae_phase2.pth'))
        print(f"\n✓ Phase 2 complete! ST Multi-Modal VAE saved.")
        
        return train_losses, test_losses
    
    def phase3_align_sc_st(self, sc_adata, st_adata, marker_genes, image_patches,
                           batch_size=128, n_epochs=30, lr=1e-4, 
                           align_method='mmd', align_weight=1.0):
        """Phase 3: SC-ST 对齐"""
        print("\n" + "="*60)
        print("PHASE 3: SC-ST Alignment")
        print("="*60)
        
        # 准备数据
        sc_adata_count = sc_adata.copy()
        sc.pp.normalize_total(sc_adata_count, target_sum=1e4)
        sc_subset = sc_adata_count[:, sc_adata_count.var.index.isin(marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        
        st_adata_count = st_adata.copy()
        sc.pp.normalize_total(st_adata_count, target_sum=1e4)
        available_genes = [g for g in marker_genes if g in st_adata_count.var.index]
        st_subset = st_adata_count[:, available_genes].copy()
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        print(f"SC data: {sc_X.shape}, ST data: {st_X.shape}")
        
        # 创建 DataLoader（成对采样）
        # 这里简化处理：随机配对 SC 和 ST
        min_size = min(len(sc_X), len(st_X))
        sc_indices = np.random.choice(len(sc_X), min_size, replace=False)
        st_indices = np.random.choice(len(st_X), min_size, replace=False)
        
        sc_paired = sc_X[sc_indices]
        st_paired = st_X[st_indices]
        img_paired = image_patches[st_indices]
        
        # 划分训练/测试集
        indices = np.arange(min_size)
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        
        # 训练数据
        sc_train = torch.FloatTensor(sc_paired[train_idx])
        st_train_gene = st_paired[train_idx]
        st_train_img = img_paired[train_idx]
        
        st_train_dataset = STImageDataset(st_train_gene, st_train_img, self.image_transform)
        
        # 冻结 VAE 编码器参数（只微调对齐）
        for param in self.sc_vae.encoder.parameters():
            param.requires_grad = False
        for param in self.st_vae.gene_encoder.parameters():
            param.requires_grad = False
        for param in self.st_vae.image_encoder.parameters():
            param.requires_grad = False
        
        # 只优化融合层和解码器
        optimizer = optim.Adam([
            {'params': self.sc_vae.decoder.parameters()},
            {'params': self.st_vae.fusion_mu.parameters() if hasattr(self.st_vae, 'fusion_mu') else []},
            {'params': self.st_vae.fusion_var.parameters() if hasattr(self.st_vae, 'fusion_var') else []},
            {'params': self.st_vae.decoder.parameters()},
        ], lr=lr)
        
        # 训练
        align_losses = []
        
        for epoch in range(n_epochs):
            self.sc_vae.train()
            self.st_vae.train()
            
            epoch_align_loss = 0
            n_batches = 0
            
            # Mini-batch 训练
            for i in range(0, len(train_idx), batch_size):
                end_idx = min(i + batch_size, len(train_idx))
                
                sc_batch = sc_train[i:end_idx].to(self.device)
                
                # ST batch
                st_gene_batch = torch.FloatTensor(st_train_gene[i:end_idx]).to(self.device)
                st_img_batch_list = []
                for j in range(i, end_idx):
                    img = st_train_img[j]
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    img_tensor = self.image_transform(img_pil)
                    st_img_batch_list.append(img_tensor)
                st_img_batch = torch.stack(st_img_batch_list).to(self.device)
                
                # 编码
                _, sc_mu, _ = self.sc_vae.encode(sc_batch)
                _, st_mu, _ = self.st_vae.encode(st_gene_batch, st_img_batch)
                
                # 对齐损失
                align_loss = alignment_loss(sc_mu, st_mu, method=align_method)
                
                optimizer.zero_grad()
                align_loss.backward()
                optimizer.step()
                
                epoch_align_loss += align_loss.item()
                n_batches += 1
            
            epoch_align_loss /= n_batches
            align_losses.append(epoch_align_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Alignment Loss: {epoch_align_loss:.4f}")
        
        # 保存对齐后的模型
        torch.save(self.sc_vae.state_dict(), 
                   os.path.join(self.output_dir, 'sc_vae_aligned.pth'))
        torch.save(self.st_vae.state_dict(), 
                   os.path.join(self.output_dir, 'st_multimodal_vae_aligned.pth'))
        
        print(f"\n✓ Phase 3 complete! Aligned models saved.")
        
        return align_losses
    
    def run_three_phase_training(self, 
                                  top_n_per_type=100, 
                                  resolution=0.5,
                                  patch_size=112,
                                  phase1_epochs=50,
                                  phase2_epochs=50,
                                  phase3_epochs=30,
                                  batch_size=256,
                                  lr=1e-3,
                                  beta=1.0,
                                  hidden_dims=[512, 256],
                                  latent_dim=128,
                                  fusion_method='concat',
                                  align_method='mmd',
                                  align_weight=1.0):
        """运行完整的三阶段训练"""
        
        # 1. 加载数据
        sc_adata, st_adata, valid_samples = self.load_data()
        
        # 2. 计算聚类和 marker genes
        print("\n" + "="*60)
        print("Computing clusters and marker genes...")
        print("="*60)
        
        self.marker_genes, sc_clusters, sc_adata_clustered = compute_clusters_and_marker_genes(
            sc_adata.copy(), 
            top_n=top_n_per_type, 
            resolution=resolution,
            save_path=os.path.join(self.output_dir, "marker_genes.txt")
        )
        
        print(f"Total marker genes: {len(self.marker_genes)}")
        
        # 3. 提取图像 patches
        image_patches = self.extract_image_patches(st_adata, patch_size=patch_size)
        
        # 4. Phase 1: 训练 SC VAE
        phase1_train_loss, phase1_test_loss = self.phase1_train_sc_vae(
            sc_adata, self.marker_genes,
            batch_size=batch_size,
            n_epochs=phase1_epochs,
            lr=lr,
            beta=beta,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )
        
        # 5. Phase 2: 训练 ST 多模态 VAE
        phase2_train_loss, phase2_test_loss = self.phase2_train_st_multimodal_vae(
            st_adata, self.marker_genes, image_patches,
            batch_size=batch_size // 2,  # 图像数据占用更多内存
            n_epochs=phase2_epochs,
            lr=lr,
            beta=beta,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            fusion_method=fusion_method
        )
        
        # 6. Phase 3: SC-ST 对齐
        phase3_align_loss = self.phase3_align_sc_st(
            sc_adata, st_adata, self.marker_genes, image_patches,
            batch_size=batch_size // 2,
            n_epochs=phase3_epochs,
            lr=lr / 10,  # 更小的学习率用于微调
            align_method=align_method,
            align_weight=align_weight
        )
        
        # 7. 保存训练曲线
        self.plot_training_curves(
            phase1_train_loss, phase1_test_loss,
            phase2_train_loss, phase2_test_loss,
            phase3_align_loss
        )
        
        print("\n" + "="*60)
        print("✓ All three phases completed!")
        print("="*60)
        
        return {
            'phase1_train_loss': phase1_train_loss,
            'phase1_test_loss': phase1_test_loss,
            'phase2_train_loss': phase2_train_loss,
            'phase2_test_loss': phase2_test_loss,
            'phase3_align_loss': phase3_align_loss,
        }
    
    def plot_training_curves(self, p1_train, p1_test, p2_train, p2_test, p3_align):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Phase 1
        axes[0].plot(p1_train, label='Train', linewidth=2)
        axes[0].plot(p1_test, label='Test', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Phase 1: SC VAE Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Phase 2
        axes[1].plot(p2_train, label='Train', linewidth=2)
        axes[1].plot(p2_test, label='Test', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Phase 2: ST Multi-Modal VAE Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Phase 3
        axes[2].plot(p3_align, label='Alignment Loss', linewidth=2, color='purple')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Phase 3: SC-ST Alignment')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves_three_phases.png'), dpi=300)
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves_three_phases.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Stage 1 Multi-Modal: SC-ST Integration with Image')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default="/home/maweicheng/ST_Graduation_Project/database",
                       help='Data directory path')
    parser.add_argument('--output_dir', type=str, default="./stage1_multimodal_results",
                       help='Output directory path')
    
    # Clustering arguments
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='Marker genes per cluster')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution')
    
    # Image arguments
    parser.add_argument('--patch_size', type=int, default=112,
                       help='Image patch size')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='VAE hidden layer dimensions')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='VAE latent space dimension')
    parser.add_argument('--fusion_method', type=str, default='concat',
                       choices=['concat', 'add', 'product'],
                       help='Multi-modal fusion method')
    
    # Training arguments
    parser.add_argument('--phase1_epochs', type=int, default=50,
                       help='Phase 1 training epochs')
    parser.add_argument('--phase2_epochs', type=int, default=50,
                       help='Phase 2 training epochs')
    parser.add_argument('--phase3_epochs', type=int, default=30,
                       help='Phase 3 alignment epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight (beta-VAE)')
    
    # Alignment arguments
    parser.add_argument('--align_method', type=str, default='mmd',
                       choices=['mmd', 'contrastive', 'kl'],
                       help='SC-ST alignment method')
    parser.add_argument('--align_weight', type=float, default=1.0,
                       help='Alignment loss weight')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SCSTMultiModalTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run three-phase training
    results = trainer.run_three_phase_training(
        top_n_per_type=args.top_n_per_type,
        resolution=args.resolution,
        patch_size=args.patch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        fusion_method=args.fusion_method,
        align_method=args.align_method,
        align_weight=args.align_weight
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
