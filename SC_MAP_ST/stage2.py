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
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import unified model definitions
from model import VAE, HeterogeneousGATDeconvolution, SpatialDeconvolutionLoss

class SpatialDataset(Dataset):
    """Spatial transcriptomics dataset"""
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

class GATDeconvolution:
    """Stage 2 GAT deconvolution trainer"""
    def __init__(self, stage1_model_path: str, output_dir: str = "./stage2_results/", device: str = None):

        self.stage1_model_path = stage1_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Stage 1 model: {stage1_model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {self.device}")
    
        # Model components
        self.vae_encoder = None
        self.gat_model = None
        self.loss_fn = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_prototypes = None
        self.celltype_expressions = None
        
        # Graph building parameters (will be set during training)
        self.k_spatial = 20
        self.k_celltype = 10  # Connect each spot to k nearest cell types
        
    def load_vae_encoder(self):
        """Load Stage 1 VAE components"""
        print("Loading pretrained VAE Encoder...")
        
        checkpoint = torch.load(self.stage1_model_path, map_location=self.device)
        
        # Rebuild VAE
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        # Only use the encoder part
        full_vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        full_vae.load_state_dict(checkpoint['vae_state_dict'])
        
        # Extract encoder
        self.vae_encoder = full_vae.encoder
        self.vae_encoder.eval()  # Freeze encoder
        
        # Other information
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        self.sc_clusters = checkpoint.get('sc_clusters', None)  # Load clustering info
        self.resolution = checkpoint.get('resolution', 0.5)     # Load resolution
        
        # Load cluster centers
        cluster_prototypes = checkpoint.get('cluster_prototypes', None)
        if cluster_prototypes is not None:
            # Convert to tensor format
            prototype_list = []
            for i in range(len(self.label_encoder.classes_)):
                cluster_id = i  # Use index instead of name
                if cluster_id in cluster_prototypes:
                    prototype_list.append(cluster_prototypes[cluster_id])
                else:
                    print(f"Warning: Missing cluster {cluster_id} center, using zero vector")
                    prototype_list.append(np.zeros(latent_dim))
            
            self.celltype_prototypes = torch.FloatTensor(np.array(prototype_list)).to(self.device)
            print(f"Loaded pretrained cluster centers: {self.celltype_prototypes.shape}")
        else:
            self.celltype_prototypes = None
            print("Warning: Pretrained cluster centers not found, will recompute")
        
        # Load cluster expression profiles
        cluster_expressions = checkpoint.get('cluster_expressions', None)
        if cluster_expressions is not None:
            # Convert to tensor format
            expression_list = []
            for i in range(len(self.label_encoder.classes_)):
                cluster_id = i  # Use index instead of name
                if cluster_id in cluster_expressions:
                    expression_list.append(cluster_expressions[cluster_id])
                else:
                    print(f"Warning: Missing cluster {cluster_id} expression, using zero vector")
                    expression_list.append(np.zeros(input_dim))
            
            self.celltype_expressions = torch.FloatTensor(np.array(expression_list)).to(self.device)
            print(f"Loaded pretrained cluster expressions: {self.celltype_expressions.shape}")
        else:
            self.celltype_expressions = None
            print("Warning: Pretrained cluster expressions not found, will recompute")
        
        # Load full-gene cluster expression profiles
        cluster_expressions_full = checkpoint.get('cluster_expressions_full', None)
        if cluster_expressions_full is not None:
            # Convert to list format
            expression_full_list = []
            for i in range(len(self.label_encoder.classes_)):
                cluster_id = i
                if cluster_id in cluster_expressions_full:
                    expression_full_list.append(cluster_expressions_full[cluster_id])
                else:
                    # If no full-gene version, use marker gene version (may cause gene count mismatch)
                    print(f"Warning: Missing cluster {cluster_id} full-gene expression")
                    expression_full_list.append(None)
            
            self.celltype_expressions_full = expression_full_list  # Save as list, different dimensions
            full_gene_count = len(expression_full_list[0]) if expression_full_list[0] is not None else 0
            print(f"Loaded pretrained cluster full-gene expressions: {len(expression_full_list)} clusters × {full_gene_count} genes")
        else:
            self.celltype_expressions_full = None
            print("⚠️  未找到预训练聚类全基因表达谱")
        
        # 加载全基因列表
        all_genes = checkpoint.get('all_genes', None)
        if all_genes is not None:
            self.all_genes = all_genes
            print(f"✅ 加载全基因列表: {len(all_genes)} 个基因")
        else:
            self.all_genes = None
            print("⚠️  未找到全基因列表")
        
        print(f"VAE Encoder加载成功: {input_dim} -> {latent_dim}")
        print(f"细胞聚类: {list(self.label_encoder.classes_)}")
        print(f"Marker基因数: {len(self.genes)}")
        
        # 冻结encoder参数
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
    
    def build_gat_model(self, n_cell_types: int, gat_hidden_dim=64, gat_layers=3, 
                       gat_heads=4, dropout=0.1, loss_lambda_pearson=1.0,
                       loss_lambda_cosine=1.0, loss_lambda_align=1.0, 
                       loss_lambda_reg=0.5, loss_lambda_sparse=0.01):
        """构建GAT解卷积模型"""
        print("build gat model")
        print(f"hidden dim: {gat_hidden_dim}")
        print(f"layers: {gat_layers}")
        print(f"attention heads: {gat_heads}")
        print(f"Dropout: {dropout}")
        print(f"loss: λ_pearson={loss_lambda_pearson}, λ_cosine={loss_lambda_cosine}, "
              f"λ_align={loss_lambda_align}, λ_reg={loss_lambda_reg}, λ_sparse={loss_lambda_sparse}")
        
        self.gat_model = HeterogeneousGATDeconvolution(
            embedding_dim=128,
            n_cell_types=n_cell_types,
            gat_hidden_dim=gat_hidden_dim,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            dropout=dropout,
            k_spatial=self.k_spatial,
            k_celltype=self.k_celltype
        ).to(self.device)
        
        self.loss_fn = SpatialDeconvolutionLoss(
            lambda_pearson=loss_lambda_pearson,
            lambda_cosine=loss_lambda_cosine,
            lambda_align=loss_lambda_align,
            lambda_reg=loss_lambda_reg,
            lambda_sparse=loss_lambda_sparse
        )
        
        gat_params = sum(p.numel() for p in self.gat_model.parameters())
        print(f"参数量: {gat_params:,}")
    
    def train_epoch_batched(self, 
                           dataloader: DataLoader,
                           optimizer) -> Dict[str, float]:
        """批处理训练一个epoch"""
        self.gat_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'pearson_loss': 0.0,
            'cosine_loss': 0.0,
            'alignment_loss': 0.0,
            'weight_reg': 0.0,
            'sparsity_loss': 0.0,
            'pearson_corr': 0.0,
            'cos_sim_rec': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # 提取批数据
            batch_st_data = batch['expression'].to(self.device)
            batch_spatial_coords = batch['coords'].to(self.device)
            
            # 计算spot embeddings
            with torch.no_grad():
                mu, log_var = self.vae_encoder(batch_st_data)
                spot_embeddings = mu
            
            # GAT前向传播
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=batch_spatial_coords,
                celltype_prototypes=self.celltype_prototypes
            )
            
            # 计算损失
            loss_outputs = self.loss_fn(
                attention_weights=gat_outputs['deconv_weights'],
                celltype_expression=self.celltype_expressions,
                true_spot_expression=batch_st_data,
                spot_embedding=gat_outputs['spot_features'],
                celltype_embedding=gat_outputs['celltype_features']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss_outputs['total_loss'].backward()
            optimizer.step()
            
            # 累积损失
            for key in epoch_losses.keys():
                if key in loss_outputs:
                    epoch_losses[key] += loss_outputs[key].item()
        
        # 计算平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train_gat_deconvolution(self, 
                               st_data: np.ndarray,
                               spatial_coords: np.ndarray,
                               sample_name: str,
                               st_adata=None,
                               n_epochs: int = 50,
                               lr: float = 1e-3,
                               batch_size: int = 512):
        """训练GAT解卷积模型"""
        print(f"开始训练GAT解卷积模型...")
        
        # 保存 st_adata 供后续使用
        self.st_adata = st_adata
 
        # 转换为tensor
        st_tensor = torch.FloatTensor(st_data).to(self.device)
        spatial_tensor = torch.FloatTensor(spatial_coords).to(self.device)
        
        # 创建数据集和数据加载器
        spot_ids = list(range(len(st_data)))
        dataset = SpatialDataset(st_data, spatial_coords, spot_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # 优化器
        optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # 训练历史
        train_losses = []
        pcc_losses = []
        cos_losses = []
        weight_regs = []
        sparsity_regs = []
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        for epoch in range(n_epochs):
            # 训练一个epoch
            epoch_losses = self.train_epoch_batched(
                dataloader=dataloader,
                optimizer=optimizer
            )
            
            # 记录平均损失
            avg_total_loss = epoch_losses['total_loss']
            train_losses.append(avg_total_loss)
            pcc_losses.append(epoch_losses['pearson_loss'])
            cos_losses.append(epoch_losses['cosine_loss'])
            weight_regs.append(epoch_losses['weight_reg'])
            sparsity_regs.append(epoch_losses.get('sparsity_loss', 0.0))
            
            # 学习率调度
            scheduler.step(avg_total_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}: Total Loss={avg_total_loss:.4f}, "
                      f"Pearson={epoch_losses['pearson_loss']:.4f}, "
                      f"Cosine={epoch_losses['cosine_loss']:.4f}, "
                      f"Align={epoch_losses['alignment_loss']:.4f}, "
                      f"WReg={epoch_losses['weight_reg']:.4f}, "
                      f"Sparse={epoch_losses['sparsity_loss']:.4f} "
                      f"(PCC={epoch_losses['pearson_corr']:.4f}, CosSim={epoch_losses['cos_sim_rec']:.4f})")
            
            # 保存最佳模型
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_model(f"{self.output_dir}/best_gat_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, pcc_losses, cos_losses, weight_regs, sparsity_regs, sample_name)
        
        # 保存最终模型
        self.save_model(f"{self.output_dir}/final_gat_model.pth")
        
        # 评估和可视化结果（使用全量数据）
        self.evaluate_and_visualize(st_data, self.st_adata, spatial_tensor, sample_name)
        
        print(f"✅ GAT解卷积训练完成! 最佳损失: {best_loss:.4f}")
        
        return {
            'best_loss': best_loss,
            'train_losses': train_losses,
            'sample_name': sample_name
        }
    
    def evaluate_and_visualize(self, 
                             st_data: np.ndarray,
                             st_adata,
                             spatial_coords: torch.Tensor,
                             sample_name: str):
        """评估模型并可视化结果，生成反卷积表达矩阵"""
        print("📊 评估模型结果...")
        
        self.gat_model.eval()
        
        st_tensor = torch.FloatTensor(st_data).to(self.device)
        
        with torch.no_grad():
            # 计算spot embeddings
            mu, log_var = self.vae_encoder(st_tensor)
            spot_embeddings = mu
            
            # GAT前向传播
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=spatial_coords,
                celltype_prototypes=self.celltype_prototypes
            )
            
            # 获取预测结果
            deconv_weights = gat_outputs['deconv_weights'].detach().cpu().numpy()
            attention_scores = gat_outputs['attention_scores'].detach().cpu().numpy()
        
        # 保存解卷积权重
        print("💾 保存反卷积结果...")
        weights_file = f"{self.output_dir}/{sample_name}_deconv_weights.npz"
        np.savez(weights_file, 
                deconv_weights=deconv_weights,
                attention_scores=attention_scores,
                clusters=self.label_encoder.classes_)
        print(f"   ✅ 权重已保存: {weights_file}")
        
        # ============ 生成表达矩阵 ============
        n_spots = deconv_weights.shape[0]
        n_clusters = deconv_weights.shape[1]
        
        print("\n🧬 生成反卷积表达矩阵...")
        
        # 获取 spot barcode
        spot_barcodes = list(st_adata.obs.index)
        
        # 1. Marker基因表达矩阵（spot × marker基因）
        print("   1️⃣  生成Marker基因表达矩阵...")
        celltype_expr_marker = self.celltype_expressions.cpu().numpy()  # [n_clusters, n_marker_genes]
        reconstructed_marker_expr = np.dot(deconv_weights, celltype_expr_marker)  # [n_spots, n_marker_genes]
        
        marker_expr_df = pd.DataFrame(
            reconstructed_marker_expr,
            columns=self.genes,
            index=spot_barcodes
        )
        
        # Filter out genes with zero expression across all spots
        non_zero_genes_marker = (marker_expr_df != 0).any(axis=0)
        marker_expr_df = marker_expr_df.loc[:, non_zero_genes_marker]
        
        marker_expr_file = f"{self.output_dir}/{sample_name}_reconstructed_marker_genes.csv"
        marker_expr_df.to_csv(marker_expr_file)
        print(f"      ✅ Marker基因表达: {marker_expr_df.shape} (移除了{sum(~non_zero_genes_marker)}个全零基因)")
        
        # 2. 全基因表达矩阵（需要celltype全基因表达）
        if self.celltype_expressions_full is not None and all(expr is not None for expr in self.celltype_expressions_full):
            print("   2️⃣  生成全基因表达矩阵...")
            celltype_expr_full = np.array(self.celltype_expressions_full)  # [n_clusters, n_all_genes]
            reconstructed_full_expr = np.dot(deconv_weights, celltype_expr_full)  # [n_spots, n_all_genes]
            
            # 获取全基因名称
            if self.all_genes is not None:
                all_gene_names = self.all_genes
                print(f"      使用保存的基因名: {len(all_gene_names)} 个")
            else:
                # 备用方案：使用基因索引
                all_gene_names = [f"Gene_{i}" for i in range(celltype_expr_full.shape[1])]
                print(f"      ⚠️  使用基因索引 (Gene_0, Gene_1, ...)")
            
            full_expr_df = pd.DataFrame(
                reconstructed_full_expr,
                columns=all_gene_names,
                index=spot_barcodes
            )
            
            # Filter out genes with zero expression across all spots
            non_zero_genes_full = (full_expr_df != 0).any(axis=0)
            full_expr_df = full_expr_df.loc[:, non_zero_genes_full]
            
            full_expr_file = f"{self.output_dir}/{sample_name}_reconstructed_all_genes.csv"
            full_expr_df.to_csv(full_expr_file)
            print(f"      ✅ 全基因表达: {full_expr_df.shape} (移除了{sum(~non_zero_genes_full)}个全零基因)")
        else:
            print("      ⚠️  未加载全基因celltype表达，跳过全基因矩阵生成")
        
        # 3. 聚类组成矩阵（spot × 聚类）
        print("   3️⃣  生成聚类组成矩阵...")
        composition_df = pd.DataFrame(
            deconv_weights,
            columns=[f"Cluster_{int(c)}" for c in self.label_encoder.classes_],
            index=spot_barcodes
        )
        composition_file = f"{self.output_dir}/{sample_name}_cell_composition.csv"
        composition_df.to_csv(composition_file)
        print(f"      ✅ 聚类组成: {composition_df.shape}")
        
        # 保存结果summary
        results = {
            'deconv_weights': deconv_weights,
            'attention_scores': attention_scores,
            'clusters': list(self.label_encoder.classes_),
            'sample_name': sample_name,
            'marker_genes': self.genes,
            'n_spots': n_spots,
            'n_clusters': n_clusters
        }
        
        results_file = f"{self.output_dir}/{sample_name}_deconvolution_results.npz"
        np.savez(results_file, **results)
        print(f"\n   💾 完整结果已保存: {results_file}")
        
        # 打印统计信息
        print(f"\n📈 解卷积权重统计 (聚类比例):")
        for i, cluster in enumerate(self.label_encoder.classes_):
            weight_mean = deconv_weights[:, i].mean()
            weight_std = deconv_weights[:, i].std()
            print(f"   Cluster {cluster}: {weight_mean:.3f} ± {weight_std:.3f}")
        
        # 验证权重和是否为1
        weight_sums = deconv_weights.sum(axis=1)
        print(f"\n🔍 解卷积权重和验证:")
        print(f"   权重和: {weight_sums.mean():.6f} ± {weight_sums.std():.6f} (应该等于1.0)")
    
    def plot_training_curves(self, train_losses, pcc_losses, cos_losses, weight_regs, sparsity_regs, sample_name):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 总损失
        ax1.plot(epochs, train_losses, 'b-')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 重建损失（PCC + Cosine）
        ax2.plot(epochs, pcc_losses, 'orange', label='PCC')
        ax2.plot(epochs, cos_losses, 'red', label='Cosine')
        ax2.set_title('Reconstruction Losses')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # 权重正则化
        ax3.plot(epochs, weight_regs, 'r-')
        ax3.set_title('Weight Regularization')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
        
        # 稀疏性正则化
        ax4.plot(epochs, sparsity_regs, 'purple')
        ax4.set_title('Sparsity Regularization')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        plt.suptitle(f'GAT Deconvolution Training - {sample_name}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gat_training_curves_{sample_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'gat_state_dict': self.gat_model.state_dict(),
            'celltype_prototypes': self.celltype_prototypes,
            'celltype_expressions': self.celltype_expressions,
            'celltype_expressions_full': getattr(self, 'celltype_expressions_full', None),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'stage1_model_path': self.stage1_model_path
        }, filepath)

def main():

    parser = argparse.ArgumentParser(description='Stage 2: GAT Deconvolution for Spatial Transcriptomics')
    # 模型和数据参数
    parser.add_argument('--stage1_model_path', type=str, 
                       default="./stage1_results/final_vae.pth",
                       help='第一阶段VAE模型路径')
    parser.add_argument('--st_file', type=str, required=True,
                       help='空间转录组数据文件路径 (.h5ad)')
    parser.add_argument('--output_dir', type=str, 
                       default="./stage2_results",
                       help='输出目录路径')

    # GAT模型参数
    parser.add_argument('--gat_hidden_dim', type=int, default=64,
                       help='GAT隐藏层维度')
    parser.add_argument('--gat_layers', type=int, default=3,
                       help='GAT层数')
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='GAT注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    
    # 聚类参数
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden聚类分辨率')
    
    # 图构建参数
    parser.add_argument('--k_spatial', type=int, default=6,
                       help='空间邻居数 (KNN)')
    parser.add_argument('--k_celltype', type=int, default=10,
                       help='每个spot连接的最近celltype数量 (KNN)')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    
    # 损失函数参数
    parser.add_argument('--loss_lambda_pearson', type=float, default=1.0,
                       help='Pearson相关系数损失权重')
    parser.add_argument('--loss_lambda_cosine', type=float, default=1.0,
                       help='Cosine相似度损失权重')
    parser.add_argument('--loss_lambda_align', type=float, default=1.0,
                       help='模态对齐损失权重')
    parser.add_argument('--loss_lambda_reg', type=float, default=0.5,
                       help='权重正则化权重')
    parser.add_argument('--loss_lambda_sparse', type=float, default=0.01,
                       help='稀疏性正则化权重（Shannon熵）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu，None为自动选择)')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.st_file):
        raise FileNotFoundError(f"❌ ST数据文件不存在: {args.st_file}")
    
    # 获取样本名称（从文件名提取）
    sample_name = os.path.splitext(os.path.basename(args.st_file))[0]
    if sample_name.endswith('_ST'):
        sample_name = sample_name[:-3]  # 移除 '_ST' 后缀
    
    print(f"   样本名称: {sample_name}")
 
    # 初始化训练器
    trainer = GATDeconvolution(
        stage1_model_path=args.stage1_model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # 设置图构建参数
    trainer.k_spatial = args.k_spatial
    trainer.k_celltype = args.k_celltype
    
    # 加载VAE Encoder
    trainer.load_vae_encoder()
    print("=" * 60)

    print("🔍 直接使用第一阶段的聚类中心和表达谱...")

    print(f"✅ 加载了 {len(trainer.label_encoder.classes_)} 个聚类")
    print("=" * 60)
    
    # 检查是否已有预训练的聚类数据
    has_prototypes = hasattr(trainer, 'celltype_prototypes') and trainer.celltype_prototypes is not None
    has_expressions = hasattr(trainer, 'celltype_expressions') and trainer.celltype_expressions is not None
    
    if has_prototypes and has_expressions:
        print("✅ 使用第一阶段预训练的聚类数据")
        print(f"   聚类中心: {trainer.celltype_prototypes.shape}")
        print(f"   聚类表达谱: {trainer.celltype_expressions.shape}")
        print(f"   聚类表达谱 (第一个): {trainer.celltype_expressions[0]}")
        print(f"   聚类表达谱 (第二个): {trainer.celltype_expressions[1]}")
        n_clusters = trainer.celltype_prototypes.shape[0]
    else:
        raise ValueError("❌ 第一阶段模型缺少聚类中心或表达谱！")
    
    print("=" * 60)
    print("加载和处理空间转录组数据...")
    
    # 加载ST数据
    print(f"加载ST数据: {args.st_file}")
    st_adata = sc.read_h5ad(args.st_file)

    # 检查空间坐标
    if 'spatial' not in st_adata.obsm:
        raise ValueError(f"❌ ST数据文件 {args.st_file} 缺少空间坐标信息！ST数据必须包含 'spatial' 坐标（adata.obsm['spatial']）。")
    
    # 提取ST marker基因数据
    st_subset = st_adata[:, trainer.genes].copy()
    print(f"ST匹配基因: {len(trainer.genes)}/{len(trainer.genes)}")
    
    # 提取ST数据（使用原始counts）
    sc.pp.log1p(st_subset)
    st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
    
    # 提取空间坐标（不添加偏移，单样本训练）
    spatial_coords = st_adata.obsm['spatial']
    
    print(f" ST数据: {st_X.shape}")
  
    
    # 构建GAT模型和损失函数
    print("=" * 60)
    print(f"构建GAT解卷积模型 (聚类数: {n_clusters})...")
    trainer.build_gat_model(
        n_cell_types=n_clusters,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        loss_lambda_pearson=args.loss_lambda_pearson,
        loss_lambda_cosine=args.loss_lambda_cosine,
        loss_lambda_align=args.loss_lambda_align,
        loss_lambda_reg=args.loss_lambda_reg,
        loss_lambda_sparse=args.loss_lambda_sparse
    )
    
    # 开始训练
    print("=" * 60)
    print(f"开始GAT解卷积训练...")
    
    trainer.train_gat_deconvolution(
        st_data=st_X,
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        st_adata=st_adata,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    
    print("="*60)
    print("GAT解卷积训练完成!")

if __name__ == "__main__":
    main()