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

# 导入统一的模型定义
from model import VAE, HeterogeneousGATDeconvolution, SpatialDeconvolutionLoss

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

class GATDeconvolution:
    """第二阶段GAT解卷积训练器"""
    def __init__(self, stage1_model_path: str, output_dir: str = "./stage2_results/", device: str = None):

        self.stage1_model_path = stage1_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"  第一阶段模型: {stage1_model_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   设备: {self.device}")
    
        # 模型组件
        self.vae_encoder = None
        self.gat_model = None
        self.loss_fn = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_prototypes = None
        self.celltype_expressions = None
        
        # 图构建参数（将在训练时设置）
        self.k_spatial = 6
        self.similarity_threshold = 0.1
        
    def load_vae_encoder(self):
        """加载第一阶段VAE组件"""
        print("loading pretrain VAE Encoder")
        
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
        
        print(f"VAE Encoder加载成功: {input_dim} -> {latent_dim}")
        print(f"细胞类型: {list(self.label_encoder.classes_)}")
        print(f"Marker基因数: {len(self.genes)}")
        
        # 冻结encoder参数
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
    
    def compute_celltype_embedding(self, sc_data: np.ndarray, sc_labels: np.ndarray) -> torch.Tensor:
        """计算每个细胞类型的平均embedding"""
        print("计算细胞类型embedding")

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
                
                print(f"{cell_type}: {cell_mask.sum()}  cells")
        
        self.celltype_prototypes = torch.stack(prototypes)  # [n_cell_types, embedding_dim]
        
        print(f" celltype embeddings: {self.celltype_prototypes.shape}")
        
        return self.celltype_prototypes
    
    def compute_celltype_expressions(self, sc_data: np.ndarray, sc_labels: np.ndarray) -> torch.Tensor:
        """计算每个细胞类型的平均表达谱 """
        print("计算细胞类型表达谱")
        
        cell_types = self.label_encoder.classes_
        celltype_expressions = []
        
        for i, cell_type in enumerate(cell_types):
            # 找到该细胞类型的所有细胞
            cell_mask = (sc_labels == i)
            if cell_mask.sum() > 0:
                cell_expression = sc_data[cell_mask].mean(axis=0)
                celltype_expressions.append(cell_expression)
                
        celltype_expressions = torch.FloatTensor(np.array(celltype_expressions)).to(self.device)
        
        # 存储在trainer中，与celltype_prototypes保持一致
        self.celltype_expressions = celltype_expressions
        
        print(f"celltype expression: {celltype_expressions.shape}")
        
        return self.celltype_expressions
    
    def build_gat_model(self, n_cell_types: int, gat_hidden_dim=64, gat_layers=3, 
                       gat_heads=4, dropout=0.1, loss_alpha=1.0, loss_beta=0.1):
        """构建GAT解卷积模型"""
        print("build gat model")
        print(f"hidden dim: {gat_hidden_dim}")
        print(f"layers: {gat_layers}")
        print(f"attention heads: {gat_heads}")
        print(f"Dropout: {dropout}")
        print(f"loss: α={loss_alpha}, β={loss_beta}")
        
        self.gat_model = HeterogeneousGATDeconvolution(
            embedding_dim=128,  # VAE embedding维度
            n_cell_types=n_cell_types,
            gat_hidden_dim=gat_hidden_dim,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            dropout=dropout,
            k_spatial=self.k_spatial,
            similarity_threshold=self.similarity_threshold
        ).to(self.device)
        
        self.loss_fn = SpatialDeconvolutionLoss(alpha=loss_alpha, beta=loss_beta)
        
        gat_params = sum(p.numel() for p in self.gat_model.parameters())
        print(f"参数量: {gat_params:,}")
    
    def train_epoch_batched(self, 
                           dataloader: DataLoader,
                           optimizer) -> Dict[str, float]:
        """批处理训练一个epoch"""
        self.gat_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'mse_loss': 0.0,
            'pcc_loss': 0.0,
            'cos_loss': 0.0,
            'weight_reg': 0.0,
            'sparsity_reg': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # 提取批数据
            batch_st_data = batch['expression'].to(self.device)  # [batch_size, n_genes]
            batch_spatial_coords = batch['coords'].to(self.device)  # [batch_size, 2]
            
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
                deconv_weights=gat_outputs['deconv_weights'],        # 解卷积权重（归一化后）
                attention_scores=gat_outputs['attention_scores'],    # 原始分数（用于正则化）
                celltype_expressions=self.celltype_expressions,
                target_expressions=batch_st_data
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss_outputs['total_loss'].backward()
            optimizer.step()
            
            # 累积损失
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_outputs[key].item()
        
        # 计算平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train_gat_deconvolution(self, 
                               st_data: np.ndarray,
                               spatial_coords: np.ndarray,
                               sample_name: str,
                               n_epochs: int = 50,
                               lr: float = 1e-3,
                               batch_size: int = 512):
        """训练GAT解卷积模型"""
        print(f"开始训练GAT解卷积模型...")
 
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
        mse_losses = []
        pcc_losses = []
        cos_losses = []
        weight_regs = []
        sparsity_regs = []
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(n_epochs):
            # 训练一个epoch
            epoch_losses = self.train_epoch_batched(
                dataloader=dataloader,
                optimizer=optimizer
            )
            
            # 记录平均损失
            avg_total_loss = epoch_losses['total_loss']
            train_losses.append(avg_total_loss)
            mse_losses.append(epoch_losses['mse_loss'])
            pcc_losses.append(epoch_losses['pcc_loss'])
            cos_losses.append(epoch_losses['cos_loss'])
            weight_regs.append(epoch_losses['weight_reg'])
            sparsity_regs.append(epoch_losses['sparsity_reg'])
            
            # 学习率调度
            scheduler.step(avg_total_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}: Total Loss={avg_total_loss:.4f}, "
                      f"MSE Loss={epoch_losses['mse_loss']:.4f}, "
                      f"PCC Loss={epoch_losses['pcc_loss']:.4f}, "
                      f"Cos Loss={epoch_losses['cos_loss']:.4f}, "
                      f"Weight Reg={epoch_losses['weight_reg']:.4f}, "
                      f"Sparsity Reg={epoch_losses['sparsity_reg']:.4f}")
            
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
        self.plot_training_curves(train_losses, mse_losses, pcc_losses, cos_losses, weight_regs, sparsity_regs, sample_name)
        
        # 保存最终模型
        self.save_model(f"{self.output_dir}/final_gat_model.pth")
        
        # 评估和可视化结果（使用全量数据）
        self.evaluate_and_visualize(st_tensor, spatial_tensor, sample_name)
        
        print(f"✅ GAT解卷积训练完成! 最佳损失: {best_loss:.4f}")
        
        return {
            'best_loss': best_loss,
            'train_losses': train_losses,
            'sample_name': sample_name
        }
    
    def evaluate_and_visualize(self, 
                             st_data: torch.Tensor,
                             spatial_coords: torch.Tensor,
                             sample_name: str):
        """评估模型并可视化结果"""
        print("📊 评估模型结果...")
        
        self.gat_model.eval()
        
        with torch.no_grad():
            # 计算spot embeddings
            mu, log_var = self.vae_encoder(st_data)
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
        
        # 保存结果
        results = {
            'deconv_weights': deconv_weights,        # 解卷积权重（归一化后的细胞类型比例）
            'attention_scores': attention_scores,    # 原始注意力分数
            'cell_types': list(self.label_encoder.classes_),
            'sample_name': sample_name
        }
        
        results_file = f"{self.output_dir}/{sample_name}_deconvolution_results.npz"
        np.savez(results_file, **results)
        
        print(f"   💾 结果已保存: {results_file}")
        
        # 打印统计信息
        print(f"   📈 解卷积权重统计 (细胞类型比例):")
        for i, cell_type in enumerate(self.label_encoder.classes_):
            weight_mean = deconv_weights[:, i].mean()
            weight_std = deconv_weights[:, i].std()
            print(f"     {cell_type}: {weight_mean:.3f} ± {weight_std:.3f}")
        
        # 验证权重和是否为1
        weight_sums = deconv_weights.sum(axis=1)
        print(f"   🔍 解卷积权重和验证:")
        print(f"     权重和: {weight_sums.mean():.6f} ± {weight_sums.std():.6f} (应该等于1.0)")
    
    def plot_training_curves(self, train_losses, mse_losses, pcc_losses, cos_losses, weight_regs, sparsity_regs, sample_name):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 总损失
        ax1.plot(epochs, train_losses, 'b-')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 重建损失（MSE + PCC + Cosine）
        ax2.plot(epochs, mse_losses, 'g-', label='MSE')
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
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'stage1_model_path': self.stage1_model_path
        }, filepath)

def main():

    parser = argparse.ArgumentParser(description='Stage 2: GAT Deconvolution for Spatial Transcriptomics')
    # 模型和数据参数
    parser.add_argument('--stage1_model_path', type=str, 
                       default="./stage1_results/best_vae.pth",
                       help='第一阶段VAE模型路径')
    parser.add_argument('--sc_file', type=str, required=True,
                       help='单细胞数据文件路径 (.h5ad)')
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
    
    # 图构建参数
    parser.add_argument('--k_spatial', type=int, default=6,
                       help='空间邻居数 (KNN)')
    parser.add_argument('--similarity_threshold', type=float, default=0.1,
                       help='Spot-CellType相似度阈值')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    
    # 损失函数参数
    parser.add_argument('--loss_alpha', type=float, default=1.0,
                       help='重建损失权重')
    parser.add_argument('--loss_beta', type=float, default=0.1,
                       help='注意力损失权重')
    
    # 设备参数
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu，None为自动选择)')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.sc_file):
        raise FileNotFoundError(f"❌ SC数据文件不存在: {args.sc_file}")
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
    trainer.similarity_threshold = args.similarity_threshold
    
    # 加载VAE Encoder
    trainer.load_vae_encoder()
    print("=" * 60)

    # 加载SC数据
    print(f"加载SC数据: {args.sc_file}")
    sc_adata = sc.read_h5ad(args.sc_file)
    print(f"SC数据: {sc_adata.shape}, 细胞类型: {len(sc_adata.obs['cell_type'].unique())}")
    
    # 提取SC marker基因数据
    sc_subset = sc_adata[:, trainer.genes].copy()

    # SC标准化（与第一阶段一致）
    sc.pp.normalize_total(sc_subset, target_sum=1e4)
    sc.pp.log1p(sc_subset)
    
    sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
    sc_labels = sc_subset.obs['cell_type'].values
    
    # 编码标签
    sc_y = trainer.label_encoder.transform(sc_labels)
    print(f"SC预处理完成: {sc_X.shape}")
    
    # 计算细胞类型原型和表达谱
    trainer.compute_celltype_embedding(sc_X, sc_y)
    trainer.compute_celltype_expressions(sc_X, sc_y)
    
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
    print(f"ST匹配基因: {len(trainer.genes)}/{len(trainer.genes)} (100%)")
    
    # 提取ST数据（使用原始counts）
    st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
    
    # 提取空间坐标（不添加偏移，单样本训练）
    spatial_coords = st_adata.obsm['spatial']
    
    print(f" ST数据: {st_X.shape}")
  
    
    # 构建GAT模型和损失函数
    n_cell_types = len(trainer.label_encoder.classes_)
    print("=" * 60)
    print(f"构建GAT解卷积模型 (细胞类型数: {n_cell_types})...")

    trainer.build_gat_model(
        n_cell_types=n_cell_types,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        loss_alpha=args.loss_alpha,
        loss_beta=args.loss_beta
    )
    
    # 开始训练
    print("=" * 60)
    print(f"开始GAT解卷积训练...")
    
    trainer.train_gat_deconvolution(
        st_data=st_X,
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    
    print("="*60)
    print("GAT解卷积训练完成!")

if __name__ == "__main__":
    main()