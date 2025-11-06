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
from tqdm import tqdm
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
    """Stage 2: GAT deconvolution trainer"""
    def __init__(self, stage1_model_path: str, output_dir: str = "./stage2_results/", device: str = None, weight_threshold: float = 0.01):

        self.stage1_model_path = stage1_model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Weight threshold for sparsification
        self.weight_threshold = weight_threshold
        
        print("="*60)
        print(f"Stage 1 model: {stage1_model_path}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {self.device}")
        print(f"Weight threshold: {weight_threshold}")
    
        # Model components
        self.vae_encoder = None
        self.gat_model = None
        self.loss_fn = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_prototypes = None
        self.celltype_expressions = None
        self.cluster_to_celltype = None  # Mapping from cluster index to celltype name
        self.celltype_key = None  # Track which celltype column was used
        
        # Graph construction parameters
        self.k_spatial = 20
        self.k_celltype = 10
        
    def load_vae_encoder(self):
        """Load Stage 1 VAE components"""
        print("="*60)
        print("Loading pretrained VAE Encoder...")
        
        checkpoint = torch.load(self.stage1_model_path, map_location=self.device, weights_only=False)
        
        # Rebuild VAE
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        output_type = checkpoint.get('output_type', 'mse')  # Get output_type from checkpoint
        
        print(f"   VAE architecture: {input_dim} -> {latent_dim}")
        print(f"   Output type: {output_type}")
        
        # 检测是否是双解码器架构
        state_dict = checkpoint['vae_state_dict']
        is_dual_decoder = any('decoder_sc' in key or 'decoder_st' in key for key in state_dict.keys())
        
        if is_dual_decoder:
            print(f"   Architecture: Dual Decoder (SC/ST-specific)")
            # 使用双解码器VAE
            from model import DualDecoderVAE
            full_vae = DualDecoderVAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)
        else:
            print(f"   Architecture: Single Decoder")
            # 使用标准VAE
            full_vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)
        
        full_vae.load_state_dict(state_dict)
        
        # Extract encoder (encoder是共享的，不管单双解码器)
        self.vae_encoder = full_vae.encoder
        self.vae_encoder.eval()  # Freeze encoder
        
        # Get latent dimension from the loaded VAE model
        self.latent_dim = full_vae.encoder.fc_mu.out_features
        
        # Other information
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        self.sc_clusters = checkpoint.get('sc_clusters', None)
        
        # 尝试从 sc_adata_clustered.h5ad 加载聚类信息（作为备选）
        if self.sc_clusters is None:
            sc_adata_path = os.path.join(os.path.dirname(self.stage1_model_path), 'sc_adata_clustered.h5ad')
            if os.path.exists(sc_adata_path):
                print(f"\n   sc_clusters not in checkpoint, loading from {sc_adata_path}")
                import scanpy as sc
                sc_adata = sc.read_h5ad(sc_adata_path)
                if 'leiden' in sc_adata.obs:
                    self.sc_clusters = sc_adata.obs['leiden'].copy()
                    if hasattr(self.sc_clusters, 'cat'):
                        self.sc_clusters = self.sc_clusters.cat.remove_unused_categories()
                    print(f"   ✓ Loaded {len(self.sc_clusters)} cell cluster labels from h5ad")
                else:
                    print(f"   ⚠️ 'leiden' column not found in {sc_adata_path}")
            else:
                print(f"   ⚠️ sc_adata_clustered.h5ad not found at {sc_adata_path}")
        else:
            print(f"   ✓ Loaded {len(self.sc_clusters)} cell cluster labels from checkpoint")
        self.resolution = checkpoint.get('resolution', 0.5)
        self.celltype_key = checkpoint.get('celltype_key', None)  # Get celltype mode info
        
        # Load cluster-to-celltype mapping if available
        self.cluster_to_celltype = None
        if self.celltype_key is not None:
            # If using celltype mode, create mapping from label_encoder
            # label_encoder.classes_ contains celltype names
            self.cluster_to_celltype = {i: ct for i, ct in enumerate(self.label_encoder.classes_)}
            print(f"Celltype mode: {self.celltype_key}")
            print(f"Cluster → CellType mapping: {self.cluster_to_celltype}")
        
        # Load cluster prototypes
        cluster_prototypes = checkpoint.get('cluster_prototypes', None)
        if cluster_prototypes is not None:
            # Convert to tensor format
            # cluster_prototypes is a dict with numeric keys (0, 1, 2, ...)
            # regardless of whether using celltype annotation or auto-clustering
            prototype_list = []
            for i in range(len(self.label_encoder.classes_)):
                if i in cluster_prototypes:
                    prototype_list.append(cluster_prototypes[i])
                else:
                    print(f"Warning: cluster {i} missing center, using zero vector")
                    prototype_list.append(np.zeros(latent_dim))
            
            self.celltype_prototypes = torch.FloatTensor(np.array(prototype_list)).to(self.device)
            print(f"Loaded cluster centers: {self.celltype_prototypes.shape}")
        else:
            self.celltype_prototypes = None
            print("Warning: cluster centers not found, will recompute")
        
        # Load cluster expressions
        cluster_expressions = checkpoint.get('cluster_expressions', None)
        if cluster_expressions is not None:
            # Convert to tensor format
            # cluster_expressions is a dict with numeric keys (0, 1, 2, ...)
            expression_list = []
            for i in range(len(self.label_encoder.classes_)):
                if i in cluster_expressions:
                    expression_list.append(cluster_expressions[i])
                else:
                    print(f"Warning: cluster {i} missing expression, using zero vector")
                    expression_list.append(np.zeros(input_dim))
            
            self.celltype_expressions = torch.FloatTensor(np.array(expression_list)).to(self.device)
            print(f"Loaded cluster expressions: {self.celltype_expressions.shape}")
        else:
            self.celltype_expressions = None
            print("Warning: cluster expressions not found, will recompute")
        
        # Load full gene cluster expressions (count version for reconstruction)
        cluster_expressions_full_count = checkpoint.get('cluster_expressions_full_count', None)
        if cluster_expressions_full_count is None:
            # Fall back to log version if count not available
            cluster_expressions_full_count = checkpoint.get('cluster_expressions_full', None)
            print("Warning: Using log1p version as count version not found")
        
        if cluster_expressions_full_count is not None:
            # Convert to list format
            # cluster_expressions_full_count is a dict with numeric keys (0, 1, 2, ...)
            expression_full_list = []
            for i in range(len(self.label_encoder.classes_)):
                if i in cluster_expressions_full_count:
                    expression_full_list.append(cluster_expressions_full_count[i])
                else:
                    print(f"Warning: cluster {i} missing full gene expression")
                    expression_full_list.append(None)
            
            self.celltype_expressions_full = expression_full_list
            full_gene_count = len(expression_full_list[0]) if expression_full_list[0] is not None else 0
            print(f"Loaded full gene expressions (count): {len(expression_full_list)} clusters × {full_gene_count} genes")
        else:
            self.celltype_expressions_full = None
            print("Warning: full gene expressions not found")
        
        # Load all genes list
        all_genes = checkpoint.get('all_genes', None)
        if all_genes is not None:
            self.all_genes = all_genes
            print(f"Loaded all genes list: {len(all_genes)} genes")
        else:
            self.all_genes = None
            print("Warning: all genes list not found")
        
        print(f"VAE Encoder loaded: {input_dim} -> {latent_dim}")
        print(f"Cell type clusters: {list(self.label_encoder.classes_)}")
        print(f"Marker genes: {len(self.genes)}")
        
        # Freeze encoder parameters
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
       
    def build_gat_model(self, n_cell_types: int, gat_hidden_dim=64, gat_layers=3, 
                       gat_heads=4, dropout=0.1, loss_lambda_pearson=1.0, loss_lambda_mse=1.0,
                       loss_lambda_cosine=1.0, 
                       loss_lambda_reg=0.5, loss_lambda_sparse=0.01, loss_lambda_diversity=0.1,
                       loss_lambda_hetero=0.05, loss_lambda_proportion=1.0):
        """Build GAT deconvolution model"""
        print("="*60)
        print("Building GAT model...")
        
        # Get embedding dimension from VAE encoder
        embedding_dim = self.latent_dim
        print(f"VAE latent dimension: {embedding_dim}")
        
        print(f"GAT hidden dim: {gat_hidden_dim}")
        print(f"GAT layers: {gat_layers}")
        print(f"Attention heads: {gat_heads}")
        print(f"Dropout: {dropout}")
        print(f"Loss weights: λ_pearson={loss_lambda_pearson}, λ_mse={loss_lambda_mse}, "
              f"λ_cosine={loss_lambda_cosine}, "
              f"λ_reg={loss_lambda_reg}, λ_sparse={loss_lambda_sparse}, λ_diversity={loss_lambda_diversity}, "
              f"λ_hetero={loss_lambda_hetero}, λ_proportion={loss_lambda_proportion}")
        
        self.gat_model = HeterogeneousGATDeconvolution(
            embedding_dim=embedding_dim,  # Use actual VAE latent dimension
            n_cell_types=n_cell_types,
            gat_hidden_dim=gat_hidden_dim,
            gat_layers=gat_layers,
            gat_heads=gat_heads,
            dropout=dropout,
            k_spatial=self.k_spatial,
            k_celltype=self.k_celltype,
            celltype_prototypes=self.celltype_prototypes  # 使用第一阶段的celltype prototypes初始化
        ).to(self.device)
        
        # 计算单细胞数据中各cluster的比例
        sc_celltype_proportions = None
        if hasattr(self, 'sc_clusters') and self.sc_clusters is not None:
            # sc_clusters 是从 stage1 加载的单细胞cluster标签
            print(f"\n   Computing cell type proportions from {len(self.sc_clusters)} cells...")
            cluster_counts = {}
            for cluster_id in self.sc_clusters:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            total_cells = len(self.sc_clusters)
            # 按cluster ID顺序构建比例数组
            proportions = []
            for i in range(n_cell_types):
                count = cluster_counts.get(str(i), 0)  # cluster ID可能是字符串
                if count == 0:
                    count = cluster_counts.get(i, 0)  # 尝试整数
                proportion = count / total_cells if total_cells > 0 else 1.0 / n_cell_types
                proportions.append(proportion)
            
            sc_celltype_proportions = proportions
            print(f"   Single-cell cluster proportions (total {total_cells} cells):")
            for i, prop in enumerate(proportions):
                count = cluster_counts.get(str(i), cluster_counts.get(i, 0))
                print(f"      Cluster {i}: {count:6d} cells ({prop*100:6.2f}%)")
        else:
            print("\n   ⚠️ Warning: sc_clusters not available, proportion loss will not be effective")
            print("      Tip: Make sure stage1 saves sc_clusters or sc_adata_clustered.h5ad exists")
        
        self.loss_fn = SpatialDeconvolutionLoss(
            lambda_pearson=loss_lambda_pearson,
            lambda_mse=loss_lambda_mse,
            lambda_cosine=loss_lambda_cosine,
            lambda_reg=loss_lambda_reg,
            lambda_sparse=loss_lambda_sparse,
            lambda_diversity=loss_lambda_diversity,
            lambda_hetero=loss_lambda_hetero,
            lambda_proportion=loss_lambda_proportion,
            sc_celltype_proportions=sc_celltype_proportions
        )
        
        gat_params = sum(p.numel() for p in self.gat_model.parameters())
        print(f"GAT parameters: {gat_params:,}")
    
    def train_epoch_batched(self, 
                           dataloader: DataLoader,
                           optimizer) -> Dict[str, float]:
        """Train one epoch with batching"""
        self.gat_model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'pearson_loss': 0.0,
            'mse_loss': 0.0,
            'cosine_loss': 0.0,
            'weight_reg': 0.0,
            'sparsity_loss': 0.0,
            'diversity_loss': 0.0,
            'hetero_loss': 0.0,
            'proportion_loss': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            batch_st_data = batch['expression'].to(self.device)
            batch_spatial_coords = batch['coords'].to(self.device)
            
            # Compute spot embeddings
            with torch.no_grad():
                mu, log_var = self.vae_encoder(batch_st_data)
                spot_embeddings = mu
            
            # GAT forward pass
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=batch_spatial_coords,
                celltype_prototypes=self.celltype_prototypes
            )
            
            # Compute loss
            loss_outputs = self.loss_fn(
                attention_weights=gat_outputs['deconv_weights'],
                celltype_expression=self.celltype_expressions,
                true_spot_expression=batch_st_data,
                spot_embedding=gat_outputs['spot_features'],
                celltype_embedding=gat_outputs['celltype_features'],
                edge_index=gat_outputs['edge_index']  # 传递edge_index用于计算空间异质性
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_outputs['total_loss'].backward()
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses.keys():
                if key in loss_outputs:
                    epoch_losses[key] += loss_outputs[key].item()
        
        # Compute average losses
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
                               batch_size: int = 512,
                               cells_per_spot: float = 10.0):
        """Train GAT deconvolution model
        
        Args:
            cells_per_spot: Average number of cells per spot (default 10 for Visium)
        """
        print("="*60)
        print("Starting GAT deconvolution training...")
        print(f"Cells per spot: {cells_per_spot}")
        
        # Save st_adata for later use
        self.st_adata = st_adata
 
        # Convert to tensor
        st_tensor = torch.FloatTensor(st_data).to(self.device)
        spatial_tensor = torch.FloatTensor(spatial_coords).to(self.device)
        
        # Create dataset and dataloader
        spot_ids = list(range(len(st_data)))
        dataset = SpatialDataset(st_data, spatial_coords, spot_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5
        )
        
        # Training history
        train_losses = []
        pearson_losses = []
        mse_losses = []
        cos_losses = []
        weight_regs = []
        sparsity_regs = []
        diversity_losses = []
        hetero_losses = []
        proportion_losses = []
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        pbar = tqdm(range(n_epochs), desc="GAT Training", unit="epoch")
        for epoch in pbar:
            # Train one epoch
            epoch_losses = self.train_epoch_batched(
                dataloader=dataloader,
                optimizer=optimizer
            )
            
            # Record average loss
            avg_total_loss = epoch_losses['total_loss']
            train_losses.append(avg_total_loss)
            pearson_losses.append(epoch_losses['pearson_loss'])
            mse_losses.append(epoch_losses['mse_loss'])
            cos_losses.append(epoch_losses['cosine_loss'])
            weight_regs.append(epoch_losses['weight_reg'])
            sparsity_regs.append(epoch_losses.get('sparsity_loss', 0.0))
            diversity_losses.append(epoch_losses.get('diversity_loss', 0.0))
            hetero_losses.append(epoch_losses.get('hetero_loss', 0.0))
            proportion_losses.append(epoch_losses.get('proportion_loss', 0.0))
            
            # Learning rate schedule
            scheduler.step(avg_total_loss)
            
            # Update progress bar with loss info
            pbar.set_postfix({
                'Total': f'{avg_total_loss:.4f}',
                'Pearson': f'{epoch_losses["pearson_loss"]:.4f}',
                'MSE': f'{epoch_losses["mse_loss"]:.4f}',
                'Cosine': f'{epoch_losses["cosine_loss"]:.4f}',
                'Diversity': f'{epoch_losses["diversity_loss"]:.4f}',
                'Hetero': f'{epoch_losses["hetero_loss"]:.4f}',
                'Proportion': f'{epoch_losses["proportion_loss"]:.4f}'
            })
            

            # Save best model
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_model(f"{self.output_dir}/best_gat_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                pbar.close()
                break
        
        # Plot training curves
        self.plot_training_curves(train_losses, pearson_losses, mse_losses, cos_losses, 
                                 weight_regs, sparsity_regs, diversity_losses, hetero_losses, 
                                 proportion_losses, sample_name)
        
        # Save final model
        self.save_model(f"{self.output_dir}/final_gat_model.pth")
        
        # Evaluate and visualize results
        self.evaluate_and_visualize(st_data, self.st_adata, spatial_tensor, sample_name, cells_per_spot)
        
        return {
            'best_loss': best_loss,
            'train_losses': train_losses,
            'sample_name': sample_name
        }
    
    def evaluate_and_visualize(self, 
                             st_data: np.ndarray,
                             st_adata,
                             spatial_coords: torch.Tensor,
                             sample_name: str,
                             cells_per_spot: float = 10.0):
        """Evaluate model and visualize results, generate deconvolution matrices
        
        Args:
            cells_per_spot: Average number of cells per spot (default 10 for Visium)
        """
        print("="*60)
        print("Evaluating model results...")
        print(f"Cells per spot: {cells_per_spot}")
        
        self.gat_model.eval()
        
        st_tensor = torch.FloatTensor(st_data).to(self.device)
        
        with torch.no_grad():
            # Compute spot embeddings
            mu, log_var = self.vae_encoder(st_tensor)
            spot_embeddings = mu
            
            # GAT forward pass
            gat_outputs = self.gat_model(
                spot_embeddings=spot_embeddings,
                spatial_coords=spatial_coords,
                celltype_prototypes=self.celltype_prototypes
            )
            
            # Get prediction results
            deconv_weights = gat_outputs['deconv_weights'].detach().cpu().numpy()
            attention_scores = gat_outputs['attention_scores'].detach().cpu().numpy()
        
        # Apply weight threshold (sparsification)
        print(f"Applying weight threshold: {self.weight_threshold}")
        original_nonzero = np.count_nonzero(deconv_weights)
        deconv_weights[deconv_weights < self.weight_threshold] = 0
        new_nonzero = np.count_nonzero(deconv_weights)
        print(f"   Non-zero elements: {original_nonzero} -> {new_nonzero} ({100*new_nonzero/deconv_weights.size:.1f}%)")
        
        # Renormalize weights to sum to 1 per spot (after thresholding)
        row_sums = deconv_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        deconv_weights = deconv_weights / row_sums
        
        print("Saving deconvolution results...")
        # weights_file = f"{self.output_dir}/{sample_name}_deconv_weights.npz"
        # np.savez(weights_file, 
        #         deconv_weights=deconv_weights,
        #         attention_scores=attention_scores,
        #         clusters=self.label_encoder.classes_)
        
        # ============ Generate expression matrices ============
        n_spots = deconv_weights.shape[0]
        n_clusters = deconv_weights.shape[1]
        
        print("Generating deconvolution expression matrices...")
        
        # Get spot barcodes
        spot_barcodes = list(st_adata.obs.index)
        
        # 1. Marker gene expression matrix (spot × marker genes)
        print("   Marker gene expression...")
        celltype_expr_marker = self.celltype_expressions.cpu().numpy()
        reconstructed_marker_expr = np.dot(deconv_weights, celltype_expr_marker)
        
        marker_expr_df = pd.DataFrame(
            reconstructed_marker_expr,
            columns=self.genes,
            index=spot_barcodes
        )
        marker_expr_file = f"{self.output_dir}/{sample_name}_reconstructed_marker_genes.csv"
        marker_expr_df.to_csv(marker_expr_file)
        
        # 2. Full gene expression matrix (use count version with cells_per_spot)
        if self.celltype_expressions_full is not None and all(expr is not None for expr in self.celltype_expressions_full):
            print("   Full gene expression...")
            celltype_expr_full = np.array(self.celltype_expressions_full)
            
            # Key change: weight represents cell fraction, so multiply by cells_per_spot
            # Example: if weight=0.1 and cells_per_spot=10, means 1 cell of this type
            # Cluster expression is mean count per cell, so 1 cell contributes 1 × cluster_expr
            reconstructed_full_expr = np.dot(deconv_weights * cells_per_spot, celltype_expr_full)
            
            # Get full gene names
            if self.all_genes is not None:
                all_gene_names = self.all_genes
            else:
                all_gene_names = [f"Gene_{i}" for i in range(celltype_expr_full.shape[1])]
            
            full_expr_df = pd.DataFrame(
                reconstructed_full_expr,
                columns=all_gene_names,
                index=spot_barcodes
            )
            full_expr_file = f"{self.output_dir}/{sample_name}_reconstructed_all_genes.csv"
            full_expr_df.to_csv(full_expr_file)
        
        # 3. Cell type composition matrix (spot × cluster/celltype)
        print("   Cell type composition...")
        
        if self.cluster_to_celltype is not None:
            # Celltype mode: only save celltype composition
            columns = [self.cluster_to_celltype[i] for i in range(len(self.label_encoder.classes_))]
            composition_df = pd.DataFrame(
                deconv_weights,
                columns=columns,
                index=spot_barcodes
            )
            composition_file = f"{self.output_dir}/{sample_name}_celltype_composition.csv"
            composition_df.to_csv(composition_file)
            print(f"   Saved celltype composition: {composition_file}")
        else:
            # Auto-cluster mode: map clusters to celltypes
            # First, load the cluster-celltype mapping from checkpoint
            cluster_list = list(self.label_encoder.classes_)
            
            # Get cluster-to-celltype mapping from checkpoint if available
            checkpoint_cluster_to_celltype = {}

            sc_clustered_path = f"{os.path.dirname(self.stage1_model_path)}/sc_adata_clustered.h5ad"
            
            if os.path.exists(sc_clustered_path):
                # Load the clustered adata to get cluster-celltype mapping
                sc_clustered = sc.read_h5ad(sc_clustered_path)
                if 'leiden' in sc_clustered.obs.columns and 'cell_type' in sc_clustered.obs.columns:
                    for cluster_id in sorted(sc_clustered.obs['leiden'].unique()):
                        cluster_mask = sc_clustered.obs['leiden'] == cluster_id
                        celltype_counts = sc_clustered.obs[cluster_mask]['cell_type'].value_counts()
                        major_celltype = celltype_counts.index[0]
                        checkpoint_cluster_to_celltype[str(cluster_id)] = major_celltype
            
            # Map cluster columns to celltype names
            celltype_columns = []
            cluster_columns = []
            for cluster_id in cluster_list:
                cluster_columns.append(str(cluster_id))
                # Use the mapping if available, otherwise use cluster ID as is
                celltype_name = checkpoint_cluster_to_celltype.get(str(cluster_id), f"Cluster_{cluster_id}")
                celltype_columns.append(celltype_name)

            # Create DataFrame with possibly duplicate column names (multiple clusters -> same celltype)
            composition_df = pd.DataFrame(
                deconv_weights,
                columns=celltype_columns,
                index=spot_barcodes
            )

            # 如果存在重复的 celltype 名称，则将对应列合并（按列求和）并记录日志
            dup_names = [name for name in set(celltype_columns) if celltype_columns.count(name) > 1]
            if len(dup_names) > 0:
                print(f"   Found duplicate celltype names: {dup_names}. Merging corresponding cluster columns by summing weights.")
                # groupby on columns will sum duplicated-named columns
                composition_by_celltype = composition_df.groupby(by=composition_df.columns, axis=1).sum()
                print(f"   Columns before: {len(composition_df.columns)}, after merge: {len(composition_by_celltype.columns)}")
            else:
                composition_by_celltype = composition_df

            # Save aggregated celltype composition
            composition_file = f"{self.output_dir}/{sample_name}_cell_composition.csv"
            composition_by_celltype.to_csv(composition_file)
            print(f"   Saved cell composition (celltype): {composition_file}")

            # Also save cluster-level composition (columns are cluster IDs) for reproducibility
            cluster_composition_df = pd.DataFrame(
                deconv_weights,
                columns=cluster_columns,
                index=spot_barcodes
            )
            cluster_file = f"{self.output_dir}/{sample_name}_cluster_composition.csv"
            cluster_composition_df.to_csv(cluster_file)
            print(f"   Saved cluster composition: {cluster_file}")
        
        # ============ Compute reconstruction quality (Cosine Similarity) ============
        print("\nComputing reconstruction quality per spot...")
        
        # Reconstruct spot expression using deconv_weights
        if self.celltype_expressions_full is not None and all(expr is not None for expr in self.celltype_expressions_full):
            # Use full gene expression
            # Check if expressions are tensors or numpy arrays
            if isinstance(self.celltype_expressions_full[0], torch.Tensor):
                celltype_expr_full = np.array([expr.cpu().numpy() for expr in self.celltype_expressions_full])
            else:
                celltype_expr_full = np.array(self.celltype_expressions_full)
            
            reconstructed_full_expr = np.dot(deconv_weights, celltype_expr_full)
            
            # IMPORTANT: 确保 true_expr 和 reconstructed_full_expr 使用相同的基因集
            # celltype_expr_full 对应的是 self.all_genes（从 stage1 加载）
            if hasattr(self, 'all_genes') and self.all_genes is not None:
                # 只提取 all_genes 对应的基因表达
                true_expr_full = st_adata[:, self.all_genes].X
                true_expr = true_expr_full.toarray() if hasattr(true_expr_full, 'toarray') else true_expr_full
                print(f"   Using {len(self.all_genes)} genes for reconstruction quality")
            else:
                # Fallback: 如果没有 all_genes，使用 st_adata 的所有基因（可能不匹配）
                print("   Warning: all_genes not found, using all ST genes (may cause dimension mismatch)")
                true_expr = st_adata.X.toarray() if hasattr(st_adata.X, 'toarray') else st_adata.X
        else:
            # Fall back to marker genes
            celltype_expr_marker = self.celltype_expressions.cpu().numpy()
            reconstructed_full_expr = np.dot(deconv_weights, celltype_expr_marker)
            
            # 使用 marker genes 对应的真实表达
            st_marker_subset = st_adata[:, self.genes].X
            true_expr = st_marker_subset.toarray() if hasattr(st_marker_subset, 'toarray') else st_marker_subset
            print(f"   Using {len(self.genes)} marker genes for reconstruction quality")
        
        # Compute cosine similarity per spot (log-normalized space)
        reconstructed_log = np.log1p(reconstructed_full_expr)
        true_log = np.log1p(true_expr)
        
        cosine_similarities = []
        for i in range(n_spots):
            rec = reconstructed_log[i]
            true = true_log[i]
            
            # Cosine similarity
            cos_sim = np.dot(rec, true) / (np.linalg.norm(rec) * np.linalg.norm(true) + 1e-8)
            cosine_similarities.append(cos_sim)
        
        cosine_similarities = np.array(cosine_similarities)
        
        # Save cosine similarities to CSV
        cosine_df = pd.DataFrame({
            'spot_id': spot_barcodes,
            'cosine_similarity': cosine_similarities
        })
        cosine_csv = f"{self.output_dir}/{sample_name}_spot_cosine_similarity.csv"
        cosine_df.to_csv(cosine_csv, index=False)
        print(f"   Cosine similarities saved: {cosine_csv}")
        
        # Plot reconstruction quality curve (sorted by similarity)
        self.plot_reconstruction_quality_curve(cosine_similarities, sample_name)
        
        # Save results summary
        # results = {
        #     'deconv_weights': deconv_weights,
        #     'attention_scores': attention_scores,
        #     'clusters': list(self.label_encoder.classes_),
        #     'sample_name': sample_name,
        #     'marker_genes': self.genes,
        #     'n_spots': n_spots,
        #     'n_clusters': n_clusters
        # }
        
        # results_file = f"{self.output_dir}/{sample_name}_deconvolution_results.npz"
        # np.savez(results_file, **results)
        # print(f"   Complete results saved: {results_file}")
        
        # ========== 额外保存 CSV 供 train.py 使用 ==========
        print("\nSaving cluster expression data for train.py...")
        
        # 1. 保存 cluster 平均 marker 基因表达 CSV
        cluster_list = list(self.label_encoder.classes_)
        celltype_expr_marker = self.celltype_expressions.cpu().numpy()
        
        marker_expr_df = pd.DataFrame(
            celltype_expr_marker,
            columns=self.genes,
            index=[f"Cluster_{i}" for i in cluster_list]
        )
        marker_expr_file = f"{self.output_dir}/{sample_name}_cluster_marker_expr.csv"
        marker_expr_df.to_csv(marker_expr_file)
        print(f"   ✅ Cluster marker gene expression: {marker_expr_file}")
        
        # 2. 保存 cluster 平均全基因表达 CSV
        if self.celltype_expressions_full is not None and all(expr is not None for expr in self.celltype_expressions_full):
            celltype_expr_full = np.array(self.celltype_expressions_full)
            
            # Get full gene names
            if self.all_genes is not None:
                all_gene_names = self.all_genes
            else:
                all_gene_names = [f"Gene_{i}" for i in range(celltype_expr_full.shape[1])]
            
            full_expr_df = pd.DataFrame(
                celltype_expr_full,
                columns=all_gene_names,
                index=[f"Cluster_{i}" for i in cluster_list]
            )
            full_expr_file = f"{self.output_dir}/{sample_name}_cluster_full_expr.csv"
            full_expr_df.to_csv(full_expr_file)
            print(f"   ✅ Cluster full gene expression: {full_expr_file}")
        
        # 3. 保存 celltype-cluster 映射 TXT
        if self.cluster_to_celltype is not None:
            # Celltype mode: 直接从 cluster_to_celltype 映射保存
            mapping_file = f"{self.output_dir}/{sample_name}_celltype_cluster_mapping.txt"
            with open(mapping_file, 'w') as f:
                f.write("cluster_id\tcelltype_name\n")
                for cluster_id in cluster_list:
                    celltype_name = self.cluster_to_celltype.get(cluster_id, f"Cluster_{cluster_id}")
                    f.write(f"{cluster_id}\t{celltype_name}\n")
            print(f"   ✅ Celltype-cluster mapping: {mapping_file}")
        else:
            # Auto-cluster mode: 从 checkpoint 加载映射
            checkpoint_cluster_to_celltype = {}

            sc_clustered_path = f"{os.path.dirname(self.stage1_model_path)}/sc_adata_clustered.h5ad"
            
            if os.path.exists(sc_clustered_path):
                sc_clustered = sc.read_h5ad(sc_clustered_path)
                if 'leiden' in sc_clustered.obs.columns and 'cell_type' in sc_clustered.obs.columns:
                    for cluster_id in sorted(sc_clustered.obs['leiden'].unique()):
                        cluster_mask = sc_clustered.obs['leiden'] == cluster_id
                        celltype_counts = sc_clustered.obs[cluster_mask]['cell_type'].value_counts()
                        major_celltype = celltype_counts.index[0]
                        checkpoint_cluster_to_celltype[str(cluster_id)] = major_celltype
            
            mapping_file = f"{self.output_dir}/{sample_name}_celltype_cluster_mapping.txt"
            with open(mapping_file, 'w') as f:
                f.write("cluster_id\tcelltype_name\n")
                for cluster_id in cluster_list:
                    celltype_name = checkpoint_cluster_to_celltype.get(str(cluster_id), f"Cluster_{cluster_id}")
                    f.write(f"{cluster_id}\t{celltype_name}\n")
            print(f"   ✅ Celltype-cluster mapping: {mapping_file}")
    
    def plot_reconstruction_quality_curve(self, cosine_similarities, sample_name):
        """Plot reconstruction quality curve (sorted by cosine similarity)
        
        Args:
            cosine_similarities: Array of cosine similarities per spot [n_spots]
            sample_name: Sample name for saving
        """
        print("\nPlotting reconstruction quality curve...")
        
        # Sort cosine similarities in ascending order
        sorted_similarities = np.sort(cosine_similarities)
        n_spots = len(sorted_similarities)
        spot_indices = np.arange(1, n_spots + 1)  # 1-based indexing
        
        # Compute statistics
        mean_sim = np.mean(cosine_similarities)
        median_sim = np.median(cosine_similarities)
        q25_sim = np.percentile(cosine_similarities, 25)
        q75_sim = np.percentile(cosine_similarities, 75)
        min_sim = np.min(cosine_similarities)
        max_sim = np.max(cosine_similarities)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot curve
        ax.plot(spot_indices, sorted_similarities, linewidth=2.5, color='#1f77b4', label='Cosine Similarity')
        
        # Add reference lines
        ax.axhline(mean_sim, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_sim:.3f}')
        ax.axhline(median_sim, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_sim:.3f}')
        ax.axhline(0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)  # Reference line at 0.8
        
        # Styling
        ax.set_xlabel('Spot Index (sorted by similarity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'Reconstruction Quality Distribution - {sample_name}', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        ax.set_xlim([0, n_spots])
        
        # Add text box with statistics
        textstr = '\n'.join([
            f'Total spots: {n_spots}',
            f'Min: {min_sim:.3f}',
            f'Q25: {q25_sim:.3f}',
            f'Median: {median_sim:.3f}',
            f'Q75: {q75_sim:.3f}',
            f'Max: {max_sim:.3f}',
            f'Mean: {mean_sim:.3f}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        output_file = f"{self.output_dir}/{sample_name}_reconstruction_quality_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   Reconstruction quality curve saved: {output_file}")
        plt.close()
        
        # Print summary
        print(f"\n   Reconstruction Quality Summary:")
        print(f"      Mean cosine similarity: {mean_sim:.4f}")
        print(f"      Median cosine similarity: {median_sim:.4f}")
        print(f"      Range: [{min_sim:.4f}, {max_sim:.4f}]")
        print(f"      Spots with similarity > 0.8: {np.sum(cosine_similarities > 0.8)} ({100*np.sum(cosine_similarities > 0.8)/n_spots:.1f}%)")
    
    def plot_training_curves(self, train_losses, pearson_losses, mse_losses, cos_losses, 
                           weight_regs, sparsity_regs, diversity_losses, hetero_losses, 
                           proportion_losses, sample_name):
        """Plot training curves"""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction losses
        axes[0, 1].plot(epochs, pearson_losses, 'orange', label='Pearson', linewidth=2)
        axes[0, 1].plot(epochs, mse_losses, 'green', label='MSE', linewidth=2)
        axes[0, 1].plot(epochs, cos_losses, 'red', label='Cosine', linewidth=2)
        axes[0, 1].set_title('Reconstruction Losses', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pearson loss
        axes[0, 2].plot(epochs, pearson_losses, 'orange', linewidth=2)
        axes[0, 2].set_title('Pearson Loss', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # MSE loss
        axes[1, 0].plot(epochs, mse_losses, 'green', linewidth=2)
        axes[1, 0].set_title('MSE Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diversity and Heterogeneity losses
        axes[1, 1].plot(epochs, diversity_losses, 'cyan', label='Diversity', linewidth=2)
        axes[1, 1].plot(epochs, hetero_losses, 'magenta', label='Heterogeneity', linewidth=2)
        axes[1, 1].set_title('Diversity & Heterogeneity', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Heterogeneity loss (单独显示)
        axes[1, 2].plot(epochs, hetero_losses, 'magenta', linewidth=2)
        axes[1, 2].set_title('Spatial Heterogeneity Loss', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epochs')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Weight regularization
        axes[2, 0].plot(epochs, weight_regs, 'brown', linewidth=2)
        axes[2, 0].set_title('Weight Regularization', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Epochs')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Sparsity regularization
        axes[2, 1].plot(epochs, sparsity_regs, 'purple', linewidth=2)
        axes[2, 1].set_title('Sparsity Regularization', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Epochs')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].grid(True, alpha=0.3)
        
        # All regularizations comparison
        axes[2, 2].plot(epochs, weight_regs, 'brown', label='Weight', linewidth=2)
        axes[2, 2].plot(epochs, sparsity_regs, 'purple', label='Sparsity', linewidth=2)
        axes[2, 2].plot(epochs, diversity_losses, 'cyan', label='Diversity', linewidth=2)
        axes[2, 2].plot(epochs, hetero_losses, 'magenta', label='Heterogeneity', linewidth=2)
        axes[2, 2].plot(epochs, proportion_losses, 'olive', label='Proportion', linewidth=2)
        axes[2, 2].set_title('All Regularizations', fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('Epochs')
        axes[2, 2].set_ylabel('Loss')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'GAT Deconvolution Training - {sample_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gat_training_curves_{sample_name}.png", dpi=300, bbox_inches='tight')
        # plt.show()
    
    def save_model(self, filepath: str):
        """Save model"""
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
    """Main function"""
    parser = argparse.ArgumentParser(description='Stage 2: GAT Deconvolution for Spatial Transcriptomics')
    # Model and data arguments
    parser.add_argument('--stage1_model_path', type=str, 
                       default="./stage1_results/final_vae.pth",
                       help='Stage 1 VAE model path')
    parser.add_argument('--st_file', type=str, required=True,
                       help='Spatial transcriptomics data file path (.h5ad)')
    parser.add_argument('--output_dir', type=str, 
                       default="./stage2_results",
                       help='Output directory path')

    # GAT model arguments
    parser.add_argument('--gat_hidden_dim', type=int, default=64,
                       help='GAT hidden layer dimension')
    parser.add_argument('--gat_layers', type=int, default=3,
                       help='Number of GAT layers')
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of GAT attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Clustering arguments
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution')
    
    # Graph construction arguments
    parser.add_argument('--k_spatial', type=int, default=6,
                       help='Number of spatial neighbors (KNN)')
    parser.add_argument('--k_celltype', type=int, default=10,
                       help='Number of nearest celltypes per spot (KNN)')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    
    # Loss function arguments
    parser.add_argument('--loss_lambda_pearson', type=float, default=1.0,
                       help='Pearson correlation loss weight')
    parser.add_argument('--loss_lambda_mse', type=float, default=1.0,
                       help='MSE reconstruction loss weight')
    parser.add_argument('--loss_lambda_cosine', type=float, default=1.0,
                       help='Cosine similarity loss weight')
    parser.add_argument('--loss_lambda_reg', type=float, default=0.5,
                       help='Weight regularization weight')
    parser.add_argument('--loss_lambda_sparse', type=float, default=0.01,
                       help='Sparsity regularization weight (Shannon entropy)')
    parser.add_argument('--loss_lambda_diversity', type=float, default=0.1,
                       help='Diversity loss weight (prevents all spots using same celltype)')
    parser.add_argument('--loss_lambda_hetero', type=float, default=0.05,
                       help='Spatial heterogeneity loss weight (preserves spatial variation)')
    parser.add_argument('--loss_lambda_proportion', type=float, default=1.0,
                       help='Global cell type proportion consistency loss weight (matches SC cluster distribution)')
    
    # Spot composition argument
    parser.add_argument('--cells_per_spot', type=float, default=10.0,
                       help='Average number of cells per spot (default 10 for Visium)')
    
    # Weight thresholding argument
    parser.add_argument('--weight_threshold', type=float, default=0.01,
                       help='Weight threshold for sparsification (default 0.01, i.e., 1%)')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.st_file):
        raise FileNotFoundError(f"ST data file not found: {args.st_file}")
    
    # Extract sample name
    sample_name = os.path.splitext(os.path.basename(args.st_file))[0]
    if sample_name.endswith('_ST'):
        sample_name = sample_name[:-3]
    
    print(f"Sample name: {sample_name}")
 
    # Initialize trainer
    trainer = GATDeconvolution(
        stage1_model_path=args.stage1_model_path,
        output_dir=args.output_dir,
        device=args.device,
        weight_threshold=args.weight_threshold
    )
    
    # Set graph construction parameters
    trainer.k_spatial = args.k_spatial
    trainer.k_celltype = args.k_celltype
    
    # Load VAE Encoder
    trainer.load_vae_encoder()

    # Stage 2 does not need to load SC data
    # All cluster info (centers, expressions, encoder) already computed in stage1
    print("Using Stage 1 cluster centers and expressions...")
    
    if trainer.sc_clusters is None:
        raise ValueError("Stage 1 model missing cluster information! Please retrain with new version.")
    
    print(f"Loaded {len(trainer.label_encoder.classes_)} clusters")
    
    # Check if pretrained cluster data available
    has_prototypes = hasattr(trainer, 'celltype_prototypes') and trainer.celltype_prototypes is not None
    has_expressions = hasattr(trainer, 'celltype_expressions') and trainer.celltype_expressions is not None
    
    if has_prototypes and has_expressions:
        print("Using Stage 1 pretrained cluster data")
        print(f"   Cluster centers: {trainer.celltype_prototypes.shape}")
        print(f"   Cluster expressions: {trainer.celltype_expressions.shape}")
        n_clusters = trainer.celltype_prototypes.shape[0]
    else:
        raise ValueError("Stage 1 model missing cluster centers or expressions!")
    
    print("="*60)
    print("Loading and processing spatial transcriptomics data...")
    
    # Load ST data
    print(f"Loading ST data: {args.st_file}")
    st_adata = sc.read_h5ad(args.st_file)
    st_adata.var_names_make_unique()  # Handle duplicate gene names

    # Check spatial coordinates
    if 'spatial' not in st_adata.obsm:
        raise ValueError(f"ST data file {args.st_file} missing spatial coordinates! ST data must contain 'spatial' coordinates (adata.obsm['spatial']).")
    
    # Extract ST marker genes
    st_subset = st_adata[:, trainer.genes].copy()
    print(f"ST matching genes: {len(trainer.genes)}/{len(trainer.genes)}")
    
    # Extract ST data
    # sc.pp.normalize_total(st_subset, target_sum=1e4)
    # sc.pp.log1p(st_subset)
    st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
    
    # Extract spatial coordinates
    spatial_coords = st_adata.obsm['spatial']
    
    print(f"ST data: {st_X.shape}")
  
    
    # Build GAT model and loss function
    print("="*60)
    print(f"Building GAT deconvolution model (clusters: {n_clusters})...")
    trainer.build_gat_model(
        n_cell_types=n_clusters,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        loss_lambda_pearson=args.loss_lambda_pearson,
        loss_lambda_mse=args.loss_lambda_mse,
        loss_lambda_cosine=args.loss_lambda_cosine,
        loss_lambda_reg=args.loss_lambda_reg,
        loss_lambda_sparse=args.loss_lambda_sparse,
        loss_lambda_diversity=args.loss_lambda_diversity,
        loss_lambda_hetero=args.loss_lambda_hetero,
        loss_lambda_proportion=args.loss_lambda_proportion
    )
    
    # Start training
    print("="*60)
    print("Starting GAT deconvolution training...")
    
    trainer.train_gat_deconvolution(
        st_data=st_X,
        spatial_coords=spatial_coords,
        sample_name=sample_name,
        st_adata=st_adata,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        cells_per_spot=args.cells_per_spot
    )
    
    print("="*60)
    print("GAT deconvolution training completed!")

if __name__ == "__main__":
    main()