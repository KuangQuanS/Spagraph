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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
import argparse
import warnings
import umap
from scipy import sparse
warnings.filterwarnings('ignore')

# Import model and utilities from deconv_model (which now includes vae_utils)
from .deconv_model import (
    VAE, DualDecoderVAE,
    train_vae, evaluate_vae, save_vae_checkpoint, 
    load_vae_pretrained, load_vae_for_inference, 
    plot_modality_alignment_umap
)

# Import stage1 utility functions
from .stage1_utils import (
    load_marker_genes_from_file,
    compute_clusters_and_marker_genes,
    compute_cluster_centers_and_expressions,
    print_weight_statistics
)

def compute_clusters_and_marker_genes_deprecated(adata, top_n=100, min_fold_change=1.5, resolution=0.5, save_path=None):
    """
    DEPRECATED: Use stage1_utils.compute_clusters_and_marker_genes instead
    """
    raise DeprecationWarning("This function has been moved to stage1_utils.py")

#============================================================
# Main Module
#============================================================
class coEncoder:
    def __init__(self, 
                 sc_file=None,
                 st_file=None,
                 output_dir="./stage1_results",
                 celltype_key=None,
                 device=None,
                 save_to_disk=True,
                 seed=42):
        """
        Initialize co-encoder
        
        Args:
            sc_file: Path to single-cell h5ad file
            st_file: Path to spatial transcriptomics h5ad file
            output_dir: Output directory path (if save_to_disk=False, only used for temp files)
            celltype_key: Optional ``scRNA.obs`` column containing reference
                cell-type annotations. If omitted, ``cell_type`` and then
                ``celltype`` are detected for backward compatibility.
            device: Computing device (cuda/cpu, None for auto)
            save_to_disk: Whether to save models and data to disk. If False, only returns in-memory objects
            seed: Random seed for reproducibility
        """
        # Set random seed first
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.sc_file = sc_file
        self.st_file = st_file
        self.output_dir = output_dir
        self.requested_celltype_key = celltype_key
        self.save_to_disk = save_to_disk
        self.seed = seed
        
        if save_to_disk and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model components
        self.vae = None
        self.label_encoder = None
        self.marker_genes = None

    def _resolve_celltype_key(self, adata: ad.AnnData, required: bool = False) -> Optional[str]:
        """Return the configured reference-annotation column, if available."""
        if self.requested_celltype_key is not None:
            if self.requested_celltype_key not in adata.obs.columns:
                raise ValueError(
                    f"celltype_key={self.requested_celltype_key!r} is not present in scRNA.obs"
                )
            return self.requested_celltype_key
        for candidate in ('cell_type', 'celltype'):
            if candidate in adata.obs.columns:
                return candidate
        if required:
            raise ValueError(
                "cell-type annotations are required; pass celltype_key or add "
                "scRNA.obs['cell_type'] / scRNA.obs['celltype']"
            )
        return None
        
    def load_data(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """Load SC and ST data from specified files"""
        if self.sc_file is None or self.st_file is None:
            raise ValueError("SC file and ST file must be specified!")
        
        if not os.path.exists(self.sc_file):
            raise FileNotFoundError(f"SC file not found: {self.sc_file}")
        
        if not os.path.exists(self.st_file):
            raise FileNotFoundError(f"ST file not found: {self.st_file}")
        
        sc_adata = sc.read_h5ad(self.sc_file)
        sc_adata.obs['modality'] = 'SC'
        sc_adata.var_names_make_unique()
        
        # Load ST data
        st_adata = sc.read_h5ad(self.st_file)
        st_adata.obs['modality'] = 'ST'
        st_adata.var_names_make_unique()
        
        # Find common genes
        common_genes = list(set(sc_adata.var_names) & set(st_adata.var_names))
        common_genes.sort()
        
        # Subset to common genes
        sc_adata = sc_adata[:, common_genes].copy()
        st_adata = st_adata[:, common_genes].copy()
        
        return sc_adata, st_adata

    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100, resolution: float = 0.5,
                               precomputed_marker_file: str = None, marker_selection_method: str = 'l1') -> Tuple:
        """Prepare training data based on marker genes
        
        Args:
            sc_adata: Single cell AnnData object
            st_adata: Spatial transcriptomics AnnData object
            top_n_per_type: Number of marker genes per cluster
            resolution: Leiden resolution for auto-clustering
            precomputed_marker_file: Path to precomputed marker genes file.
                                   If provided, will load marker genes directly from this file
                                   instead of computing them from scratch.
            marker_selection_method: Marker selection method. In addition to
                Leiden-based methods, ``celltype_specific`` builds a balanced
                marker union from supplied single-cell annotations.
        """

        # Step 0: Compute auto_library_size first (using raw counts)
        try:
            # 复制原始数据用于HVG计算
            sc_hvg_adata = sc_adata.copy()
            sc.pp.normalize_total(sc_hvg_adata, target_sum=1e4)
            sc.pp.log1p(sc_hvg_adata)
            sc.pp.highly_variable_genes(
                sc_hvg_adata,
                n_top_genes=min(2000, sc_hvg_adata.shape[1]),
                flavor="seurat"
            )
            sc_hvg_genes = list(sc_hvg_adata.var.index[sc_hvg_adata.var["highly_variable"]])
            
            if len(sc_hvg_genes) > 0:
                # 找出在ST中也存在的HVG
                sc_hvg_in_st = [g for g in sc_hvg_genes if g in st_adata.var_names]
                
                if len(sc_hvg_in_st) > 0:
                    # 用原始count计算总和
                    sc_hvg_subset = sc_adata[:, sc_hvg_in_st]
                    sc_hvg_raw = sc_hvg_subset.X.toarray() if hasattr(sc_hvg_subset.X, 'toarray') else sc_hvg_subset.X
                    sc_hvg_total_count = sc_hvg_raw.sum()
                    
                    st_hvg_subset = st_adata[:, sc_hvg_in_st]
                    st_hvg_raw = st_hvg_subset.X.toarray() if hasattr(st_hvg_subset.X, 'toarray') else st_hvg_subset.X
                    st_hvg_total_count = st_hvg_raw.sum()
                    
                    if st_hvg_total_count > 0 and sc_hvg_total_count > 0:
                        self.auto_library_size =  sc_hvg_total_count / st_hvg_total_count
                        print(f"Auto-computed library_size factor: {self.auto_library_size:.4f}")
                        print(f"   - SC HVG total count: {sc_hvg_total_count:.0f}")
                        print(f"   - ST HVG total count: {st_hvg_total_count:.0f}")
                        print(f"   - HVG genes used: {len(sc_hvg_in_st)}")
                    else:
                        self.auto_library_size = 1.0
                        print("Warning: Cannot compute library_size (count=0), using default value 1.0")
                else:
                    self.auto_library_size = 1.0
                    print("Warning: No SC HVG intersection in ST, using default library_size 1.0")
            else:
                self.auto_library_size = 1.0
                print("Warning: SC HVG is empty, using default library_size 1.0")
        except Exception as e:
            self.auto_library_size = 1.0
            print(f"Warning: Failed to compute library_size: {e}, using default value 1.0")

        # 1. Compute clusters and marker genes  
        if precomputed_marker_file is not None:
            # Load precomputed marker genes
            self.marker_genes = load_marker_genes_from_file(precomputed_marker_file)
            # For compatibility, create dummy clustering results
            # We'll use a simple clustering approach for the rest of the pipeline
            sc_adata_clustered = sc_adata.copy()
            sc.pp.normalize_total(sc_adata_clustered, target_sum=1e4)
            sc.pp.log1p(sc_adata_clustered)
            sc.pp.highly_variable_genes(sc_adata_clustered, min_mean=0.0125, max_mean=3, min_disp=0.5)
            sc_adata_clustered.raw = sc_adata_clustered
            sc_adata_clustered = sc_adata_clustered[:, sc_adata_clustered.var.highly_variable]
            sc.pp.scale(sc_adata_clustered, max_value=10)
            sc.tl.pca(sc_adata_clustered, svd_solver='arpack')
            sc.pp.neighbors(sc_adata_clustered, n_neighbors=10, n_pcs=40)
            sc.tl.leiden(sc_adata_clustered, resolution=resolution)
            sc_clusters = sc_adata_clustered.obs['leiden'].copy()
            print(f"   Number of clusters: {len(sc_clusters.unique())}")
        elif marker_selection_method == 'celltype_specific':
            celltype_col = self._resolve_celltype_key(sc_adata, required=True)
            annotations = sc_adata.obs[celltype_col]
            if annotations.isna().any():
                raise ValueError("cell-type annotations cannot contain missing values")
            labels = annotations.astype(str).to_numpy()
            encoded = LabelEncoder().fit_transform(labels)
            expression = sc_adata.X
            totals = np.asarray(expression.sum(axis=1)).ravel()
            inverse = np.zeros_like(totals, dtype=np.float64)
            inverse[totals > 0] = 1.0 / totals[totals > 0]
            if sparse.issparse(expression):
                normalized = sparse.diags(inverse) @ expression
            else:
                normalized = np.asarray(expression, dtype=np.float64) * inverse[:, None]
            signatures = np.vstack([
                np.asarray(normalized[encoded == group].mean(axis=0)).ravel()
                for group in np.unique(encoded)
            ])
            from .deconv_initialization import select_celltype_specific_genes

            selected = select_celltype_specific_genes(
                signatures, top_per_celltype=top_n_per_type
            )
            self.marker_genes = [str(sc_adata.var_names[index]) for index in selected]

            # The selected genes guide VAE integration, while Leiden labels are
            # retained for backward-compatible Stage 1 artifacts. Stage 2 can
            # independently aggregate the reference by supplied annotations.
            clustered = sc_adata.copy()
            sc.pp.normalize_total(clustered, target_sum=1e4)
            sc.pp.log1p(clustered)
            sc.pp.highly_variable_genes(
                clustered,
                n_top_genes=min(2000, clustered.n_vars),
                flavor='seurat',
            )
            hvg = clustered[:, clustered.var['highly_variable']].copy()
            sc.pp.scale(hvg, max_value=10)
            n_comps = max(2, min(40, hvg.n_obs - 1, hvg.n_vars - 1))
            sc.tl.pca(hvg, n_comps=n_comps, svd_solver='arpack')
            sc.pp.neighbors(hvg, n_neighbors=min(10, hvg.n_obs - 1), n_pcs=n_comps)
            sc.tl.leiden(hvg, resolution=resolution)
            sc_adata_clustered = sc_adata.copy()
            sc_adata_clustered.obs['leiden'] = hvg.obs['leiden'].copy()
            sc_clusters = sc_adata_clustered.obs['leiden'].copy()
            print(
                f"Selected {len(self.marker_genes)} balanced annotation-specific genes "
                f"from {len(np.unique(labels))} cell types"
            )
        else:
            # Auto-cluster using Leiden
            cluster_save_path = f"{self.output_dir}/marker_genes.txt" if (self.save_to_disk and self.output_dir) else None
            self.marker_genes, sc_clusters, sc_adata_clustered = compute_clusters_and_marker_genes(
                sc_adata.copy(), 
                top_n=top_n_per_type, 
                resolution=resolution,
                save_path=cluster_save_path,
                marker_selection_method=marker_selection_method,
                min_cells_per_cluster=2
            )

        # Save clustered adata for annotation
        self.sc_adata_clustered = sc_adata_clustered
        # cluster_adata_file = f"{self.output_dir}/sc_adata_clustered.h5ad"
        # sc_adata_clustered.write_h5ad(cluster_adata_file)

        # Save clustering info and resolution
        self.sc_clusters = sc_clusters
        self.resolution = resolution
        
        # 2. Process SC data (extract marker genes then normalize + log1p)
        # ✅ First filter to match sc_clusters.index (some cells may be filtered during clustering)
        sc_adata_count = sc_adata[sc_clusters.index].copy()

        # Preserve supplied SC reference annotations aligned to all in-memory
        # artifacts. This is reference information, not spot-composition truth.
        celltype_col = self._resolve_celltype_key(sc_adata_count)
        if celltype_col is not None:
            sc_celltype_labels = sc_adata_count.obs[celltype_col].astype(str).to_numpy()
            self.celltype_key = celltype_col
        else:
            sc_celltype_labels = None
            self.celltype_key = None
        
        # Extract raw counts for all genes (for later scaling and expression reconstruction)
        sc_all_genes_raw = sc_adata_count.X.toarray() if hasattr(sc_adata_count.X, 'toarray') else sc_adata_count.X

        # Normalize and log1p transform for VAE training
        sc.pp.normalize_total(sc_adata_count, target_sum=1e4)
        sc.pp.log1p(sc_adata_count)
  
        # Extract full gene expression (log1p norm) for training features (not used anymore)
        sc_X_full_count = sc_adata_count.X.toarray() if hasattr(sc_adata_count.X, 'toarray') else sc_adata_count.X
        sc_all_genes = list(sc_adata_count.var.index)
        
        # Extract marker genes (log1p norm for training)
        sc_subset = sc_adata_count[:, sc_adata_count.var.index.isin(self.marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_clusters.values
  
        # Encode labels
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)

        # ST (log1p on library-normalized counts)
        st_raw_all = st_adata.X.toarray() if hasattr(st_adata.X, 'toarray') else st_adata.X
        st_proc = st_adata.copy()
        sc.pp.normalize_total(st_proc, target_sum=1e4)
        sc.pp.log1p(st_proc)

        available_genes = [g for g in self.marker_genes if g in st_proc.var.index]
        st_subset = st_proc[:, available_genes].copy()
        
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X

        # Compute SC/ST HVG intersection (retained for Stage2 scale_basis='hvg')
        try:
            # SC HVG
            sc_hvg = sc_adata_count.copy()
            sc.pp.highly_variable_genes(
                sc_hvg,
                n_top_genes=min(3000, sc_hvg.shape[1]),
                flavor="seurat"
            )
            sc_hvg_genes = list(sc_hvg.var.index[sc_hvg.var["highly_variable"]])

            # ST HVG
            st_hvg = st_proc.copy()
            sc.pp.highly_variable_genes(
                st_hvg,
                n_top_genes=min(3000, st_hvg.shape[1]),
                flavor="seurat"
            )
            st_hvg_genes = list(st_hvg.var.index[st_hvg.var["highly_variable"]])

            # SC/ST HVG 交集（用于Stage2的scale_basis='hvg'）
            hvg_intersection = list(sorted(set(sc_hvg_genes) & set(st_hvg_genes)))
            self.hvg_genes_union = hvg_intersection
        except Exception as e:
            self.hvg_genes_union = None
            print(f"Warning: Failed to compute HVG intersection: {e}")
        
        # 4. Ensure SC and ST feature dimensions are consistent
        final_genes = [g for g in self.marker_genes 
                      if g in sc_subset.var.index and g in st_subset.var.index]
        
        sc_gene_indices = [list(sc_subset.var.index).index(g) for g in final_genes]
        st_gene_indices = [list(st_subset.var.index).index(g) for g in final_genes]
        
        sc_X_final = sc_X[:, sc_gene_indices]
        st_X_final = st_X[:, st_gene_indices]
        
        # 5. Split data for training (with test set for early stopping)
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.05, stratify=sc_y, random_state=self.seed
        )
        
        st_train, st_test = train_test_split(
            st_X_final, test_size=0.05, random_state=self.seed
        )
        
        # 6. Combine train and test sets
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

        # Build label arrays for conditional MMD (SC: cluster id, ST: -1)
        train_labels = np.concatenate([y_train, np.full(len(st_train), -1)])
        test_labels = np.concatenate([y_test, np.full(len(st_test), -1)])
        
        # Keep ALL SC data (train + test) for cluster embedding computation
        # Note: Save raw counts for all genes (for later scaling and expression reconstruction)
        # sc_all_genes_raw: [n_cells, n_all_genes] raw counts for all genes
        sc_all_indices = np.arange(len(sc_X_final))  # All indices
        sc_all_labels = sc_y  # All labels
        
        # Save gene list
        self.genes = final_genes
        self.all_genes = sc_all_genes  # Save all gene list
        
        # Only save files when save_to_disk=True
        if self.save_to_disk and self.output_dir:
            genes_file = f"{self.output_dir}/final_genes.txt"
            with open(genes_file, 'w') as f:
                for gene in self.genes:
                    f.write(f"{gene}\n")

        return (
            train_X,
            test_X,
            train_modality,
            test_modality,
            train_labels,
            test_labels,
            y_train,
            y_test,
            sc_X_final,
            sc_all_genes_raw,
            sc_all_labels,
            sc_celltype_labels,
        )
    
    def build_vae(self, input_dim: int, hidden_dims=[512, 256], latent_dim=128, dropout=0.2, loss_type='mse', use_dual_decoder=False):
        """Build VAE model
        
        Args:
            use_dual_decoder: If True, use DualDecoderVAE (shared encoder + separate SC/ST decoders)
                            If False, use standard VAE (single encoder + single decoder)
        """
        # 根据loss_type设置output_type
        output_type = 'zinb' if loss_type == 'zinb' else 'mse'
        
        if use_dual_decoder:
            # 使用双解码器架构
            self.vae = DualDecoderVAE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                output_type=output_type
            ).to(self.device)
        else:
            # 使用标准VAE
            self.vae = VAE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                output_type=output_type
            ).to(self.device)
    
    def save_vae(self, filepath):
        """Save VAE model (only model weights and basic info, no cluster data)"""
        save_vae_checkpoint(
            vae=self.vae,
            filepath=filepath,
            label_encoder=self.label_encoder,
            marker_genes=self.marker_genes,
            genes=self.genes,
            sc_clusters=getattr(self, 'sc_clusters', None),
            resolution=getattr(self, 'resolution', 0.5),
            all_genes=getattr(self, 'all_genes', None)
        )
    
    def save_cluster_data(self, filepath):
        """Save cluster data to separate NPZ file"""
        
        # Prepare cluster data - convert dicts to arrays
        cluster_ids = np.arange(len(self.label_encoder.classes_))
        
        # Convert cluster_prototypes dict to array
        if isinstance(self.cluster_prototypes, dict):
            cluster_ids_sorted = sorted(self.cluster_prototypes.keys())
            prototypes_array = np.stack([self.cluster_prototypes[cid] for cid in cluster_ids_sorted], axis=0)
        else:
            prototypes_array = self.cluster_prototypes
        
        # Convert cluster_expressions dict to array
        if isinstance(self.cluster_expressions, dict):
            cluster_ids_sorted = sorted(self.cluster_expressions.keys())
            expressions_array = np.stack([self.cluster_expressions[cid] for cid in cluster_ids_sorted], axis=0)

        # Convert cluster_expressions_full_count dict to list of arrays
        if isinstance(self.cluster_expressions_full_count, dict):
            cluster_ids_sorted = sorted(self.cluster_expressions_full_count.keys())
            expressions_full_list = [self.cluster_expressions_full_count[cid] for cid in cluster_ids_sorted]
  
        # Prepare celltype mapping as structured array if available
        if self.cluster_to_celltype is not None:
            celltype_mapping = np.array(
                [(int(k), str(v)) for k, v in self.cluster_to_celltype.items()],
                dtype=[('cluster_id', 'i4'), ('celltype', 'U100')]
            )
        else:
            celltype_mapping = None
        
        # Save to NPZ
        save_dict = {
            'cluster_ids': cluster_ids,
            'cluster_prototypes': prototypes_array,
            'cluster_expressions': expressions_array,
            'cluster_expressions_full': np.array(expressions_full_list, dtype=object),
        }
        
        if celltype_mapping is not None:
            save_dict['cluster_to_celltype'] = celltype_mapping
        
        if hasattr(self, 'cluster_cell_weights') and self.cluster_cell_weights is not None:
            save_dict['cluster_cell_weights'] = self.cluster_cell_weights

        # 可选：保存 SC/ST 3000-HVG 交集，供 Stage 2 读取
        if hasattr(self, 'hvg_genes_union') and self.hvg_genes_union is not None:
            save_dict['hvg_genes_union'] = np.array(self.hvg_genes_union, dtype=object)
        
        np.savez(filepath, **save_dict)
    
    def load_vae(self, filepath):
        """Load VAE model (basic loading for inference) using vae_io module"""
        self.vae, loaded_data = load_vae_for_inference(filepath, self.device)
        
        self.label_encoder = loaded_data['label_encoder']
        self.marker_genes = loaded_data['marker_genes']
        self.genes = loaded_data['genes']
    
    def load_pretrained(self, filepath):
        """Load pretrained VAE weights for continued training using vae_io module"""
        self.vae, components, output_type, latent_dim = load_vae_pretrained(filepath, self.device)
        
        # Unpack loaded components
        self.label_encoder = components['label_encoder']
        self.marker_genes = components['marker_genes']
        self.genes = components['genes']
        self.all_genes = components['all_genes']
        self.sc_clusters = components['sc_clusters']
        self.resolution = components['resolution']
        self.cluster_prototypes = components['cluster_prototypes']
        self.cluster_expressions = components['cluster_expressions']
        self.cluster_expressions_full = components['cluster_expressions_full']
        self.cluster_expressions_full_count = components['cluster_expressions_full_count']
        self.cluster_cell_weights = components.get('cluster_cell_weights', None)
        
        return output_type, latent_dim
    
    def run_stage1_training(self, top_n_per_type=100, resolution=0.5, batch_size=256, n_epochs=100, 
                           lr=1e-3, beta=1.0, hidden_dims=[512, 256], latent_dim=128, loss_type='mse', 
                           lambda_mmd=0.0, pretrained_path=None, precomputed_marker_file=None, use_dual_decoder=False,
                           aggregation_method='weighted', marker_selection_method='l1', print_every=50):
        """Run stage 1 training: VAE on SC + ST with marker genes
        
        Args:
            top_n_per_type: Number of marker genes per cluster
            resolution: Leiden resolution for auto-clustering
            batch_size: Training batch size
            n_epochs: Number of training epochs
            lr: Learning rate
            beta: KL divergence weight
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            loss_type: 'mse' or 'zinb'
            lambda_mmd: MMD loss weight
            pretrained_path: Path to pretrained checkpoint
            precomputed_marker_file: Path to precomputed marker genes file.
                                   If provided, will load marker genes directly from this file
                                   instead of computing them from scratch.
            use_dual_decoder: If True, use DualDecoderVAE with separate SC/ST decoders + MMD alignment
            aggregation_method: Cluster aggregation method: 'mean', 'median', or 'weighted'
                - 'mean': Simple average (fast, basic)
                - 'median': Median aggregation (robust to outliers)
                - 'weighted': Weighted average with UMI, representativeness, and marker activity (recommended)
            marker_selection_method: Method for marker gene selection ('l1',
                'variance', 'correlation', or annotation-balanced
                'celltype_specific')
            print_every: Print loss every N epochs (default: 50)
        """
        print(f"\n{'='*60}")
        print(f"Stage 1: VAE Training")
        print(f"{'='*60}")
        print(f"  Epochs:        {n_epochs}")
        print(f"  LR:            {lr}")
        print(f"  Batch Size:    {batch_size}")
        print(f"  Latent Dim:    {latent_dim}")
        print(f"  Beta (KL):     {beta}")
        print(f"  Lambda MMD:    {lambda_mmd}")
        print(f"  Resolution:    {resolution}")
        print(f"  Top N/Type:    {top_n_per_type}")
        print(f"  Dual Decoder:  {use_dual_decoder}")
        print(f"  Seed:          {self.seed}")
        print(f"  Save to Disk:  {self.save_to_disk}")
        print(f"{'='*60}\n")
        
        # 1. Load data
        sc_adata, st_adata = self.load_data()
        
        # 2. Prepare data based on marker genes (with test split for early stopping)
        (
            train_X,
            test_X,
            train_modality,
            test_modality,
            train_labels,
            test_labels,
            y_train,
            y_test,
            sc_X_final,
            sc_all_genes_raw,
            sc_all_labels,
            sc_celltype_labels,
        ) = self.prepare_marker_gene_data(
            sc_adata, st_adata, top_n_per_type=top_n_per_type, resolution=resolution, 
            precomputed_marker_file=precomputed_marker_file, marker_selection_method=marker_selection_method
        )
        
        # 3. Build or load VAE
        input_dim = len(self.genes)
        
        if pretrained_path and os.path.exists(pretrained_path):
            # Load pretrained weights
            pretrained_output_type, pretrained_latent_dim = self.load_pretrained(pretrained_path)
            
            # Verify architecture compatibility
            if input_dim != self.vae.encoder.encoder[0].in_features:
                self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, loss_type=loss_type, use_dual_decoder=use_dual_decoder)
        else:
            # Build from scratch
            self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, loss_type=loss_type, use_dual_decoder=use_dual_decoder)
        
        # 4. Train VAE with test set for early stopping
        # Only pass output_dir when save_to_disk=True (for saving training curves)
        best_loss = train_vae(
            vae=self.vae,
            train_X=train_X,
            test_X=test_X,  # Use test set for validation and early stopping
            train_modality=train_modality,
            test_modality=test_modality,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            beta=beta,
            loss_type=loss_type,
            lambda_mmd=lambda_mmd,
            device=self.device,
            output_dir=self.output_dir if self.save_to_disk else None,
            print_every=print_every,
            patience=20,
            min_delta=1,
            train_labels=train_labels,
            test_labels=test_labels,
        )
        
        # Save training data for cluster center computation
        self.train_X = train_X
        self.train_modality = train_modality  
        self.y_train = y_train
        # Save ALL SC data (train + test) for cluster embedding computation
        self.sc_X_final = sc_X_final  # ALL SC marker gene data (log1p normalized)
        self.sc_all_genes_raw = sc_all_genes_raw  # ALL SC raw counts for all genes [n_cells, n_all_genes]
        self.sc_all_labels = sc_all_labels  # ALL SC cluster labels
        self.sc_celltype_labels = sc_celltype_labels
        
        # Use trained VAE to compute embeddings for ALL SC cells
        self.vae.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size_embed = 1000
            all_embeddings = []
            
            for i in range(0, len(sc_X_final), batch_size_embed):
                batch_data = sc_X_final[i:i+batch_size_embed]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                
                # Get latent representation
                mu, log_var = self.vae.encoder(batch_tensor)
                all_embeddings.append(mu.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
        
        # Compute cluster centers and expressions using ALL SC data
        # Note: Static mode still requires cluster aggregated expression, so pass raw counts for all genes
        cluster_prototypes, cluster_expressions, cluster_expressions_full_count, cluster_cell_weights = \
            compute_cluster_centers_and_expressions(
                embeddings=embeddings,
                sc_train_data=sc_X_final,  # Use ALL marker gene data (log1p)
                sc_train_labels=sc_all_labels,  # Use ALL labels
                sc_X_full_train_count=sc_all_genes_raw,  # Pass raw counts for all genes (for static mode cluster aggregation)
                aggregation_method=aggregation_method
            )
        
        # Save cluster centers and expressions
        self.cluster_prototypes = cluster_prototypes
        self.cluster_expressions = cluster_expressions
        self.cluster_expressions_full = cluster_expressions_full_count
        self.cluster_expressions_full_count = cluster_expressions_full_count
        self.cluster_cell_weights = cluster_cell_weights
        
        # Step 5.5: Save SC embeddings and raw expression (for dynamic cluster usage)
        # This data will be passed to Stage2 for pre-computing k-nearest cells or in-training use
        self.sc_cell_embeddings = embeddings  # [n_sc_cells, latent_dim]
        self.sc_cell_expressions_raw = sc_all_genes_raw  # [n_sc_cells, n_all_genes] raw counts for all genes
        self.sc_cell_labels = sc_all_labels  # [n_sc_cells] cluster labels
        
        # 6. Extract celltype-cluster mapping (if celltype available in sc_adata)
        cluster_to_celltype = {}
        
        # Check for celltype column (prioritize 'cell_type', then 'celltype')
        celltype_col = self._resolve_celltype_key(self.sc_adata_clustered)
        
        if celltype_col is not None:
            for cluster_id in sorted(self.sc_adata_clustered.obs['leiden'].unique()):
                cluster_mask = self.sc_adata_clustered.obs['leiden'] == cluster_id
                celltype_counts = self.sc_adata_clustered.obs[cluster_mask][celltype_col].value_counts()
                major_celltype = celltype_counts.index[0]
                total_cells = celltype_counts.sum()
                cluster_to_celltype[str(cluster_id)] = major_celltype
                #print(f"   Cluster {cluster_id} -> {major_celltype} ({celltype_counts.iloc[0]}/{total_cells} cells)")
            self.cluster_to_celltype = cluster_to_celltype
        else:
            self.cluster_to_celltype = None
        
        # 7. Plot UMAP for modality alignment visualization using vae_viz module
        if self.save_to_disk and self.output_dir:
            plot_modality_alignment_umap(
                vae=self.vae,
                train_X=train_X,
                train_modality=train_modality,
                y_train=y_train,
                device=self.device,
                output_dir=self.output_dir
            )
        
        # Step 8: Don't save model files (pure memory mode, passed through Stage1Artifacts)
        model_path = None
        npz_path = None
        
        # Stage 1 no longer saves .pth and .npz, all data passed through memory
        
        # Convert cluster_prototypes/expressions to numpy arrays (if dict)
        if isinstance(self.cluster_prototypes, dict):
            cluster_ids_sorted = sorted(self.cluster_prototypes.keys())
            prototypes_array = np.stack([self.cluster_prototypes[cid] for cid in cluster_ids_sorted], axis=0)
        else:
            prototypes_array = self.cluster_prototypes
        
        if isinstance(self.cluster_expressions, dict):
            cluster_ids_sorted = sorted(self.cluster_expressions.keys())
            expressions_array = np.stack([self.cluster_expressions[cid] for cid in cluster_ids_sorted], axis=0)
        else:
            expressions_array = self.cluster_expressions
        
        if isinstance(self.cluster_expressions_full_count, dict):
            cluster_ids_sorted = sorted(self.cluster_expressions_full_count.keys())
            expressions_full_list = [self.cluster_expressions_full_count[cid] for cid in cluster_ids_sorted]
        else:
            expressions_full_list = self.cluster_expressions_full_count
        
        # Return results: contains all in-memory artifacts for Stage1Artifacts usage
        return {
            'best_loss': best_loss,
            'n_genes': len(self.genes),
            'n_clusters': len(self.label_encoder.classes_),
            'model_path': model_path,
            'cluster_data_path': npz_path,
            'clusters': list(self.label_encoder.classes_),
            # ===== Memory artifacts (for pure memory mode) =====
            'vae_encoder': self.vae.encoder,  # Directly return encoder module
            'vae_state_dict': self.vae.state_dict(),  # Also return state_dict as backup
            'input_dim': len(self.genes),
            'latent_dim': self.vae.encoder.fc_mu.out_features,
            'output_type': getattr(self.vae, 'output_type', 'mse'),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'sc_clusters': self.sc_clusters,
            'resolution': self.resolution,
            'all_genes': getattr(self, 'all_genes', None),
            'cluster_to_celltype': self.cluster_to_celltype,
            'celltype_prototypes': prototypes_array,
            'celltype_expressions': expressions_array,
            'celltype_expressions_full': expressions_full_list,
            'hvg_genes_union': getattr(self, 'hvg_genes_union', None),
            'auto_library_size': getattr(self, 'auto_library_size', 1.0),  # Auto-computed library_size
            # ===== Data required for dynamic cluster =====
            'sc_cell_embeddings': self.sc_cell_embeddings,  # [n_sc_cells, latent_dim]
            'sc_cell_expressions_raw': self.sc_cell_expressions_raw,  # [n_sc_cells, n_all_genes] raw counts for all genes
            'sc_cell_labels': self.sc_cell_labels,  # [n_sc_cells] cluster labels
            'sc_cell_marker_expressions': self.sc_X_final,
            'sc_celltype_labels': self.sc_celltype_labels,
            'celltype_key': getattr(self, 'celltype_key', None),
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stage 1: VAE Training for SC-ST Integration')
    
    # Data arguments
    parser.add_argument('--sc_file', type=str, required=True,
                       help='Path to single-cell h5ad file')
    parser.add_argument('--st_file', type=str, required=True,
                       help='Path to spatial transcriptomics h5ad file')
    parser.add_argument('--output_dir', type=str, default="./stage1_results",
                       help='Output directory path')
    parser.add_argument('--celltype_key', type=str, default=None,
                       help='Optional scRNA.obs column with reference cell-type annotations')
    
    # Clustering arguments
    parser.add_argument('--resolution', type=float, default=4,
                       help='Leiden clustering resolution')
    
    # Model arguments
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='Marker genes per cluster/celltype')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='VAE hidden layer dimensions')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='VAE latent space dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=150,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='KL divergence weight (beta-VAE)')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'zinb'],
                       help='Reconstruction loss type: mse (default) or zinb')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                       help='MMD loss weight for modality alignment (0=disabled, 1.0=recommended)')
    parser.add_argument('--use_dual_decoder', type=bool, default=True,
                       help='Use DualDecoderVAE with separate SC/ST decoders for better modality alignment')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained VAE model to continue training')
    parser.add_argument('--precomputed_marker_file', type=str, default=None,
                       help='Path to precomputed marker genes file. If provided, '
                            'marker genes will be loaded directly from this file instead of computing them.')
    parser.add_argument('--aggregation_method', type=str, default='mean', 
                       choices=['mean', 'median', 'weighted'],
                       help='Cluster aggregation method: mean (simple average), median (robust to outliers), '
                            'weighted (UMI+representativeness+marker activity, recommended)')
    parser.add_argument('--marker_selection_method', type=str, default='variance', 
                       choices=['l1', 'variance', 'correlation', 'celltype_specific'],
                       help='Method for marker gene selection: l1 (L1-regularized logistic regression), '
                            'variance (variance threshold), correlation (correlation-based filtering), '
                            'or celltype_specific (balanced annotation markers)')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    

    # Create VAE encoder
    co_encoder = coEncoder(
        sc_file=args.sc_file,
        st_file=args.st_file,
        output_dir=args.output_dir,
        celltype_key=args.celltype_key,
        device=args.device
    )
    
    # Run stage 1 VAE training
    results = co_encoder.run_stage1_training(
        top_n_per_type=args.top_n_per_type,
        resolution=args.resolution,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        beta=args.beta,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        loss_type=args.loss_type,
        lambda_mmd=args.lambda_mmd,
        pretrained_path=args.pretrained_path,
        precomputed_marker_file=args.precomputed_marker_file,
        use_dual_decoder=args.use_dual_decoder,
        aggregation_method=args.aggregation_method,
        marker_selection_method=args.marker_selection_method
    )
    
if __name__ == "__main__":
    main()
