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
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import math
import argparse
import warnings
from tqdm import tqdm
import umap
warnings.filterwarnings('ignore')

# Import unified model definitions
from model import VAE, vae_loss_function, zinb_loss_function, compute_mmd

def load_marker_genes_from_file(file_path):
    """
    Load marker genes from a text file
    
    Args:
        file_path: Path to the text file containing marker genes (one gene per line)
    
    Returns:
        marker_genes: List of marker genes
    """
    print("="*60)
    print(f"Loading marker genes from file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Marker genes file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        marker_genes = [line.strip() for line in f if line.strip()]
    
    print(f"   Loaded {len(marker_genes)} marker genes")
    return marker_genes
    """
    Compute clusters and extract top marker genes for each cluster
    """
    print("="*60)
    print("Starting clustering analysis...")
    
    # Backup original data
    adata_backup = adata.copy()
    
    # Preprocessing: normalization and PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Build neighbor graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    print(f"Clustering results: {len(adata.obs['leiden'].unique())} clusters")
    
    # Restore to original gene set for marker analysis
    adata_full = adata_backup.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    # sc.pp.log1p(adata_full)
    
    # Transfer clustering results to full dataset
    adata_full.obs['leiden'] = adata.obs['leiden'].copy()
    
    # Compute marker genes for each cluster
    sc.tl.rank_genes_groups(
        adata_full, 
        'leiden', 
        method='wilcoxon',
        key_added='rank_genes_groups',
        n_genes=top_n * 2
    )
    
    # Extract marker genes
    marker_genes = set()
    result = adata_full.uns['rank_genes_groups']
    
    print(f"Marker genes per cluster:")
    lasso_selected = {}
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
            
            # Apply Lasso regression for further selection
            if len(selected_genes) > 0:
                sub_adata = adata_full[:, selected_genes].copy()
                y = (adata_full.obs['leiden'] == cluster).astype(int)
                X = sub_adata.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                
                scaler = StandardScaler(with_mean=False)
                X_scaled = scaler.fit_transform(X)
                
                clf = LogisticRegressionCV(
                    Cs=10,
                    penalty='l1',
                    solver='saga',
                    class_weight='balanced',
                    cv=5,
                    max_iter=3000,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                clf.fit(X_scaled, y)
                coef = clf.coef_.ravel()
                
                lasso_selected_genes = [g for g, c in zip(selected_genes, coef) if abs(c) > 1e-5]
                lasso_selected_genes = sorted(lasso_selected_genes, key=lambda g: abs(coef[selected_genes.index(g)]), reverse=True)
                
                lasso_selected[cluster] = lasso_selected_genes
                marker_genes.update(lasso_selected_genes)
                print(f"   {cluster}: {len(selected_genes)} -> {len(lasso_selected_genes)} (after Lasso)")
            else:
                lasso_selected[cluster] = []
                print(f"   {cluster}: 0 genes")
    
    print(f"Total: {len(marker_genes)} marker genes")
    
    # Return clustering info, marker genes, and full adata for annotation
    return sorted(list(marker_genes)), adata_full.obs['leiden'].copy(), adata_full

def extract_marker_genes_from_celltype(adata, celltype_col='cell_type', top_n=100, min_fold_change=1.5, save_path=None):
    """
    Extract marker genes using existing celltype annotation (no clustering)
    
    Args:
        adata: AnnData object with celltype annotation
        celltype_col: Column name in adata.obs containing celltype labels
        top_n: Number of top marker genes per celltype
        min_fold_change: Minimum log2 fold change threshold
        save_path: Path to save marker genes
    
    Returns:
        marker_genes: List of marker genes
        celltype_labels: Series with celltype labels
        adata: Annotated data
    """
    print("="*60)
    print(f"Extracting marker genes using existing celltype annotation...")
    print(f"   Celltype column: {celltype_col}")
    
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs! Available columns: {list(adata.obs.columns)}")
    
    # Normalize
    adata_work = adata.copy()
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    
    # Compute marker genes for each celltype
    sc.tl.rank_genes_groups(
        adata_work, 
        celltype_col, 
        method='wilcoxon',
        key_added='rank_genes_groups',
        n_genes=top_n * 2
    )
    
    # Extract marker genes
    marker_genes = set()
    result = adata_work.uns['rank_genes_groups']
    
    print(f"Marker genes per celltype:")
    lasso_selected = {}
    for celltype in sorted(adata_work.obs[celltype_col].unique()):
        if celltype in result['names'].dtype.names:
            genes = result['names'][celltype]
            scores = result['scores'][celltype]
            pvals = result['pvals_adj'][celltype]
            logfoldchanges = result['logfoldchanges'][celltype]
            
            selected_genes = []
            for i in range(len(genes)):
                if (pvals[i] < 0.05 and 
                    scores[i] > 0 and 
                    logfoldchanges[i] >= np.log2(min_fold_change)):
                    selected_genes.append(genes[i])
                    
                if len(selected_genes) >= top_n:
                    break
            
            # Apply Lasso regression for further selection
            if len(selected_genes) > 0:
                sub_adata = adata_work[:, selected_genes].copy()
                y = (adata_work.obs[celltype_col] == celltype).astype(int)
                X = sub_adata.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                
                scaler = StandardScaler(with_mean=False)
                X_scaled = scaler.fit_transform(X)
                
                clf = LogisticRegressionCV(
                    Cs=10,
                    penalty='l1',
                    solver='saga',
                    class_weight='balanced',
                    cv=5,
                    max_iter=3000,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                clf.fit(X_scaled, y)
                coef = clf.coef_.ravel()
                
                lasso_selected_genes = [g for g, c in zip(selected_genes, coef) if abs(c) > 1e-5]
                lasso_selected_genes = sorted(lasso_selected_genes, key=lambda g: abs(coef[selected_genes.index(g)]), reverse=True)
                
                lasso_selected[celltype] = lasso_selected_genes
                marker_genes.update(lasso_selected_genes)
                print(f"   {celltype}: {len(selected_genes)} -> {len(lasso_selected_genes)} (after Lasso)")
            else:
                lasso_selected[celltype] = []
                print(f"   {celltype}: 0 genes")
    
    print(f"Total: {len(marker_genes)} marker genes")
    
    if save_path:
        with open(save_path, 'w') as f:
            for gene in sorted(marker_genes):
                f.write(f"{gene}\n")
    
    return sorted(list(marker_genes)), adata.obs[celltype_col].copy(), adata

#============================================================
# Main Module
#============================================================
class coEncoder:
    def __init__(self, 
                 sc_file=None,
                 st_file=None,
                 output_dir="./stage1_results",
                 device=None):
        """
        Initialize co-encoder
        
        Args:
            sc_file: Path to single-cell h5ad file
            st_file: Path to spatial transcriptomics h5ad file
            output_dir: Output directory path
            device: Computing device (cuda/cpu, None for auto)
        """
        self.sc_file = sc_file
        self.st_file = st_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")

        # Model components
        self.vae = None
        self.label_encoder = None
        self.marker_genes = None
        self.celltype_key = None  # Track if using celltype annotation
        
    def load_data(self) -> Tuple[ad.AnnData, ad.AnnData]:
        """Load SC and ST data from specified files"""
        print("="*60)
        print("Loading datasets...")
        
        if self.sc_file is None or self.st_file is None:
            raise ValueError("SC file and ST file must be specified!")
        
        if not os.path.exists(self.sc_file):
            raise FileNotFoundError(f"SC file not found: {self.sc_file}")
        
        if not os.path.exists(self.st_file):
            raise FileNotFoundError(f"ST file not found: {self.st_file}")
        
        # Load SC data
        print(f"   Loading SC: {self.sc_file}")
        sc_adata = sc.read_h5ad(self.sc_file)
        sc_adata.obs['modality'] = 'SC'
        print(f"   SC shape: {sc_adata.shape}")
        
        # Load ST data
        print(f"   Loading ST: {self.st_file}")
        st_adata = sc.read_h5ad(self.st_file)
        st_adata.obs['modality'] = 'ST'
        print(f"   ST shape: {st_adata.shape}")
        
        # Find common genes
        common_genes = list(set(sc_adata.var_names) & set(st_adata.var_names))
        common_genes.sort()
        print(f"   Common genes: {len(common_genes)}")
        
        # Subset to common genes
        sc_adata = sc_adata[:, common_genes].copy()
        st_adata = st_adata[:, common_genes].copy()
        
        print(f"   SC final: {sc_adata.shape}")
        print(f"   ST final: {st_adata.shape}")
        
        return sc_adata, st_adata
        return sc_adata, st_adata

    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100, resolution: float = 0.5,
                               celltype_key: str = None, precomputed_marker_file: str = None) -> Tuple:
        """Prepare training data based on marker genes
        
        Args:
            sc_adata: Single cell AnnData object
            st_adata: Spatial transcriptomics AnnData object
            top_n_per_type: Number of marker genes per cluster/celltype
            resolution: Leiden resolution (for auto-clustering mode)
            celltype_key: Column name in sc_adata.obs for celltype annotation.
                         If None, will auto-cluster using Leiden. If provided,
                         will use existing celltype annotation from adata.obs[celltype_key]
            precomputed_marker_file: Path to precomputed marker genes file.
                                   If provided, will load marker genes directly from this file
                                   instead of computing them from scratch.
        """

        # 1. Compute clusters and marker genes  
        print("="*60)
        if precomputed_marker_file is not None:
            # Load precomputed marker genes
            print(f"Using precomputed marker genes from: {precomputed_marker_file}")
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
            self.celltype_key = None  # Not using celltype annotation
        elif celltype_key is not None:
            # Use existing celltype annotation
            print(f"Using existing celltype annotation from '{celltype_key}'...")
            cluster_save_path = f"{self.output_dir}/marker_genes_celltype.txt"
            self.marker_genes, sc_clusters, sc_adata_clustered = extract_marker_genes_from_celltype(
                sc_adata.copy(),
                celltype_col=celltype_key,
                top_n=top_n_per_type,
                save_path=cluster_save_path
            )
            # Rename the celltype column to 'leiden' for compatibility with downstream code
            sc_adata_clustered.obs['leiden'] = sc_clusters.copy()
            self.celltype_key = celltype_key  # Save celltype_key for checkpoint
        else:
            # Auto-cluster using Leiden
            print("Computing clusters and marker genes...")
            cluster_save_path = f"{self.output_dir}/marker_genes.txt"
            self.marker_genes, sc_clusters, sc_adata_clustered = compute_clusters_and_marker_genes(
                sc_adata.copy(), 
                top_n=top_n_per_type, 
                resolution=resolution,
                save_path=cluster_save_path
            )
            self.celltype_key = None  # Not using celltype annotation
        
        # Save clustered adata for annotation
        self.sc_adata_clustered = sc_adata_clustered
        cluster_adata_file = f"{self.output_dir}/sc_adata_clustered.h5ad"
        sc_adata_clustered.write_h5ad(cluster_adata_file)
        print(f"Saved clustered SC adata: {cluster_adata_file}")
        
        # Save clustering info and resolution
        self.sc_clusters = sc_clusters
        self.resolution = resolution
        
        # 2. Process SC data (extract marker genes then normalize)
        print("Processing SC data...")
                
        # SC normalization - use count space (no log1p) for VAE training
        sc_adata_count = sc_adata.copy()
        sc.pp.normalize_total(sc_adata_count, target_sum=1e4)
        # NO log1p - use count directly
        # sc.pp.log1p(sc_adata_count)

        # Extract full gene expression (count version)
        sc_X_full_count = sc_adata_count.X.toarray() if hasattr(sc_adata_count.X, 'toarray') else sc_adata_count.X
        sc_all_genes = list(sc_adata_count.var.index)
        
        # Extract marker genes (count for training)
        sc_subset = sc_adata_count[:, sc_adata_count.var.index.isin(self.marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_clusters.values
        print("="*60)
        print(f"SC data (count) min: {np.min(sc_X)}, max: {np.max(sc_X)}")
        print("="*60)
        # Encode labels
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"   SC data: {sc_X.shape}")
        print(f"   Number of clusters: {len(self.label_encoder.classes_)}")

        # ST
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        # NO log1p - use count directly
        # sc.pp.log1p(st_adata)
        available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
        st_subset = st_adata[:, available_genes].copy()
        
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        print("="*60)
        print(f"ST data (count) min: {np.min(st_X)}, max: {np.max(st_X)}")
        print("="*60)
        print(f"   ST data: {st_X.shape}, available genes: {len(available_genes)}/{len(self.marker_genes)}")
        
        # 4. Ensure SC and ST feature dimensions are consistent
        final_genes = [g for g in self.marker_genes 
                      if g in sc_subset.var.index and g in st_subset.var.index]
        
        sc_gene_indices = [list(sc_subset.var.index).index(g) for g in final_genes]
        st_gene_indices = [list(st_subset.var.index).index(g) for g in final_genes]
        
        sc_X_final = sc_X[:, sc_gene_indices]
        st_X_final = st_X[:, st_gene_indices]
        
        print(f"   Final gene count: {len(final_genes)}")
        
        # 5. Split data
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.1, stratify=sc_y, random_state=42
        )
        
        # Split full gene SC data with same indices
        sc_train_indices = np.arange(len(sc_X_final))
        sc_train_idx, sc_test_idx = train_test_split(
            sc_train_indices, test_size=0.1, stratify=sc_y, random_state=42
        )
        sc_X_full_train_count = sc_X_full_count[sc_train_idx]
        
        st_train, st_test = train_test_split(
            st_X_final, test_size=0.1, random_state=42
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
        
        print(f"   Train set: {train_X.shape} (SC: {len(sc_train)}, ST: {len(st_train)})")
        print(f"   Test set: {test_X.shape} (SC: {len(sc_test)}, ST: {len(st_test)})")
        
        # Save gene list
        self.genes = final_genes
        self.all_genes = sc_all_genes  # Save all gene list
        genes_file = f"{self.output_dir}/final_genes.txt"
        with open(genes_file, 'w') as f:
            for gene in self.genes:
                f.write(f"{gene}\n")

        return train_X, test_X, train_modality, test_modality, y_train, y_test, sc_X_full_train_count
    
    def build_vae(self, input_dim: int, hidden_dims=[512, 256], latent_dim=128, dropout=0.2, loss_type='mse'):
        """Build VAE model"""
        print("="*60)
        print("Building VAE model...")
        print(f"   Loss type: {loss_type.upper()}")
        
        # 根据loss_type设置output_type
        output_type = 'zinb' if loss_type == 'zinb' else 'mse'
        
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            output_type=output_type
        ).to(self.device)
        
        print(f"   Input: {input_dim} -> Latent: {latent_dim}")
        print(f"   Hidden layers: {hidden_dims}")
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"   Parameters: {vae_params:,}")
    
    def train_vae(self, train_X, test_X, train_modality, test_modality,
                  batch_size=256, n_epochs=100, lr=1e-3, beta=1.0, loss_type='mse', lambda_mmd=1.0):
        """Train VAE with optional MMD loss for modality alignment"""

        print("="*60)
        print("Starting VAE training...")
        print(f"   Train data: {train_X.shape} (SC: {sum(train_modality==0)}, ST: {sum(train_modality==1)})")
        print(f"   Test data: {test_X.shape} (SC: {sum(test_modality==0)}, ST: {sum(test_modality==1)})")
        print(f"   Loss type: {loss_type.upper()}")
        print(f"   MMD weight: {lambda_mmd}")

        class SimpleDataset(Dataset):
            def __init__(self, X, modality):
                self.X = torch.FloatTensor(X)
                self.modality = torch.LongTensor(modality)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.modality[idx]

        # Data loader
        train_dataset = SimpleDataset(train_X, train_modality)
        test_dataset = SimpleDataset(test_X, test_modality)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
        
        # Training history
        train_losses = []
        test_losses = []
        recon_losses = []
        kl_losses = []
        mmd_losses = []  # Track MMD loss
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        pbar = tqdm(range(n_epochs), desc="VAE Training", unit="epoch")
        for epoch in pbar:
            # Training
            self.vae.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_mmd = 0.0
            
            for batch_data, batch_modality in train_loader:
                batch_data = batch_data.to(self.device)
                batch_modality = batch_modality.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE forward pass
                if loss_type == 'zinb':
                    mean, disp, pi, mu, log_var, z = self.vae(batch_data)
                    total_loss, recon_loss, kl_div = zinb_loss_function(
                        mean, disp, pi, batch_data, mu, log_var, beta=beta
                    )
                else:
                    recon_data, mu, log_var, z = self.vae(batch_data)
                    total_loss, recon_loss, kl_div = vae_loss_function(
                        recon_data, batch_data, mu, log_var, beta=beta
                    )
                
                # Compute MMD loss for modality alignment
                mmd_loss = torch.tensor(0.0, device=self.device)
                if lambda_mmd > 0:
                    # Separate SC and ST embeddings in this batch
                    sc_mask = batch_modality == 0
                    st_mask = batch_modality == 1
                    
                    # Only compute MMD if both modalities present in batch
                    if sc_mask.sum() > 0 and st_mask.sum() > 0:
                        sc_embeddings = z[sc_mask]
                        st_embeddings = z[st_mask]
                        mmd_loss = compute_mmd(sc_embeddings, st_embeddings, kernel='rbf')
                
                # Total loss with MMD
                total_loss = total_loss + lambda_mmd * mmd_loss
                
                # Normalize loss
                total_loss = total_loss / len(batch_data)
                recon_loss = recon_loss / len(batch_data)
                kl_div = kl_div / len(batch_data)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_div.item()
                epoch_mmd += mmd_loss.item() if lambda_mmd > 0 else 0.0
            
            avg_loss = epoch_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            avg_mmd = epoch_mmd / len(train_loader)
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            mmd_losses.append(avg_mmd)
            
            # Evaluate
            if (epoch + 1) % 5 == 0:
                test_loss = self.evaluate_vae(test_loader, beta, loss_type)
                test_losses.append(test_loss)
                
                scheduler.step(test_loss)
                
                # Update progress bar with MMD info
                if lambda_mmd > 0:
                    pbar.set_postfix({'Train': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 
                                     'KL': f'{avg_kl:.4f}', 'MMD': f'{avg_mmd:.4f}', 'Test': f'{test_loss:.4f}'})
                else:
                    pbar.set_postfix({'Train': f'{avg_loss:.4f}', 'Recon': f'{avg_recon:.4f}', 
                                     'KL': f'{avg_kl:.4f}', 'Test': f'{test_loss:.4f}'})
                
                # Save best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    # Will save model after computing cluster centers
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= patience:
                    pbar.close()
                    break
        
        # Plot training curves
        self.plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses, mmd_losses)
        
        return best_loss
    
    def evaluate_vae(self, test_loader, beta=1.0, loss_type='mse'):
        """Evaluate VAE"""
        self.vae.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)
                
                if loss_type == 'zinb':
                    mean, disp, pi, mu, log_var, z = self.vae(batch_data)
                    loss, _, _ = zinb_loss_function(mean, disp, pi, batch_data, mu, log_var, beta)
                else:
                    recon_data, mu, log_var, z = self.vae(batch_data)
                    loss, _, _ = vae_loss_function(recon_data, batch_data, mu, log_var, beta)
                    
                total_loss += loss.item() / len(batch_data)
        
        return total_loss / len(test_loader)
    
    def plot_vae_training_curves(self, train_losses, test_losses, recon_losses, kl_losses, mmd_losses=None):
        """Plot VAE training curves"""
        # Determine if we need to plot MMD
        has_mmd = mmd_losses is not None and len(mmd_losses) > 0 and max(mmd_losses) > 0
        
        if has_mmd:
            fig, axes = plt.subplots(2, 3, figsize=(22, 10))
            ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
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
        
        # Reconstruction loss
        ax2.plot(recon_losses, 'g-')
        ax2.set_title('Reconstruction Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # KL divergence
        ax3.plot(kl_losses, 'r-')
        ax3.set_title('KL Divergence')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('KL Div')
        ax3.grid(True)
        
        # Loss components comparison
        ax4.plot(recon_losses, label='Reconstruction', color='green')
        ax4.plot(kl_losses, label='KL Divergence', color='red')
        if has_mmd:
            ax4.plot(mmd_losses, label='MMD', color='purple')
        ax4.set_title('Loss Components')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        if has_mmd:
            # MMD loss
            ax5.plot(mmd_losses, 'purple')
            ax5.set_title('MMD Loss (Modality Alignment)')
            ax5.set_xlabel('Epochs')
            ax5.set_ylabel('MMD')
            ax5.grid(True)
            
            # All components normalized
            ax6.plot(np.array(recon_losses) / (max(recon_losses) + 1e-8), label='Recon (norm)', color='green')
            ax6.plot(np.array(kl_losses) / (max(kl_losses) + 1e-8), label='KL (norm)', color='red')
            ax6.plot(np.array(mmd_losses) / (max(mmd_losses) + 1e-8), label='MMD (norm)', color='purple')
            ax6.set_title('Normalized Loss Components')
            ax6.set_xlabel('Epochs')
            ax6.set_ylabel('Normalized Loss')
            ax6.legend()
            ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/vae_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_modality_alignment_umap(self, train_X, train_modality, y_train=None):
        """
        Plot UMAP visualization of SC and ST modality alignment
        
        Args:
            train_X: Training data (combined SC + ST)
            train_modality: Modality labels (0=SC, 1=ST)
            y_train: Optional cluster labels for SC samples
        """
        print("="*60)
        print("Generating UMAP visualization for modality alignment...")
        
        # Get embeddings from trained VAE
        self.vae.eval()
        with torch.no_grad():
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(train_X), batch_size):
                batch_data = train_X[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                mu, log_var = self.vae.encoder(batch_tensor)
                all_embeddings.append(mu.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
        
        print(f"   Computing UMAP on {embeddings.shape[0]} samples with {embeddings.shape[1]} dims...")
        
        # Compute UMAP
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
        umap_coords = reducer.fit_transform(embeddings)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Color by modality (SC vs ST)
        ax1 = axes[0]
        sc_mask = train_modality == 0
        st_mask = train_modality == 1
        
        ax1.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
                   c='#1f77b4', s=20, alpha=0.6, label=f'SC (n={sum(sc_mask)})', edgecolors='none')
        ax1.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
                   c='#ff7f0e', s=20, alpha=0.6, label=f'ST (n={sum(st_mask)})', edgecolors='none')
        
        ax1.set_title('UMAP: SC vs ST Modality Alignment', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP 1', fontsize=12)
        ax1.set_ylabel('UMAP 2', fontsize=12)
        ax1.legend(fontsize=11, markerscale=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Color by cluster (SC only) + ST
        ax2 = axes[1]
        
        if y_train is not None:
            # Get SC data with cluster labels
            sc_clusters = y_train
            n_clusters = len(np.unique(sc_clusters))
            
            # Use a colormap for clusters
            cmap = plt.cm.get_cmap('tab20', n_clusters)
            
            # Plot each cluster
            for cluster_id in np.unique(sc_clusters):
                cluster_mask_in_sc = sc_clusters == cluster_id
                # Convert to global index (all train_X)
                sc_indices = np.where(sc_mask)[0]
                cluster_global_mask = np.zeros(len(train_X), dtype=bool)
                cluster_global_mask[sc_indices[cluster_mask_in_sc]] = True
                
                ax2.scatter(umap_coords[cluster_global_mask, 0], 
                           umap_coords[cluster_global_mask, 1],
                           c=[cmap(cluster_id)], s=20, alpha=0.6, 
                           label=f'Cluster {cluster_id}', edgecolors='none')
            
            # Plot ST in gray
            ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
                       c='lightgray', s=20, alpha=0.4, label=f'ST (n={sum(st_mask)})', edgecolors='none')
            
            ax2.set_title(f'UMAP: SC Clusters (n={n_clusters}) + ST', fontsize=14, fontweight='bold')
        else:
            # If no cluster labels, just plot SC and ST
            ax2.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], 
                       c='#1f77b4', s=20, alpha=0.6, label=f'SC', edgecolors='none')
            ax2.scatter(umap_coords[st_mask, 0], umap_coords[st_mask, 1], 
                       c='#ff7f0e', s=20, alpha=0.6, label=f'ST', edgecolors='none')
            ax2.set_title('UMAP: SC + ST', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('UMAP 1', fontsize=12)
        ax2.set_ylabel('UMAP 2', fontsize=12)
        ax2.legend(fontsize=9, markerscale=2, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/modality_alignment_umap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   UMAP visualization saved to: {self.output_dir}/modality_alignment_umap.png")
    
    def save_vae(self, filepath):
        """Save VAE model"""
        # Check if cluster info exists
        cluster_prototypes = getattr(self, 'cluster_prototypes', None)
        cluster_expressions = getattr(self, 'cluster_expressions', None)
        cluster_expressions_full = getattr(self, 'cluster_expressions_full', None)
        cluster_expressions_full_count = getattr(self, 'cluster_expressions_full_count', None)
        
        print("="*60)
        print(f"Saving model to: {filepath}")
        if cluster_prototypes is not None:
            print(f"   Cluster centers: {len(cluster_prototypes)} clusters")
        else:
            print(f"   Warning: cluster centers missing")
            
        if cluster_expressions is not None:
            print(f"   Cluster expressions (marker genes, count): {len(cluster_expressions)} clusters")
        else:
            print(f"   Warning: cluster expressions missing")
        
        if cluster_expressions_full is not None:
            print(f"   Cluster expressions (all genes, count): {len(cluster_expressions_full)} clusters")
        else:
            print(f"   Warning: full gene expressions missing")
        
        if cluster_expressions_full_count is not None:
            print(f"   Cluster expressions (all genes, count backup): {len(cluster_expressions_full_count)} clusters")
        else:
            print(f"   Warning: full gene expressions (count backup) missing")
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'input_dim': len(self.genes),
            'latent_dim': self.vae.latent_dim,
            'output_type': self.vae.output_type,
            'sc_clusters': getattr(self, 'sc_clusters', None),
            'resolution': getattr(self, 'resolution', 0.5),
            'celltype_key': getattr(self, 'celltype_key', None),
            'cluster_prototypes': cluster_prototypes,
            'cluster_expressions': cluster_expressions,
            'cluster_expressions_full': cluster_expressions_full,
            'cluster_expressions_full_count': cluster_expressions_full_count,
            'all_genes': getattr(self, 'all_genes', None)
        }, filepath)
    
    def load_vae(self, filepath):
        """Load VAE model (basic loading for inference)"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        # Detect output_type from checkpoint
        output_type = checkpoint.get('output_type', 'mse')
        
        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        
        print(f"VAE model loaded: {filepath}")
    
    def load_pretrained(self, filepath):
        """Load pretrained VAE weights for continued training"""
        print("="*60)
        print(f"Loading pretrained weights from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pretrained model not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model architecture info
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        output_type = checkpoint.get('output_type', 'mse')
        
        print(f"   Input dim: {input_dim}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Output type: {output_type}")
        
        # Build VAE model with same architecture
        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_type=output_type).to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        # Load other components
        self.label_encoder = checkpoint.get('label_encoder', None)
        self.marker_genes = checkpoint.get('marker_genes', None)
        self.genes = checkpoint.get('genes', None)
        self.all_genes = checkpoint.get('all_genes', None)
        self.sc_clusters = checkpoint.get('sc_clusters', None)
        self.resolution = checkpoint.get('resolution', 0.5)
        self.celltype_key = checkpoint.get('celltype_key', None)
        
        # Load cluster info if available
        self.cluster_prototypes = checkpoint.get('cluster_prototypes', None)
        self.cluster_expressions = checkpoint.get('cluster_expressions', None)
        self.cluster_expressions_full = checkpoint.get('cluster_expressions_full', None)
        self.cluster_expressions_full_count = checkpoint.get('cluster_expressions_full_count', None)
        
        if self.cluster_prototypes is not None:
            print(f"   Loaded {len(self.cluster_prototypes)} cluster prototypes")
        
        if self.celltype_key is not None:
            print(f"   Using celltype annotation: {self.celltype_key}")
        
        print("   Pretrained weights loaded successfully!")
        print("="*60)
        
        return output_type, latent_dim
    
    def run_stage1_training(self, top_n_per_type=100, resolution=0.5, batch_size=256, n_epochs=100, 
                           lr=1e-3, beta=1.0, hidden_dims=[512, 256], latent_dim=128, loss_type='mse', 
                           lambda_mmd=0.0, pretrained_path=None, celltype_key=None, precomputed_marker_file=None):
        """Run stage 1 training: VAE on SC + ST with marker genes
        
        Args:
            top_n_per_type: Number of marker genes per cluster/celltype
            resolution: Leiden resolution (for auto-clustering, ignored if celltype_key provided)
            batch_size: Training batch size
            n_epochs: Number of training epochs
            lr: Learning rate
            beta: KL divergence weight
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            loss_type: 'mse' or 'zinb'
            lambda_mmd: MMD loss weight
            pretrained_path: Path to pretrained checkpoint
            celltype_key: Column name in sc_adata.obs for celltype annotation.
                         If None, will auto-cluster using Leiden.
                         If provided, will use existing celltype annotation.
            precomputed_marker_file: Path to precomputed marker genes file.
                                   If provided, will load marker genes directly from this file
                                   instead of computing them from scratch.
        """
        print("="*60)
        print("Stage 1 Training: VAE (SC + ST, Marker Genes)")
        print("="*60)
        print(f"Configuration:")
        print(f"   Marker genes per type: {top_n_per_type}")
        print(f"   Clustering mode: {'Celltype' if celltype_key else 'Auto-cluster (Leiden)'}")
        if celltype_key:
            print(f"   Celltype column: {celltype_key}")
        else:
            print(f"   Leiden resolution: {resolution}")
        if precomputed_marker_file:
            print(f"   Precomputed marker genes: {precomputed_marker_file}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Beta (KL weight): {beta}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Loss type: {loss_type.upper()}")
        print(f"   Lambda MMD: {lambda_mmd}")
        if pretrained_path:
            print(f"   Pretrained: {pretrained_path}")
        print("="*60)
        
        # 1. Load data
        sc_adata, st_adata = self.load_data()
        
        # 2. Prepare data based on marker genes
        train_X, test_X, train_modality, test_modality, y_train, y_test, sc_X_full_train_count = self.prepare_marker_gene_data(
            sc_adata, st_adata, top_n_per_type=top_n_per_type, resolution=resolution, 
            celltype_key=celltype_key, precomputed_marker_file=precomputed_marker_file
        )
        
        # 3. Build or load VAE
        input_dim = len(self.genes)
        
        if pretrained_path and os.path.exists(pretrained_path):
            # Load pretrained weights
            pretrained_output_type, pretrained_latent_dim = self.load_pretrained(pretrained_path)
            
            # Verify architecture compatibility
            if input_dim != self.vae.encoder.encoder[0].in_features:
                print(f"   Warning: Input dim mismatch! Pretrained: {self.vae.encoder.encoder[0].in_features}, Current: {input_dim}")
                print(f"   Rebuilding VAE from scratch...")
                self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, loss_type=loss_type)
            else:
                print(f"   Using pretrained VAE architecture")
                # Update loss_type from pretrained if not specified
                if loss_type == 'mse' and pretrained_output_type != 'mse':
                    print(f"   Note: Pretrained model uses {pretrained_output_type}, current setting is {loss_type}")
        else:
            # Build from scratch
            self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, loss_type=loss_type)
        
        # 4. Train VAE
        best_loss = self.train_vae(train_X, test_X, train_modality, test_modality,
                                  batch_size=batch_size, n_epochs=n_epochs, lr=lr, beta=beta, 
                                  loss_type=loss_type, lambda_mmd=lambda_mmd)
        
        # Save training data for cluster center computation
        self.train_X = train_X
        self.train_modality = train_modality  
        self.y_train = y_train
        self.sc_X_full_train_count = sc_X_full_train_count  # Full gene SC training data (count, for reconstruction)
        
        # 5. Compute and save cluster centers
        print("="*60)
        print("Computing cluster centers...")
        
        # Use training data to compute cluster centers (already preprocessed with marker genes)
        sc_train_mask = train_modality == 0
        sc_train_data = train_X[sc_train_mask]
        sc_train_labels = y_train
        
        print(f"   SC training data: {sc_train_data.shape}")
        print(f"   Number of clusters: {len(np.unique(sc_train_labels))}")
        
        # Use trained VAE to compute embeddings
        self.vae.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(sc_train_data), batch_size):
                batch_data = sc_train_data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                
                # Get latent representation
                mu, log_var = self.vae.encoder(batch_tensor)
                all_embeddings.append(mu.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
        
        # Compute cluster centers and expressions
        cluster_prototypes = {}
        cluster_expressions = {}  # Count version for training
        cluster_expressions_full_count = {}  # Count version (all genes)
        
        for cluster_id in np.unique(sc_train_labels):
            cluster_mask = sc_train_labels == cluster_id
 
            # Compute cluster center (latent space)
            cluster_center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_prototypes[cluster_id] = cluster_center
            
            # Compute cluster expression (marker genes, count)
            cluster_expression = np.mean(sc_train_data[cluster_mask], axis=0)
            cluster_expressions[cluster_id] = cluster_expression
        
        # Compute full gene expressions (count version)
        print("   Computing full gene cluster expressions...")
        print(f"      Total genes: {len(self.all_genes)}")
        
        for cluster_id in np.unique(sc_train_labels):
            cluster_mask = sc_train_labels == cluster_id
            # Count version (for reconstruction)
            cluster_expr_full_count = np.mean(sc_X_full_train_count[cluster_mask], axis=0)
            cluster_expressions_full_count[cluster_id] = cluster_expr_full_count
        
        # Save cluster centers and expressions
        self.cluster_prototypes = cluster_prototypes
        self.cluster_expressions = cluster_expressions
        self.cluster_expressions_full = cluster_expressions_full_count  # Use count version
        self.cluster_expressions_full_count = cluster_expressions_full_count  # Also keep this for backward compatibility
        print(f"   Completed: {len(cluster_prototypes)} clusters with center and expressions (all genes)")
        
        # 6. Plot UMAP for modality alignment visualization
        print("="*60)
        print("Visualizing modality alignment...")
        self.plot_modality_alignment_umap(train_X, train_modality, y_train)
        
        self.save_vae(f"{self.output_dir}/final_vae.pth")
        
        return {
            'best_loss': best_loss,
            'n_genes': len(self.genes),
            'n_clusters': len(self.label_encoder.classes_),
            'model_path': f"{self.output_dir}/final_vae.pth",
            'clusters': list(self.label_encoder.classes_)
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
    
    # Clustering arguments
    parser.add_argument('--celltype_key', type=str, default=None,
                       help='Column name in SC adata.obs for celltype annotation. '
                            'If provided, uses existing celltype instead of auto-clustering (e.g., "cell_type")')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution (only used if celltype_key is not provided)')
    
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
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight (beta-VAE)')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'zinb'],
                       help='Reconstruction loss type: mse (default) or zinb')
    parser.add_argument('--lambda_mmd', type=float, default=0.0,
                       help='MMD loss weight for modality alignment (0=disabled, 1.0=recommended)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained VAE model to continue training')
    parser.add_argument('--precomputed_marker_file', type=str, default=None,
                       help='Path to precomputed marker genes file. If provided, '
                            'marker genes will be loaded directly from this file instead of computing them.')
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    

    # Create VAE encoder
    co_encoder = coEncoder(
        sc_file=args.sc_file,
        st_file=args.st_file,
        output_dir=args.output_dir,
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
        celltype_key=args.celltype_key,
        precomputed_marker_file=args.precomputed_marker_file
    )
    
if __name__ == "__main__":
    main()