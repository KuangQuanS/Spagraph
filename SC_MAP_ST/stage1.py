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
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import unified model definitions
from model import VAE, vae_loss_function

def compute_clusters_and_marker_genes(adata, top_n=100, min_fold_change=1.5, resolution=0.5, save_path=None):
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
    sc.pp.log1p(adata_full)
    
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
    
    print(f"Total: {len(marker_genes)} marker genes")
    
    # Return clustering info, marker genes, and full adata for annotation
    return sorted(list(marker_genes)), adata_full.obs['leiden'].copy(), adata_full

#============================================================
# Main Module
#============================================================
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
        
        print(f"Using device: {self.device}")

        # Model components
        self.vae = None
        self.label_encoder = None
        self.marker_genes = None
        
    def load_data(self) -> Tuple[ad.AnnData, ad.AnnData, List[str]]:

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
            sc_file = os.path.join(sample_dir, f"{sample}_SC.h5ad")
            st_file = os.path.join(sample_dir, f"{sample}_ST.h5ad")
            
            if os.path.exists(sc_file) and os.path.exists(st_file):
                print(f"   Loading {sample}...")
                
                # Load SC data
                sc_adata = sc.read_h5ad(sc_file)
                sc_adata.obs['sample'] = sample
                sc_adata.obs['modality'] = 'SC'
                
                # Load ST data
                st_adata = sc.read_h5ad(st_file)
                st_adata.obs['sample'] = sample
                st_adata.obs['modality'] = 'ST'
                
                print(f"   SC: {sc_adata.shape}")
                print(f"   ST: {st_adata.shape}")
                
                sc_data_list.append(sc_adata)
                st_data_list.append(st_adata)
                valid_samples.append(sample)
            else:
                print(f"   Complete data not found: {sample}")
        
        # Merge SC data - use inner join to keep common genes
        print(f"   Merging {len(sc_data_list)} SC samples...")
        combined_sc = ad.concat(sc_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        
        # Merge ST data - use inner join to keep common genes  
        print(f"   Merging {len(st_data_list)} ST samples...")
        combined_st = ad.concat(st_data_list, axis=0, join='inner', 
                                keys=valid_samples, index_unique='-')
        
        print(f"   SC total: {combined_sc.shape}")
        print(f"   ST total: {combined_st.shape}")
        # print(f"   Cell types: {combined_sc.obs['cell_type'].unique()}")
        
        return combined_sc, combined_st, valid_samples

    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100, resolution: float = 0.5) -> Tuple:
        """Prepare training data based on marker genes"""

        # 1. Compute clusters and marker genes  
        print("="*60)
        print("Computing clusters and marker genes...")
        cluster_save_path = f"{self.output_dir}/marker_genes.txt"
        self.marker_genes, sc_clusters, sc_adata_clustered = compute_clusters_and_marker_genes(
            sc_adata.copy(), 
            top_n=top_n_per_type, 
            resolution=resolution,
            save_path=cluster_save_path
        )
        
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
                
        # SC normalization - process all genes first
        sc_adata_normalized = sc_adata.copy()
        sc.pp.normalize_total(sc_adata_normalized, target_sum=1e4)
        sc.pp.log1p(sc_adata_normalized)
        
        # Extract full gene expression for later use
        sc_X_full = sc_adata_normalized.X.toarray() if hasattr(sc_adata_normalized.X, 'toarray') else sc_adata_normalized.X
        sc_all_genes = list(sc_adata_normalized.var.index)
        
        # Extract marker genes
        sc_subset = sc_adata_normalized[:, sc_adata_normalized.var.index.isin(self.marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_clusters.values
        print("="*60)
        print(f"SC data min: {np.min(sc_X)}, max: {np.max(sc_X)}")
        print("="*60)
        # Encode labels
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"   SC data: {sc_X.shape}")
        print(f"   Number of clusters: {len(self.label_encoder.classes_)}")

        # ST
        available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
        st_subset = st_adata[:, available_genes].copy()
        
        sc.pp.log1p(st_subset)
        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        print("="*60)
        print(f"ST data min: {np.min(st_X)}, max: {np.max(st_X)}")
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
        sc_X_full_train = sc_X_full[sc_train_idx]
        
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

        return train_X, test_X, train_modality, test_modality, y_train, y_test, sc_X_full_train
    
    def build_vae(self, input_dim: int, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        """Build VAE model"""
        print("="*60)
        print("Building VAE model...")
        
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        ).to(self.device)
        
        print(f"   Input: {input_dim} -> Latent: {latent_dim}")
        print(f"   Hidden layers: {hidden_dims}")
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"   Parameters: {vae_params:,}")
    
    def train_vae(self, train_X, test_X, train_modality, test_modality,
                  batch_size=256, n_epochs=100, lr=1e-3, beta=1.0):
        """Train VAE"""

        print("="*60)
        print("Starting VAE training...")
        print(f"   Train data: {train_X.shape} (SC: {sum(train_modality==0)}, ST: {sum(train_modality==1)})")
        print(f"   Test data: {test_X.shape} (SC: {sum(test_modality==0)}, ST: {sum(test_modality==1)})")

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
            
            for batch_data, batch_modality in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE forward pass
                recon_data, mu, log_var, z = self.vae(batch_data)
                
                # Compute loss
                total_loss, recon_loss, kl_div = vae_loss_function(
                    recon_data, batch_data, mu, log_var, beta=beta
                )
                
                # Normalize loss
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
            
            # Evaluate
            if (epoch + 1) % 5 == 0:
                test_loss = self.evaluate_vae(test_loader, beta)
                test_losses.append(test_loss)
                
                scheduler.step(test_loss)
                
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
        self.plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses)
        
        return best_loss
    
    def evaluate_vae(self, test_loader, beta=1.0):
        """Evaluate VAE"""
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
        """Plot VAE training curves"""
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
        ax4.set_title('Loss Components')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/vae_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_vae(self, filepath):
        """Save VAE model"""
        # Check if cluster info exists
        cluster_prototypes = getattr(self, 'cluster_prototypes', None)
        cluster_expressions = getattr(self, 'cluster_expressions', None)
        cluster_expressions_full = getattr(self, 'cluster_expressions_full', None)
        
        print("="*60)
        print(f"Saving model to: {filepath}")
        if cluster_prototypes is not None:
            print(f"   Cluster centers: {len(cluster_prototypes)} clusters")
        else:
            print(f"   Warning: cluster centers missing")
            
        if cluster_expressions is not None:
            print(f"   Cluster expressions (marker genes): {len(cluster_expressions)} clusters")
        else:
            print(f"   Warning: cluster expressions missing")
        
        if cluster_expressions_full is not None:
            print(f"   Cluster expressions (all genes): {len(cluster_expressions_full)} clusters")
        else:
            print(f"   Warning: full gene expressions missing")
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'input_dim': len(self.genes),
            'latent_dim': self.vae.latent_dim,
            'sc_clusters': getattr(self, 'sc_clusters', None),
            'resolution': getattr(self, 'resolution', 0.5),
            'cluster_prototypes': cluster_prototypes,
            'cluster_expressions': cluster_expressions,
            'cluster_expressions_full': cluster_expressions_full,
            'all_genes': getattr(self, 'all_genes', None)
        }, filepath)
    
    def load_vae(self, filepath):
        """Load VAE model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        input_dim = checkpoint['input_dim']
        latent_dim = checkpoint['latent_dim']
        
        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        
        self.label_encoder = checkpoint['label_encoder']
        self.marker_genes = checkpoint['marker_genes']
        self.genes = checkpoint['genes']
        
        print(f"VAE model loaded: {filepath}")
    
    def run_stage1_training(self, top_n_per_type=100, resolution=0.5, batch_size=256, n_epochs=100, 
                           lr=1e-3, beta=1.0, hidden_dims=[512, 256], latent_dim=128):
        """Run stage 1 training: VAE on SC + ST with marker genes"""
        print("="*60)
        print("Stage 1 Training: VAE (SC + ST, Marker Genes)")
        print("="*60)
        print(f"Configuration:")
        print(f"   Marker genes per type: {top_n_per_type}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Beta (KL weight): {beta}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Latent dim: {latent_dim}")
        print("="*60)
        
        # 1. Load data
        sc_adata, st_adata, samples = self.load_data()
        
        # 2. Prepare data based on marker genes
        train_X, test_X, train_modality, test_modality, y_train, y_test, sc_X_full_train = self.prepare_marker_gene_data(
            sc_adata, st_adata, top_n_per_type=top_n_per_type, resolution=resolution
        )
        
        # 3. Build VAE
        input_dim = len(self.genes)
        self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        
        # 4. Train VAE
        best_loss = self.train_vae(train_X, test_X, train_modality, test_modality,
                                  batch_size=batch_size, n_epochs=n_epochs, lr=lr, beta=beta)
        
        # Save training data for cluster center computation
        self.train_X = train_X
        self.train_modality = train_modality  
        self.y_train = y_train
        self.sc_X_full_train = sc_X_full_train  # Full gene SC training data
        
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
        cluster_expressions = {}
        cluster_expressions_full = {}
        
        for cluster_id in np.unique(sc_train_labels):
            cluster_mask = sc_train_labels == cluster_id
 
            # Compute cluster center (latent space)
            cluster_center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_prototypes[cluster_id] = cluster_center
            
            # Compute cluster expression (marker genes)
            cluster_expression = np.mean(sc_train_data[cluster_mask], axis=0)
            cluster_expressions[cluster_id] = cluster_expression
        
        # Compute full gene expressions
        print("   Computing full gene cluster expressions...")
        print(f"      Total genes: {len(self.all_genes)}")
        
        # Use sc_X_full_train (already normalized and log1p transformed)
        for cluster_id in np.unique(sc_train_labels):
            cluster_mask = sc_train_labels == cluster_id
            cluster_expr_full = np.mean(sc_X_full_train[cluster_mask], axis=0)
            cluster_expressions_full[cluster_id] = cluster_expr_full
        
        # Save cluster centers and expressions
        self.cluster_prototypes = cluster_prototypes
        self.cluster_expressions = cluster_expressions
        self.cluster_expressions_full = cluster_expressions_full
        print(f"   Completed: {len(cluster_prototypes)} clusters with center and expressions (all genes)")
        df_marker = pd.DataFrame.from_dict(
            self.cluster_expressions, orient='index', columns=self.genes
        )
        df_marker.index.name = 'cluster_id'
        df_marker.to_csv(f"{self.output_dir}/cluster_marker_expressions.csv")
        df_full = pd.DataFrame.from_dict(
            self.cluster_expressions_full, orient='index', columns=self.all_genes
        )
        df_full.index.name = 'cluster_id'
        df_full.to_csv(f"{self.output_dir}/cluster_full_expressions.csv")
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
    """Main function"""
    parser = argparse.ArgumentParser(description='Stage 1: VAE Training for SC-ST Integration')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default="/home/maweicheng/ST_Graduation_Project/database",
                       help='Data directory path')
    parser.add_argument('--output_dir', type=str, default="./stage1_results",
                       help='Output directory path')
    
    # Model arguments
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='Marker genes per cluster')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution')
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
    
    # Device argument
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    

    # Create VAE encoder
    co_encoder = coEncoder(
        data_dir=args.data_dir,
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
        latent_dim=args.latent_dim
    )
    
if __name__ == "__main__":
    main()