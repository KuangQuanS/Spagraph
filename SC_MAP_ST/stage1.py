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

# Import unified model definitions
from model import VAE, vae_loss_function
from stage1_utils import compute_clusters_and_marker_genes, load_data
# Main module
class coEncoder:
    def __init__(self, 
                 data_dir="/home/maweicheng/ST_Graduation_Project/database/Wu",
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
        
    def prepare_marker_gene_data(self, sc_adata: ad.AnnData, st_adata: ad.AnnData, 
                               top_n_per_type: int = 100, resolution: float = 0.5) -> Tuple:
        """Prepare training data based on marker genes"""

        # 1. Compute clustering and marker genes
        print("Computing clusters and marker genes...")
        cluster_save_path = f"{self.output_dir}/marker_genes.txt"

        self.marker_genes, sc_clusters = compute_clusters_and_marker_genes(
            sc_adata.copy(), 
            top_n=top_n_per_type, 
            resolution=resolution,
            save_path=cluster_save_path
        )
        
        self.sc_clusters = sc_clusters
        self.resolution = resolution
        
        # 2. Process SC data (extract marker genes and normalize)
        print("Processing SC data...")
        
        # SC normalization
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.log1p(sc_adata)
        sc_data_full = sc_adata.copy()

        sc_subset = sc_adata[:, sc_adata.var.index.isin(self.marker_genes)].copy()
        sc_X = sc_subset.X.toarray() if hasattr(sc_subset.X, 'toarray') else sc_subset.X
        sc_labels = sc_clusters.values  # Use clustering labels
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        sc_y = self.label_encoder.fit_transform(sc_labels)
        
        print(f"  SC data: {sc_X.shape}")
        print(f"  Number of clusters: {len(self.label_encoder.classes_)}")
        
        # 3. Process ST data
        print("Processing ST data...")
        available_genes = [g for g in self.marker_genes if g in st_adata.var.index]
        
        
        sc.pp.log1p(st_adata)

        st_subset = st_adata[:, available_genes].copy()

        st_X = st_subset.X.toarray() if hasattr(st_subset.X, 'toarray') else st_subset.X
        
        print(f"  ST data: {st_X.shape}, available genes: {len(available_genes)}/{len(self.marker_genes)}")
        
        # 4. Ensure SC and ST feature dimensions match
        final_genes = [g for g in available_genes if g in sc_subset.var.index]
        
        sc_gene_indices = [list(sc_subset.var.index).index(g) for g in final_genes]
        st_gene_indices = [list(st_subset.var.index).index(g) for g in final_genes]
        
        sc_X_final = sc_X[:, sc_gene_indices]
        st_X_final = st_X[:, st_gene_indices]
        
        print(f"  Final genes: {len(final_genes)}")
        
        # 5. Data splitting
        # sc_y ensure every cluster is represented
        sc_train, sc_test, y_train, y_test = train_test_split(
            sc_X_final, sc_y, test_size=0.2, stratify=sc_y, random_state=42
        )
        
        st_train, st_test = train_test_split(
            st_X_final, test_size=0.2, random_state=42
        )
        
        # 6. Merge training and test sets
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
        
        print(f"  Training set: {train_X.shape} (SC: {len(sc_train)}, ST: {len(st_train)})")
        print(f"  Test set: {test_X.shape} (SC: {len(sc_test)}, ST: {len(st_test)})")
        
        # Save gene list
        self.genes = final_genes
        genes_file = f"{self.output_dir}/marker_genes.txt"
        with open(genes_file, 'w') as f:
            for gene in self.genes:
                f.write(f"{gene}\n")

        return train_X, test_X, train_modality, test_modality, sc_train, sc_test, y_train, y_test, sc_data_full
    
    def build_vae(self, input_dim: int, hidden_dims=[512, 256], latent_dim=128, dropout=0.2):
        """Build VAE model"""
        print("Building VAE model...")
        
        self.vae = VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        ).to(self.device)
        
        print(f"  VAE: {input_dim} -> {latent_dim}")
        print(f"  Hidden dimensions: {hidden_dims}")
        vae_params = sum(p.numel() for p in self.vae.parameters())
        print(f"  Parameters: {vae_params:,}")
    
    def train_vae(self, train_X, test_X, train_modality, test_modality,
                  batch_size=256, n_epochs=100, lr=1e-3, beta=1.0):
        """Train VAE"""

        print("Starting VAE training...")
        print(f"  Training data: {train_X.shape} (SC: {sum(train_modality==0)}, ST: {sum(train_modality==1)})")
        print(f"  Test data: {test_X.shape} (SC: {sum(test_modality==0)}, ST: {sum(test_modality==1)})")

        class SimpleDataset(Dataset):
            def __init__(self, X, modality):
                self.X = torch.FloatTensor(X)
                self.modality = torch.LongTensor(modality)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.modality[idx]

        # Data loaders
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
        
        for epoch in range(n_epochs):
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
            
            # Evaluation
            if (epoch + 1) % 5 == 0:
                test_loss = self.evaluate_vae(test_loader, beta)
                test_losses.append(test_loss)
                
                scheduler.step(test_loss)
                
                print(f"Epoch {epoch+1:3d}: Train Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, "
                      f"KL={avg_kl:.4f}), Test Loss={test_loss:.4f}")
                
                # Save best model
                if test_loss < best_loss:
                    best_loss = test_loss
                    # Don't save best_vae.pth here, save after computing cluster centers
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training curves
        self.plot_vae_training_curves(train_losses, test_losses, recon_losses, kl_losses)
        
        print(f"VAE training completed! Best test loss: {best_loss:.4f}")
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
        
        # Loss components
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
        # Check if clustering info exists
        cluster_prototypes = getattr(self, 'cluster_prototypes', None)
        cluster_expressions = getattr(self, 'cluster_expressions', None)
        cluster_expressions_full = getattr(self, 'cluster_expressions_full', None)  # Full genes version
        
        print(f"Saving model to: {filepath}")
        if cluster_prototypes is not None:
            print(f"  Contains cluster centers: {len(cluster_prototypes)} clusters")
        else:
            print(f"  Warning: Missing cluster centers")
            
        if cluster_expressions is not None:
            print(f"  Contains cluster expression (marker genes): {len(cluster_expressions)} clusters")
        else:
            print(f"  Warning: Missing cluster expression")
        
        if cluster_expressions_full is not None:
            print(f"  Contains cluster expression (all genes): {len(cluster_expressions_full)} clusters")
        else:
            print(f"  Warning: Missing full-gene cluster expression")
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'label_encoder': self.label_encoder,
            'marker_genes': self.marker_genes,
            'genes': self.genes,
            'input_dim': len(self.genes),
            'latent_dim': self.vae.latent_dim,
            'sc_clusters': getattr(self, 'sc_clusters', None),  # Save clustering info
            'resolution': getattr(self, 'resolution', 0.5),    # Save resolution parameter
            'cluster_prototypes': cluster_prototypes,  # Save cluster centers
            'cluster_expressions': cluster_expressions,  # Save cluster expression (marker genes)
            'cluster_expressions_full': cluster_expressions_full,  # Save cluster expression (all genes)
            'all_genes': self.all_genes  # Save all genes list
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
        """Execute Stage 1 VAE training"""
        print("Stage 1: VAE Training (SC + ST, Marker genes)")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Marker genes per cluster: {top_n_per_type}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Epochs: {n_epochs}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Beta (KL weight): {beta}")
        print(f"  - Hidden dimensions: {hidden_dims}")
        print(f"  - Latent dimension: {latent_dim}")
        print("="*60)
        
        # 1. Load data
        sc_adata, st_adata, samples = load_data(self.data_dir)
        print("="*60)
        # 2. Prepare data based on marker genes
        train_X, test_X, train_modality, test_modality, sc_train, sc_test, y_train, y_test, sc_adata_full = self.prepare_marker_gene_data(
            sc_adata, st_adata, top_n_per_type=top_n_per_type, resolution=resolution
        )
        print("="*60)
        # 3. Build VAE
        input_dim = len(self.genes)
        self.build_vae(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        print("="*60)
        # 4. Train VAE
        best_loss = self.train_vae(train_X, test_X, train_modality, test_modality,
                                  batch_size=batch_size, n_epochs=n_epochs, lr=lr, beta=beta)
        print("="*60)
        # 5. Compute and save cluster centers
        print("Computing cluster centers...")
        
        # sc_all_data = np.vstack([sc_train, sc_test])
        # sc_all_labels = np.concatenate([y_train, y_test])

        sc_all_data = sc_train
        sc_all_labels = y_train
        print(f"  SC data for cluster computation (all): {sc_all_data.shape}")
        print(f"  Number of clusters: {len(np.unique(sc_all_labels))}")
        
        # Load original SC data (all genes) for full-gene expression computation
        print("  Loading original SC data (for full-gene expression)...")
        # sc_adata_full = load_data(self.data_dir)[0].copy()
        self.all_genes = list(sc_adata_full.var.index)
        print(f"    Total genes: {len(self.all_genes)}")
        
        # Perform standard preprocessing (normalization and log transform)

        sc_full_X = sc_adata_full.X.toarray() if hasattr(sc_adata_full.X, 'toarray') else sc_adata_full.X
        
        # Compute embeddings using trained VAE
        self.vae.eval()
        with torch.no_grad():
            # Batch processing to avoid memory issues
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(sc_all_data), batch_size):
                batch_data = sc_all_data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)
                
                # Get latent representation
                mu, log_var = self.vae.encoder(batch_tensor)
                all_embeddings.append(mu.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
        
        # Compute cluster centers and expression profiles
        print("="*60)
        print("  Computing cluster centers and expression profiles...")
        
        # Pre-compute marker gene indices (avoid repeated computation)
        marker_gene_indices = [self.all_genes.index(g) for g in self.genes]
        
        cluster_prototypes = {}
        cluster_expressions = {}
        cluster_expressions_full = {}  # Full genes version
        
        for cluster_id in np.unique(sc_all_labels):

            cluster_mask = sc_all_labels == cluster_id
            # calculate number of cells in the cluster
            cluster_cells = np.sum(cluster_mask)
            
            # Compute cluster center (latent space)
            cluster_center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_prototypes[cluster_id] = cluster_center
            
            # Compute full-gene expression first
            cluster_indices = np.where(sc_all_labels == cluster_id)[0]
            cluster_cells_full_expr = sc_full_X[cluster_indices]
            cluster_expr_full = np.mean(cluster_cells_full_expr, axis=0)
            cluster_expressions_full[cluster_id] = cluster_expr_full
            
            # Extract marker genes expression from full-gene expression
            cluster_expression = cluster_expr_full[marker_gene_indices]
            cluster_expressions[cluster_id] = cluster_expression
            
            print(f"    Cluster {cluster_id}: {cluster_cells} cells, expr_mean={cluster_expr_full.mean():.6f}, marker_expr_mean={cluster_expression.mean():.6f}")
        
        # Save cluster centers and expression profiles
        self.cluster_prototypes = cluster_prototypes
        self.cluster_expressions = cluster_expressions
        self.cluster_expressions_full = cluster_expressions_full  # Full genes version
        print(f"  Completed: {len(cluster_prototypes)} cluster centers and expression profiles (all genes included)")
        print("="*60)
        # 6. Save final model
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
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default="None",
                       help='Data directory containing sample folders with *SC.h5ad and *ST.h5ad files')
    parser.add_argument('--output_dir', type=str, default="./stage1_results",
                       help='Output directory path')
    
    # Model parameters
    parser.add_argument('--top_n_per_type', type=int, default=100,
                       help='Number of marker genes per cluster')
    parser.add_argument('--resolution', type=float, default=0.5,
                       help='Leiden clustering resolution')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='VAE hidden layer dimensions')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='VAE latent space dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight (beta-VAE)')
    
    # Device parameter
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu, None for auto-select)')
    
    args = parser.parse_args()
    

    # Create VAE encoder
    co_encoder = coEncoder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run Stage 1 VAE training
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

    print("="*60)
    print("Stage 1 Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("="*60)
    print("Stage 1 training completed successfully!")

    
if __name__ == "__main__":
    main()